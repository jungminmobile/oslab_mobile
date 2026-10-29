"""
s2_profile_ort.py — ORT 연산별 시간 분해: 왜 S8 후보는 4스레드에서 FP32보다 느린가 (G3 지연 원인 분석)
==================================================================================================

[배경] 지연 최종 측정(results/siglip2/G3_cumulative/speed, 10/14 18:19): 4스레드에서 S8 후보(fc2L2 등)는 FP32보다 느리고
  (+2.5~+14.7%), U8은 모든 스레드에서 FP32의 약 절반이다.
[가설] (실행 전에 정함)
  H1 변환 오버헤드: S8에서는 MatMul 13개(실행 후 확인: QKV 투영 12 + 헤드 kv, attention 행렬곱 아님)가 정수로 합쳐지지 않아 QuantizeLinear·DequantizeLinear가 많이 남고,
     이 변환 시간은 스레드를 늘려도 잘 줄지 않는다 → 4스레드에서 Q/DQ 시간 비중이 커진다.
  H2 커널 확장성: 정수 행렬곱(QLinearMatMul·MatMulInteger) 자체가 FP32 MatMul보다 스레드 확장성이 낮다
     → 1→4스레드에서 정수 행렬곱 시간이 FP32 MatMul보다 덜 준다.
[방법] ORT_ENABLE_ALL 세션에 enable_profiling을 켜고, 배치 1로 워밍업 10장 후 30장을 추론한다.
  프로파일 JSON의 Node 이벤트(kernel 시간)를 연산 종류별로 합해, 이미지 1장당 ms와 비율을 구한다.
  분류: int_matmul(QLinearMatMul·MatMulInteger·DynamicQuantizeMatMul 등), float_matmul(MatMul·FusedMatMul·Gemm),
        qdq(QuantizeLinear·DequantizeLinear), 나머지 연산 이름 그대로.
  ※ 프로파일링은 자체 오버헤드가 있어 절대 지연은 bench_speed 값과 다르다. 비교는 같은 조건의 비율로 한다.
[사용]  (공용 잠금 + PC가 조용할 때)
  RN50_RESULTS=../results/siglip2/G3_cumulative/profile "$SIGLIP2_PYTHON" s2_profile_ort.py --threads 1 4 \
      --models fp32=siglip2_image_fp32.onnx fc2L2=<...>_fc2L2.onnx fc2L2_u8=<...>_u8_..._fc2L2.onnx
"""
import argparse
import collections
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
from scripts import env_info  # noqa: E402

RC = S.RC
INT_MM = ("QLinearMatMul", "MatMulInteger", "DynamicQuantizeMatMul", "MatMulIntegerToFloat", "QGemm", "QLinearConv")
FLOAT_MM = ("MatMul", "FusedMatMul", "Gemm")
QDQ = ("QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear")


def group(op):
    if op in INT_MM:
        return "int_matmul"
    if op in FLOAT_MM:
        return "float_matmul"
    if op in QDQ:
        return "qdq"
    return op


def profile(path, threads, x, warmup, n):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    tmp = tempfile.mkdtemp()
    so.enable_profiling = True
    so.profile_file_prefix = os.path.join(tmp, "prof")
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for i in range(warmup):
        sess.run(None, {name: x[i:i + 1]})
    t0 = time.perf_counter()
    for i in range(warmup, warmup + n):
        sess.run(None, {name: x[i:i + 1]})
    wall = (time.perf_counter() - t0) / n * 1000
    pf = sess.end_profiling()
    ev = json.load(open(pf, encoding="utf-8"))
    runs = [e for e in ev if e.get("cat") == "Session" and e.get("name") == "model_run"]
    assert len(runs) == warmup + n, (len(runs), warmup + n)
    t_start = runs[warmup]["ts"]                                   # 워밍업 뒤 첫 run 시작 이후의 Node 이벤트만
    by_op, by_grp, cnt = collections.Counter(), collections.Counter(), collections.Counter()
    for e in ev:
        if e.get("cat") != "Node" or e["ts"] < t_start or not e["name"].endswith("_kernel_time"):
            continue
        op = e.get("args", {}).get("op_name", "?")
        by_op[op] += e["dur"]
        by_grp[group(op)] += e["dur"]
        cnt[op] += 1
    os.remove(pf)
    total = sum(by_grp.values())
    per_img = lambda d: {k: round(v / n / 1000, 2) for k, v in sorted(d.items(), key=lambda kv: -kv[1])}
    return {"threads": threads, "wall_ms_per_img_profiled": round(wall, 1), "kernel_ms_per_img": round(total / n / 1000, 1),
            "by_group_ms": per_img(by_grp), "by_group_frac": {k: round(v / total, 3) for k, v in sorted(by_grp.items(), key=lambda kv: -kv[1])},
            "by_op_ms": per_img(by_op), "node_calls_per_img": {k: v // n for k, v in cnt.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="이름=파일")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    x = np.ascontiguousarray(np.load(RC.CACHE_DIR / "pets_dev_x.npy", mmap_mode="r")[:a.warmup + a.n]).astype(np.float32)
    rows = []
    for t in a.threads:
        for spec in a.models:
            nm, f = spec.split("=", 1)
            r = {"model": nm, "file": f, **profile(RC.onnx_path(f), t, x, a.warmup, a.n)}
            rows.append(r)
            print(f"t={t} {nm:10s} wall {r['wall_ms_per_img_profiled']:7.1f} ms  kernel {r['kernel_ms_per_img']:7.1f} ms  "
                  + "  ".join(f"{k}={v}" for k, v in list(r['by_group_ms'].items())[:6]), flush=True)
    res = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"], "dirty": env["git"]["dirty"],
           "warmup": a.warmup, "n": a.n, "data": f"pets_dev[0:{a.warmup + a.n}]", **env_info.flat(env), "rows": rows}
    with open(out / "profile.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
