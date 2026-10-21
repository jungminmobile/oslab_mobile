"""
s2_why_after.py — G8 E4: 회전·M2가 거대 활성값을 실제로 없앴는지 수치로 확인 (2026-10-21, 보고서 6.6 검토 지적)
==========================================================================================================
[질문] 원인 분석(G3_why)은 "거대 값 때문에 8비트 눈금이 넓어져 보통 값의 84~99%가 0으로 반올림된다"였다.
       최종 모델(회전 + M2)에서 이 비율이 실제로 줄었는가? 회전만·M2만이면 어떤가? (2×2와 같은 4개 FP32 그래프)
[방법] FP32 ONNX 4개(원래 / 회전만 / M2만 / 회전+M2)의 **모든 MatMul 입력 활성값**을 꺼내(그래프 출력으로 추가)
       dev 64장(Pets·Caltech 앞 32장씩, G3_why와 같은 사진)으로 잰다. 텐서마다:
  - 범위: 64장 전체의 최솟값·최댓값 (MinMax 보정과 같은 텐서당 눈금 1개, UINT8 비대칭) → 눈금 scale = (max − min) / 255
  - **0으로 반올림되는 비율**: |x| < scale/2 인 원소 비율 (원래부터 0인 값은 거의 없음 — 참고로 |x| < 1e-6 비율도 기록)
  - 쏠림: 사진마다 max|x| / mean|x| 를 내서 평균 (GELU 출력은 중앙값이 0에 가까워 중앙값 대신 평균)
  배포 모델의 보정은 Percentile 99.999이지만 여기서는 원인 분석과 같은 MinMax 눈금으로 비교한다 (그래프끼리 같은 기준).
[출력] $RN50_RESULTS/why_after.jsonl (그래프 × MatMul 입력마다 한 줄) + 요약 표 출력 (fc2 입력 블록별, 전체 평균)
[사용] RN50_ONNX=<FP32 4개가 있는 폴더> RN50_RESULTS=../results/siglip2/G8_candidates/b16/E4_why_after \
       "$SIGLIP2_PYTHON" s2_why_after.py --models 원래=… 회전만=… M2만=… 회전+M2=…
"""
import argparse
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
from scripts import env_info  # noqa: E402

RC = S.RC
SETS = ["pets_dev", "caltech_dev"]


def matmul_inputs(model):
    """MatMul 노드의 첫 번째 입력(활성값) 이름과 노드 이름. 가중치(initializer)가 첫 입력인 경우는 없다."""
    inits = {i.name for i in model.graph.initializer}
    out = []
    for n in model.graph.node:
        if n.op_type == "MatMul" and n.input[0] not in inits:
            out.append((n.name, n.input[0]))
    return out


def role(node_name):
    m = re.search(r"blocks\.(\d+)/(attn/qkv|attn/proj|attn/MatMul_1|attn/MatMul|mlp/fc1|mlp/fc2)", node_name)
    if m:
        return f"b{m.group(1)}:{m.group(2).replace('attn/', '')}"
    m = re.search(r"/(mlp/fc[12]|attn/(qkv|proj))(_\d+)?/MatMul", node_name)    # M 블록(블록 5~11)은 이름에 블록 번호가 없다
    if m:
        return f"m:{m.group(1).replace('attn/', '')}{m.group(3) or ''}"
    return "other:" + node_name.split("/")[-2] if "/" in node_name else node_name


def run(path, x, threads):
    import onnx
    import onnxruntime as ort
    m = onnx.load(str(path))
    mins = matmul_inputs(m)
    have = {o.name for o in m.graph.output}
    for _, t in mins:
        if t not in have:
            m.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    tmp = Path(tempfile.mkdtemp()) / "with_outputs.onnx"
    onnx.save(m, str(tmp), save_as_external_data=True, location="w.bin")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL   # 중간 텐서 이름을 그대로 두기 위해 (FP32 값은 같음)
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
    names = [t for _, t in mins]
    lo = {t: np.inf for t in names}
    hi = {t: -np.inf for t in names}
    ratio = {t: [] for t in names}
    for i in range(len(x)):                                  # 1회차: 텐서별 범위 (사진 64장 전체), 사진별 max/중앙값
        for t, v in zip(names, sess.run(names, {"image": x[i:i + 1]})):
            av = np.abs(v.astype(np.float32))
            lo[t] = min(lo[t], float(v.min()))
            hi[t] = max(hi[t], float(v.max()))
            ratio[t].append(float(av.max() / max(av.mean(), 1e-12)))
    scale = {t: (max(hi[t], 0.0) - min(lo[t], 0.0)) / 255.0 for t in names}   # ORT 비대칭 범위는 0을 포함
    zc = {t: 0 for t in names}
    ez = {t: 0 for t in names}
    nn = {t: 0 for t in names}
    for i in range(len(x)):                                  # 2회차: 64장 범위로 정한 눈금에서 0으로 반올림되는 원소 수 (메모리 절약)
        for t, v in zip(names, sess.run(names, {"image": x[i:i + 1]})):
            av = np.abs(v.astype(np.float32))
            zc[t] += int((av < scale[t] / 2).sum())
            ez[t] += int((av < 1e-6).sum())
            nn[t] += av.size
    rows = [{"node": node, "role": role(node), "n": nn[t], "min": min(lo[t], 0.0), "max": max(hi[t], 0.0), "scale": scale[t],
             "zero_frac": zc[t] / nn[t], "exact_zero_frac": ez[t] / nn[t], "max_over_mean": float(np.mean(ratio[t]))}
            for node, t in mins]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="이름=FP32 onnx 파일 이름($RN50_ONNX 아래)")
    ap.add_argument("--n", type=int, default=32, help="데이터셋당 사진 수")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    x = np.concatenate([np.asarray(np.load(RC.CACHE_DIR / f"{d}_x.npy", mmap_mode="r")[:a.n]) for d in SETS]).astype(np.float32)
    summary = {}
    with open(out / "why_after.jsonl", "a", encoding="utf-8") as f:
        for spec in a.models:
            name, fn = spec.split("=", 1)
            rows = run(RC.onnx_path(fn), x, a.threads)
            for r in rows:
                f.write(json.dumps({"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
                                    "dirty": env["git"]["dirty"], "graph": name, "file": fn, "n_images": len(x), **r}, ensure_ascii=False) + "\n")
            fc2 = [r for r in rows if r["role"].endswith("fc2") or re.search(r"fc2(_\d+)?$", r["role"])]
            summary[name] = {"matmul_inputs": len(rows),
                             "zero_frac_mean_all": round(float(np.mean([r["zero_frac"] for r in rows])), 4),
                             "zero_frac_fc2": [round(r["zero_frac"], 4) for r in fc2],
                             "max_over_mean_fc2": [round(r["max_over_mean"], 1) for r in fc2],
                             "zero_frac_max_all": round(float(max(r["zero_frac"] for r in rows)), 4)}
            print(name, json.dumps(summary[name], ensure_ascii=False), flush=True)
    json.dump(summary, open(out / "why_after_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
