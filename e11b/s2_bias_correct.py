"""
s2_bias_correct.py — E11b S2: SigLIP2 R0(QDQ, MatMul만 INT8)에 경험적·순차적 Bias Correction
==========================================================================================
- 대상: 가중치가 DequantizeLinear로 들어오는 MatMul(= 양자화된 Linear) 중 뒤에 bias Add(상수)가 있는 것.
  (MatMul → Add 또는 MatMul → Q → DQ → Add). attention 점수·가중합(활성값×활성값)과 FP32로 남은 블록 5·9 fc2는 제외.
- Δ_c = mean(fake-quant 출력) − mean(FP32 출력), 채널 = 마지막 축. 앞 층부터 차례로 bias 상수에서 뺀다.
- FP32 기준은 양자화기 입력과 같은 그래프 `siglip2_image_fp32_pre.onnx`(텐서 이름이 같다).
- RN50 E5b 교훈 반영: ① FP32 쪽 텐서 이름이 같은 뜻인지 확인(없으면 멈춤) ② 확인 ③은 보정이 들어가는 Add 출력에서 잰다.
사용: s2_venv python s2_bias_correct.py --src <R0.onnx> --tag BC [--calib-seed 0|1] [--zero-delta]
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper as nh

S2 = Path(__file__).resolve().parents[2] / "oslab_s2" / "siglip2"
sys.path.insert(0, str(S2))
import s2common as S                      # noqa: E402
from s2_quantize import calib_images      # noqa: E402

RC = S.RC
FP32_PRE = "siglip2_image_fp32_pre.onnx"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def bc_images(seed, n=256):
    """seed 0: 양자화 보정과 같은 256장. seed 1: 후보 풀(311장)에서 seed 0 선택을 뺀 나머지 55장 + seed 0 중 일부가 아니라
    겹침을 피하려 '나머지 전부(55장)'만 쓴다."""
    if seed == 0:
        return calib_images("highres", n, 0)
    files = sorted(RC.CACHE_DIR.glob("calib_*_f32.npy"))
    pool = np.concatenate([np.load(f) for f in files])
    rng = np.random.default_rng(0)
    idx0 = rng.choice(len(pool), n, replace=False)        # calib_images(highres, 256, 0)와 같은 뽑기 순서
    rest = np.setdiff1d(np.arange(len(pool)), idx0)
    return pool[rest].astype(np.float32)


def targets(m):
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    init = {i.name for i in g.initializer}
    T = []
    for n in g.node:
        if n.op_type != "MatMul" or len(n.input) < 2:
            continue
        w = prod.get(n.input[1])
        if w is None or w.op_type != "DequantizeLinear" or w.input[0] not in init:
            continue
        cur = n.output[0]
        add = None
        for _ in range(3):
            nx = cons.get(cur, [])
            a = [c for c in nx if c.op_type == "Add" and any(i in init for i in c.input)]
            if a:
                add = a[0]
                break
            q = [c for c in nx if c.op_type in ("QuantizeLinear", "DequantizeLinear")]
            if not q:
                break
            cur = q[0].output[0]
        if add is None:
            continue                                      # bias 없는 MatMul (있으면 기록)
        k = [i for i in add.input if i in init][0]
        T.append({"node": n.name, "out": n.output[0], "f": k, "chk": add.output[0]})
    return T


def run_means(model, outs, x, bs=16, fake=True, threads=6):
    m = onnx.load(model) if isinstance(model, str) else onnx.ModelProto.FromString(model.SerializeToString())
    have = {o.name for o in m.graph.output}
    for t in outs:
        if t not in have:
            m.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL if fake else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    iname = s.get_inputs()[0].name
    sums, cnt = {t: 0.0 for t in outs}, {t: 0 for t in outs}
    for i in range(0, len(x), bs):
        for t, y in zip(outs, s.run(outs, {iname: x[i:i + bs]})):
            y = y.astype(np.float64)
            sums[t] = sums[t] + y.reshape(-1, y.shape[-1]).sum(0)
            cnt[t] += int(np.prod(y.shape[:-1]))
    return {t: sums[t] / cnt[t] for t in outs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--zero-delta", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    src = RC.onnx_path(a.src)
    dst = src.with_name(src.stem + f"_{a.tag}.onnx")
    m = onnx.load(src.as_posix())
    T = targets(m)
    fp_path = RC.onnx_path(FP32_PRE).as_posix()
    fp_names = {o for n in onnx.load(fp_path, load_external_data=False).graph.node for o in n.output}
    miss = [t["out"] for t in T if t["out"] not in fp_names] + [t["chk"] for t in T if t["chk"] not in fp_names]
    assert not miss, f"FP32 그래프에 없는 텐서: {miss[:5]}"
    x = bc_images(a.calib_seed)
    names = list(dict.fromkeys([t["out"] for t in T] + [t["chk"] for t in T]))
    fp = run_means(fp_path, names, x, fake=True)
    log = []
    for i, t in enumerate(T):
        r0 = run_means(m, [t["out"], t["chk"]], x, fake=True)
        delta = r0[t["out"]] - fp[t["out"]]
        before = float(np.abs(delta).mean())
        before_chk = float(np.abs(r0[t["chk"]] - fp[t["chk"]]).mean())
        if not a.zero_delta:
            init = {z.name: z for z in m.graph.initializer}
            f = nh.to_array(init[t["f"]])
            assert f.size == delta.size, (t["node"], f.shape, delta.shape)
            init[t["f"]].CopyFrom(nh.from_array((f.astype(np.float64) - delta.reshape(f.shape)).astype(f.dtype), t["f"]))
        log.append({"i": i, "node": t["node"], "ch": int(delta.size), "absdelta_mean": before,
                    "absdelta_chk_before": before_chk, "rel": before / (float(np.abs(fp[t["out"]]).mean()) + 1e-12)})
        print(f"[{i + 1}/{len(T)}] {t['node']:50s} |Δ| {before:.4f}", flush=True)
    onnx.save(m, dst.as_posix())
    after = run_means(m, [t["chk"] for t in T], x, fake=True)
    for r, t in zip(log, T):
        r["absdelta_after_fake"] = float(np.abs(after[t["chk"]] - fp[t["chk"]]).mean())
    red = 1 - sum(r["absdelta_after_fake"] for r in log) / sum(r["absdelta_chk_before"] for r in log)
    row = {"src": src.name, "src_sha": sha(src), "dst": dst.name, "dst_sha": sha(dst), "tag": a.tag,
           "calib_seed": a.calib_seed, "n_img": int(len(x)), "zero_delta": a.zero_delta, "n_layers": len(T),
           "reduction_fake": red, "sec": round(time.time() - t0)}
    out = S.require_results_dir()
    with open(out / "bc_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({**row, "layers": log}, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
