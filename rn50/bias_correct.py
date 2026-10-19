"""
bias_correct.py — E5b: 경험적·순차적 Bias Correction (Nagel 2019, Hailo IBC 계열)을 ORT QDQ 모델에 적용
=======================================================================================================
계획서: 프로젝트 문서 18, 사전 등록 results/E5b_bias_corr/PREREG.md

[하는 일]  QDQ 모델의 Conv·MatMul·Gemm 출력 채널별 평균 이동 Δ_c = mean(y_q) − mean(y_fp)를 앞 층부터 차례로 재서
           bias에서 뺀다. 바뀌는 것은 bias 값뿐(그래프·가중치·scale·zero point 그대로).
  - Conv: bias가 int32 DQ(scale = s_x·s_w,c) → int32 값에서 round(Δ_c / scale_c)를 뺀다
  - AttnPool q·k·v MatMul: 뒤따르는 Add의 float 상수에서 Δ를 뺀다 / c_proj Gemm: Conv와 같이 int32 DQ bias에서 뺀다
  - y_q는 fake-quant 실행(그래프 최적화 끔 → QDQ가 float로 계산)의 층 출력(ReLU 전), y_fp는 FP32 q1 모델의 같은 텐서
  - 평균은 배치별 합으로 누적(메모리 절약). 앞 층을 고친 모델로 다음 층을 잰다(순차).
[보정 사진] --calib-seed 0: 양자화 보정과 같은 256장(quantize_onnx.calib_images와 같은 선택)
            --calib-seed 1: 같은 후보 풀에서 시드 0 선택을 뺀 나머지 중 256장(겹침 없음)
[확인 모드] --zero-delta: Δ를 0으로 강제 → 출력 파일이 입력과 sha256까지 같아야 함
사용: python bias_correct.py --src <QDQ.onnx> --tag BC [--calib-seed 0] [--zero-delta]
"""
import argparse
import hashlib
import json
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper as nh

import common as C

FP32 = "rn50_image_fp32_q1.onnx"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def bc_images(seed, n=256):
    files = sorted(C.CACHE_DIR.glob("calib_*_u8.npy"))
    pool = np.concatenate([np.load(f) for f in files])
    idx0 = np.random.default_rng(0).choice(len(pool), n, replace=False)   # quantize_onnx.calib_images(seed 0)와 같은 선택
    if seed == 0:
        idx = idx0
    else:
        rest = np.setdiff1d(np.arange(len(pool)), idx0)
        idx = rest[np.random.default_rng(seed).choice(len(rest), n, replace=False)]
    return C.to_input(pool[idx]).astype(np.float32), idx


def targets(m):
    """양자화된 Conv/MatMul/Gemm과 그 bias 위치를 위상 순서로."""
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    init = {i.name for i in g.initializer}
    T = []
    for n in g.node:
        if n.op_type not in ("Conv", "MatMul", "Gemm") or len(n.input) < 2:
            continue
        w = prod.get(n.input[1])
        if w is None or w.op_type != "DequantizeLinear":
            continue                                       # FP32로 남은 층·attention 계산은 건너뜀
        t = {"node": n.name, "op": n.op_type, "out": n.output[0]}
        if n.op_type in ("Conv", "Gemm") and len(n.input) > 2:
            b = prod.get(n.input[2])
            if b is not None and b.op_type == "DequantizeLinear":
                t.update(kind="int32", q=b.input[0], scale=b.input[1])
            elif n.input[2] in init:
                t.update(kind="float", f=n.input[2])
        elif n.op_type == "MatMul":
            q = [c for c in cons.get(n.output[0], []) if c.op_type == "QuantizeLinear"]
            dq = [c for c in cons.get(q[0].output[0], []) if c.op_type == "DequantizeLinear"] if q else []
            add = [c for c in cons.get(dq[0].output[0], []) if c.op_type == "Add"] if dq else []
            if add:
                k = [i for i in add[0].input if i in init]
                if k:
                    t.update(kind="float", f=k[0])
        if "kind" not in t:
            raise SystemExit(f"bias를 찾지 못함: {n.name}")
        T.append(t)
    return T


def chan_axis_mean(y):
    """채널(마지막 축 또는 NCHW의 1축)별 합과 개수."""
    if y.ndim == 4:
        return y.sum(axis=(0, 2, 3)), y.shape[0] * y.shape[2] * y.shape[3]
    return y.reshape(-1, y.shape[-1]).sum(0), int(np.prod(y.shape[:-1]))


def run_means(model, outs, x, bs=16, fake=True, threads=4):
    """model(ModelProto 또는 경로)에서 outs 텐서들의 채널별 평균."""
    m = onnx.load(model) if isinstance(model, str) else onnx.ModelProto.FromString(model.SerializeToString())
    have = {o.name for o in m.graph.output}
    for t in outs:
        if t not in have:
            m.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL if fake else ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    sums, cnt = {t: 0.0 for t in outs}, {t: 0 for t in outs}
    for i in range(0, len(x), bs):
        r = s.run(outs, {"image": x[i:i + bs]})
        for t, y in zip(outs, r):
            if y.dtype != np.float32:
                y = y.astype(np.float32)
            a, c = chan_axis_mean(y)
            sums[t] = sums[t] + a
            cnt[t] += c
    return {t: sums[t] / cnt[t] for t in outs}


def apply(m, t, delta):
    init = {i.name: i for i in m.graph.initializer}
    if t["kind"] == "int32":
        q = nh.to_array(init[t["q"]]).astype(np.int64)
        sc = nh.to_array(init[t["scale"]]).astype(np.float64)
        step = np.round(delta / sc).astype(np.int64)
        new = np.clip(q - step, -2**31, 2**31 - 1).astype(np.int32)
        init[t["q"]].CopyFrom(nh.from_array(new, t["q"]))
        return float(np.abs(step * sc).mean())
    f = nh.to_array(init[t["f"]])
    new = (f.astype(np.float64) - delta.reshape(f.shape) if f.size == delta.size else f - delta).astype(f.dtype)
    init[t["f"]].CopyFrom(nh.from_array(new, t["f"]))
    return float(np.abs(delta).mean())


def int_points(m, T):
    """정수 커널 확인용: 각 Conv 뒤 첫 QuantizeLinear 출력(uint8)과 그 FP32 대응 텐서(Q의 입력)."""
    g = m.graph
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    init = {i.name: i for i in g.initializer}
    P = []
    for t in T:
        if t["op"] != "Conv":
            continue
        cur, q = t["out"], None
        for _ in range(3):
            nxt = cons.get(cur, [])
            q = next((c for c in nxt if c.op_type == "QuantizeLinear"), None)
            if q or not nxt or nxt[0].op_type not in ("Relu",):
                break
            cur = nxt[0].output[0]
        if q is not None:
            P.append({"node": t["node"], "q_out": q.output[0], "fp_in": q.input[0],
                      "scale": float(nh.to_array(init[q.input[1]])), "zp": float(nh.to_array(init[q.input[2]]))})
    return P


def int_shift(model_path, P, x):
    """정수 커널(ORT_ENABLE_ALL)에서 Q 출력을 역양자화한 채널 평균 vs FP32 같은 위치 평균 → 층별 평균 |Δ|."""
    mq = run_means(model_path, [p["q_out"] for p in P], x, fake=False)
    mf = run_means(C.onnx_path(FP32).as_posix(), [p["fp_in"] for p in P], x, fake=True)
    return {p["node"]: float(np.abs((mq[p["q_out"]] - p["zp"]) * p["scale"] - mf[p["fp_in"]]).mean()) for p in P}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--zero-delta", action="store_true")
    ap.add_argument("--check-int", action="store_true", help="0단계 확인 ③: 정수 커널 기준 감소율도 잰다")
    a = ap.parse_args()
    t0 = time.time()
    src = C.onnx_path(a.src)
    dst = src.with_name(src.stem + f"_{a.tag}.onnx")
    m = onnx.load(src.as_posix())
    T = targets(m)
    x, idx = bc_images(a.calib_seed)
    fp = run_means(C.onnx_path(FP32).as_posix(), [t["out"] for t in T], x, fake=True)
    log = []
    for i, t in enumerate(T):
        mq = run_means(m, [t["out"]], x, fake=True)[t["out"]]
        delta = mq - fp[t["out"]]
        before = float(np.abs(delta).mean())
        if a.zero_delta:
            delta = np.zeros_like(delta)
        applied = apply(m, t, delta) if not a.zero_delta else 0.0
        rel = before / (float(np.abs(fp[t["out"]]).mean()) + 1e-12)
        log.append({"i": i, "node": t["node"], "op": t["op"], "kind": t["kind"], "ch": int(delta.size),
                    "absdelta_mean": before, "absdelta_max": float(np.abs(mq - fp[t["out"]]).max()),
                    "rel_to_mean_abs_act": rel, "applied_mean": applied})
        print(f"[{i + 1}/{len(T)}] {t['node']:45s} |Δ| {before:.4f} rel {rel:.3f}", flush=True)
    onnx.save(m, dst.as_posix())
    # 확인 ③(fake): 고친 모델로 모든 층 |Δ|를 다시 잼
    after = run_means(m, [t["out"] for t in T], x, fake=True)
    for r, t in zip(log, T):
        r["absdelta_after_fake"] = float(np.abs(after[t["out"]] - fp[t["out"]]).mean())
    red_fake = 1 - sum(r["absdelta_after_fake"] for r in log) / sum(r["absdelta_mean"] for r in log)
    row = {"src": src.name, "src_sha": sha(src), "dst": dst.name, "dst_sha": sha(dst), "tag": a.tag, "calib_seed": a.calib_seed,
           "calib_idx_head": idx[:8].tolist(), "zero_delta": a.zero_delta, "n_layers": len(T),
           "reduction_fake": red_fake, "sec": round(time.time() - t0), "meta": C.run_meta()}
    if a.check_int:
        P = int_points(m, T)
        b0, bc = int_shift(src.as_posix(), P, x), int_shift(dst.as_posix(), P, x)
        row["int_points"] = len(P)
        row["reduction_int"] = 1 - sum(bc.values()) / sum(b0.values())
        row["int_shift_before"], row["int_shift_after"] = b0, bc
    with open(C.RESULTS_DIR / "bc_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({**row, "layers": log}, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k not in ("int_shift_before", "int_shift_after", "meta")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
