"""
equalize.py — RN50 이미지 인코더 FP32 ONNX에 채널 균형 맞추기를 적용해 새 FP32 원본을 만든다 (E5a)
===================================================================================================

[무엇을 하나]
  Conv A → ReLU → (AveragePool) → Conv B 짝에서, A의 출력 채널 i를 1/s_i배, B의 입력 채널 i를 s_i배 한다.
  ReLU·AveragePool은 양수 배율과 순서를 바꿔도 결과가 같아서(ReLU(x/s)=ReLU(x)/s), FP32 출력은 수학적으로 그대로다.
  바뀌는 것은 "채널마다 값의 크기"뿐 → 8비트 자(텐서 하나에 눈금 하나)에 더 고르게 들어간다.

[방법]
  --method cle : DFQ(Nagel 2019) 가중치 범위 균형.  s_i = sqrt(r1_i / r2_i), 3번 반복
                 r1_i = A의 출력 채널 i 가중치 최대|값|, r2_i = B의 입력 채널 i 가중치 최대|값|
  --method sq  : SmoothQuant(Xiao 2023)식 활성값 균형.  s_i = a_i^α / r2_i^(1-α)
                 a_i = 보정 사진(highres 256장, 시드 0 — 양자화와 같은 사진)에서 B 입력 채널 i의 최대|값|
  짝 조건: A 출력을 ReLU 하나만 쓰고, ReLU(·AvgPool) 출력을 B 하나만 쓴다 (블록 안 conv1→conv2, conv2→conv3, stem 2쌍 = 34쌍).
  잔차 덧셈 앞뒤(conv3 출력, 블록 입력)는 지름길과 공유되므로 건드리지 않는다.

[사용]  python equalize.py --method cle            → onnx/rn50_image_fp32_q1cle.onnx
        python equalize.py --method sq --alpha 0.5 → onnx/rn50_image_fp32_q1sq05.onnx
        그다음 quantize_onnx.py --variant _q1cle ... (노드 이름이 같아 --only 규칙을 그대로 쓴다)
"""
import argparse
import time

import numpy as np
import onnx
from onnx import numpy_helper

import common as C


def find_pairs(g):
    cons = {}
    for n in g.node:
        for t in n.input:
            cons.setdefault(t, []).append(n)
    pairs = []
    for a in g.node:
        if a.op_type != "Conv":
            continue
        c = cons.get(a.output[0], [])
        if len(c) != 1 or c[0].op_type != "Relu":
            continue
        t = c[0].output[0]
        c = cons.get(t, [])
        if len(c) == 1 and c[0].op_type == "AveragePool":
            t = c[0].output[0]
            c = cons.get(t, [])
        grp = next((x.i for x in c[0].attribute if x.name == "group"), 1) if len(c) == 1 else 0
        if len(c) == 1 and c[0].op_type == "Conv" and grp == 1:
            pairs.append((a, c[0], t))
    return pairs


def channel_absmax(model_path, tensors, x, bs=16):
    """FP32로 보정 사진을 흘려 각 텐서(NCHW)의 채널별 최대|값|을 구한다."""
    import onnxruntime as ort
    m = onnx.load(model_path)
    have = {o.name for o in m.graph.output}
    for t in tensors:
        if t not in have:
            m.graph.output.append(onnx.helper.make_empty_tensor_value_info(t))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in s.get_outputs()]
    mx = {t: None for t in tensors}
    for i in range(0, len(x), bs):
        outs = dict(zip(names, s.run(None, {"image": x[i:i + bs]})))
        for t in tensors:
            v = np.abs(outs[t]).max(axis=(0, 2, 3))
            mx[t] = v if mx[t] is None else np.maximum(mx[t], v)
    return mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="_q1")
    ap.add_argument("--method", choices=["cle", "sq"], required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--smin", type=float, default=1e-2, help="배율 하한·상한(1/smin) — 0에 가까운 채널 폭주 방지")
    a = ap.parse_args()
    t0 = time.time()
    src = C.ONNX_DIR / f"{C.PREFIX}_image_fp32{a.src}.onnx"
    tag = a.src + ("cle" if a.method == "cle" else f"sq{a.alpha:g}".replace(".", ""))
    dst = C.ONNX_DIR / f"{C.PREFIX}_image_fp32{tag}.onnx"
    m = onnx.load(str(src))
    g = m.graph
    init = {i.name: i for i in g.initializer}
    W = {k: numpy_helper.to_array(v).astype(np.float64) for k, v in init.items()}
    pairs = find_pairs(g)
    print(f"pairs {len(pairs)}")
    act = None
    if a.method == "sq":
        from quantize_onnx import calib_images
        act = channel_absmax(str(src), [p[2] for p in pairs], calib_images("highres", 256, 0))
    log = []
    for it in range(a.iters if a.method == "cle" else 1):
        for A, B, t in pairs:
            wa, ba, wb = A.input[1], (A.input[2] if len(A.input) > 2 else None), B.input[1]
            r1 = np.abs(W[wa]).reshape(W[wa].shape[0], -1).max(1)
            r2 = np.abs(W[wb]).transpose(1, 0, 2, 3).reshape(W[wb].shape[1], -1).max(1)
            if a.method == "cle":
                s = np.sqrt(r1 / np.maximum(r2, 1e-12))
            else:
                s = act[t] ** a.alpha / np.maximum(r2, 1e-12) ** (1 - a.alpha)
            s = np.where((r1 > 0) & (r2 > 0), s, 1.0)
            s = np.clip(s, a.smin, 1 / a.smin)
            W[wa] /= s[:, None, None, None]
            if ba:
                W[ba] /= s
            W[wb] *= s[None, :, None, None]
            if it == 0:
                log.append({"A": A.name, "B": B.name, "s_min": float(s.min()), "s_med": float(np.median(s)), "s_max": float(s.max())})
            if act is not None:
                act[t] = act[t] / s
    for k in W:
        init[k].CopyFrom(numpy_helper.from_array(W[k].astype(np.float32), k))
    onnx.save(m, str(dst))

    # FP32 동등성 확인: 원본과 출력이 같아야 한다 (cos ≈ 1)
    import onnxruntime as ort
    xin = C.to_input(np.load(C.CACHE_DIR / "pets_dev_x.npy", mmap_mode="r")[:32])
    o1 = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).run(None, {"image": xin})[0]
    o2 = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"]).run(None, {"image": xin})[0]
    cos = float((C.l2n(o1) * C.l2n(o2)).sum(1).min())
    rel = float(np.abs(o1 - o2).max() / np.abs(o1).max())
    print(f"saved {dst.name}  fp32 equivalence: min cos {cos:.7f}, max rel diff {rel:.2e}  ({time.time() - t0:.0f}s)")
    C.append_jsonl("equalize", {"dst": dst.name, "method": a.method, "alpha": a.alpha, "iters": a.iters,
                                "pairs": len(pairs), "fp32_min_cos": cos, "fp32_max_rel": rel, "scales": log,
                                "sec": round(time.time() - t0)})
    assert cos > 0.99999, "FP32 출력이 바뀌었다 — 변환 오류"


if __name__ == "__main__":
    main()
