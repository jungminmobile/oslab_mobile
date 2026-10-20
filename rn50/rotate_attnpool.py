"""
rotate_attnpool.py — E11 R4: RN50 AttnPool에 헤드별 하다마드 회전을 오프라인으로 넣는다 (SigLIP2 G3Q 회전의 CNN 이식)
==================================================================================================================
- AttnPool은 32헤드 × 64채널. 같은 헤드 안에서만 섞는 블록 대각 직교 행렬 R = diag(H64/8, ..., H64/8)을 쓴다.
  (헤드를 넘어 섞으면 헤드마다 다른 attention 가중치 때문에 같은 모델이 아니게 된다)
- vc  : v 투영 출력과 c_proj 입력에 R을 넣는다.  Wv' = Wv·R, bv' = bv·R, c_proj Wc' = Wc·R  (Wc는 (1024,2048), transB=1)
         → attention 가중합은 헤드별 선형 결합이라 out' = out·R, c_proj(out') = c_proj(out)
- qkvc: vc + q·k 투영 출력에도 같은 R.  QKᵀ는 헤드별로 (qR)(kR)ᵀ = qkᵀ라 그대로
- stem·layer들은 ReLU(양의 동차성만 있음) 때문에 회전할 수 없다. FP32 출력이 같은지(cos ≥ 0.99999) 확인한다.
사용: python rotate_attnpool.py --mode vc|qkvc   → onnx/rn50_image_fp32_q1rot<mode>.onnx
"""
import argparse
import hashlib

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper as nh

import common as C

SRC = "rn50_image_fp32_q1.onnx"
W = {"q": ("onnx::MatMul_893", "onnx::Add_888"), "k": ("onnx::MatMul_894", "onnx::Add_890"),
     "v": ("onnx::MatMul_895", "onnx::Add_892")}
WC = "attnpool.c_proj.weight"


def hadamard(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def rot(heads=32, d=64):
    return np.kron(np.eye(heads), hadamard(d))          # (2048, 2048) 블록 대각, 직교


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vc", "qkvc"], required=True)
    ap.add_argument("--check-n", type=int, default=64)
    a = ap.parse_args()
    R = rot()
    assert np.allclose(R @ R.T, np.eye(2048), atol=1e-10)
    m = onnx.load(C.onnx_path(SRC).as_posix())
    init = {i.name: i for i in m.graph.initializer}
    def upd(name, f):
        x = nh.to_array(init[name]).astype(np.float64)
        init[name].CopyFrom(nh.from_array(f(x).astype(np.float32), name))
    for p in (("q", "k", "v") if a.mode == "qkvc" else ("v",)):
        upd(W[p][0], lambda w: w @ R)                    # x·W 형태 (2048 입력, 2048 출력)
        upd(W[p][1], lambda b: b @ R)
    upd(WC, lambda w: w @ R)                             # y = out·Wcᵀ → Wc' = Wc·R
    dst = C.onnx_path(SRC).with_name(f"rn50_image_fp32_q1rot{a.mode}.onnx")
    onnx.save(m, dst.as_posix())
    # FP32 동치 확인: pets·caltech dev 앞쪽 사진
    x = np.concatenate([np.load(C.CACHE_DIR / f"{d}_dev_x.npy", mmap_mode="r")[: a.check_n // 2] for d in ("pets", "caltech")])
    x = C.to_input(np.asarray(x)).astype(np.float32)
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    e0 = ort.InferenceSession(C.onnx_path(SRC).as_posix(), so, providers=["CPUExecutionProvider"]).run(None, {"image": x})[0]
    e1 = ort.InferenceSession(dst.as_posix(), so, providers=["CPUExecutionProvider"]).run(None, {"image": x})[0]
    n = lambda e: e / np.linalg.norm(e, axis=1, keepdims=True)
    cos = (n(e0) * n(e1)).sum(1)
    h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
    row = {"mode": a.mode, "file": dst.name, "sha256": h, "n": int(len(x)), "cos_min": float(cos.min()),
           "maxabs": float(np.abs(e0 - e1).max()), "meta": C.run_meta()}
    C.append_jsonl("rotate", row)
    print(row)
    assert cos.min() >= 0.99999, "FP32 동치 실패"


if __name__ == "__main__":
    main()
