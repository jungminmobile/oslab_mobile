"""
clip_weights.py — 4비트용 가중치 잘라내기(clipping) 최적화 (R6b 후속 실험)
=========================================================================

[왜 필요한가]
  R6에서 단순 MinMax 4비트는 모델을 망가뜨렸다. 채널마다 가장 큰 가중치가 표준편차의 4~8배라,
  그 값에 맞춰 눈금 16칸을 정하면 대부분의 가중치가 0 근처 몇 칸에 몰려 해상도가 사라진다.
  → 아주 큰 몇 개는 잘라내고(clip) 나머지를 촘촘하게 표현하는 편이 전체 오차가 작다.

[하는 일]
  지정한 노드(Conv/MatMul/Gemm)의 가중치마다, 채널별로 잘라낼 비율 α (0.30~1.00, 0.02 간격)를 바꿔 가며
  "4비트 대칭 양자화 후 평균제곱오차(MSE)"가 가장 작은 α를 찾고, 가중치를 [-α·max, α·max]로 잘라 둔다.
  잘라 둔 FP32 모델을 만들면 quantize_onnx.py의 MinMax가 그 범위로 눈금을 정하므로 결과적으로 MSE 최적 눈금이 된다.
  (가중치 값만 바꾸고 활성값·다른 층은 그대로)

[사용]
  python clip_weights.py --src rn50_image_fp32_q1.onnx --out-variant _q1clip4 --prefix /layer3,/layer4 --exact <AP 노드들>
  python quantize_onnx.py --enc image --variant _q1clip4 ... --w4 /layer3,/layer4 --w4-exact <AP 노드들>
"""
import argparse

import numpy as np
import onnx
from onnx import numpy_helper

import common as C


def best_clip(wm, bits=4, grid=np.arange(0.30, 1.001, 0.02)):
    """wm: (채널, 나머지) → 채널별 MSE 최소 α와 그때의 SQNR(dB) 개선."""
    q = 2 ** (bits - 1) - 1
    amax = np.abs(wm).max(1, keepdims=True) + 1e-12
    best_err = np.full((wm.shape[0], 1), np.inf)
    best_a = np.ones((wm.shape[0], 1))
    for a in grid:
        s = a * amax / q
        e = ((wm - np.clip(np.round(wm / s), -q, q) * s) ** 2).sum(1, keepdims=True)
        better = e < best_err
        best_err[better], best_a[better] = e[better], a
    return best_a, best_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="기본: <prefix>_image_fp32_q1.onnx")
    ap.add_argument("--out-variant", default="_q1clip4")
    ap.add_argument("--prefix", default="", help="대상 노드 접두사(쉼표)")
    ap.add_argument("--exact", default="", help="대상 노드 정확한 이름(쉼표)")
    a = ap.parse_args()
    src = C.onnx_path(a.src or f"{C.PREFIX}_image_fp32_q1.onnx")
    m = onnx.load(str(src))
    inits = {i.name: i for i in m.graph.initializer}
    pre = tuple(p for p in a.prefix.split(",") if p)
    ex = {p for p in a.exact.split(",") if p}
    n_done = 0
    for n in m.graph.node:
        if n.op_type not in ("Conv", "MatMul", "Gemm") or not ((pre and n.name.startswith(pre)) or n.name in ex):
            continue
        if n.input[1] not in inits:
            continue
        init = inits[n.input[1]]
        w = numpy_helper.to_array(init).astype(np.float32)
        trans_b = any(at.name == "transB" and at.i == 1 for at in n.attribute)
        ax = 1 if n.op_type == "MatMul" or (n.op_type == "Gemm" and not trans_b) else 0   # quantize_onnx.py와 같은 채널 축
        wm = np.moveaxis(w, ax, 0).reshape(w.shape[ax], -1)
        alpha, err = best_clip(wm)
        q = 7
        e_full = ((wm - np.round(wm / (np.abs(wm).max(1, keepdims=True) / q)) * (np.abs(wm).max(1, keepdims=True) / q)) ** 2).sum()
        lim = alpha * np.abs(wm).max(1, keepdims=True)
        wm_c = np.clip(wm, -lim, lim)
        w_c = np.moveaxis(wm_c.reshape(np.moveaxis(w, ax, 0).shape), 0, ax)
        init.CopyFrom(numpy_helper.from_array(w_c.astype(np.float32), init.name))
        sig = (wm ** 2).sum()
        print(f"{n.name:45s} α 평균 {alpha.mean():.2f}  4비트 SQNR {10*np.log10(sig/e_full):5.1f} → {10*np.log10(sig/err.sum()):5.1f} dB")
        n_done += 1
    assert n_done, "대상 노드 없음"
    out = C.ONNX_DIR / f"{C.PREFIX}_image_fp32{a.out_variant}.onnx"
    onnx.save(m, str(out))
    print("saved", out.name, "nodes", n_done)


if __name__ == "__main__":
    main()
