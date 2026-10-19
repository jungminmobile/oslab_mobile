"""g5b_calib.py — 벤더 레시피용 보정 사진: 우리 보정 풀(train, highres, 시드 0, 256장 NHWC 정규화 값)의 앞 N장을 원본 픽셀(0~255)로 되돌린다.
벤더 .alls 첫 줄 normalization([127.5]*3, [127.5]*3)이 정규화를 HEF 안에 넣으므로, 보정 입력은 정규화 전 값이어야 한다.
x_pix = round(x * 127.5 + 127.5), 0~255로 자름, float32로 저장 (칩에 넣는 uint8 값과 같은 격자).
사용: python g5b_calib.py --src calib_highres256_s0_nhwc.npy --n 64 --out calib_pix64_s0_nhwc.npy
"""
import argparse, hashlib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--n", type=int, default=64); ap.add_argument("--out", required=True)
a = ap.parse_args()
x = np.load(a.src, mmap_mode="r")
assert x.ndim == 4 and x.shape[1:] == (224, 224, 3) and x.shape[0] >= a.n, x.shape
x = np.asarray(x[: a.n], dtype=np.float32)
assert -1.0001 <= x.min() and x.max() <= 1.0001, (x.min(), x.max())
p = np.clip(np.round(x * 127.5 + 127.5), 0, 255).astype(np.float32)
np.save(a.out, p)
print(f"calib n={len(p)} shape={p.shape} min={p.min()} max={p.max()} mean={p.mean():.2f} "
      f"roundtrip_maxabs={np.abs((p - 127.5) / 127.5 - x).max():.5f} sha256={hashlib.sha256(open(a.out, 'rb').read()).hexdigest()}")
