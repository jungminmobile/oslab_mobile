"""
s2_dfc_calib.py — Hailo DFC optimize용 보정 사진 (NHWC float32) 만들기 (ROADMAP G1)
================================================================================

[목적]  DFC는 float ONNX에서 직접 양자화한다(PRD 4-2). 보정 사진은 ORT 보정과 **같은 후보 풀·같은 뽑기 함수**
  (s2_quantize.calib_images: highres, 시드 0)에서 N장(기본 64, ROADMAP G1 "보정 64장")을 뽑는다.
  ONNX 입력은 NCHW·정규화(평균·표준편차 0.5) 후 값이다. DFC 파서는 입력을 NHWC로 바꾸므로 NHWC로 저장한다.
  정규화를 모델 안(.alls normalization)으로 옮기지 않는 기본 구성(S-C)이라 정규화된 값을 그대로 넣는다.

[사용]
  "$SIGLIP2_PYTHON" hailo/s2_dfc_calib.py --n 64   → $SIGLIP2_ARTIFACTS/dfc/calib_highres64_s0_nhwc.npy
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import s2common as S        # noqa: E402
import s2_quantize as SQ    # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    x = SQ.calib_images("highres", a.n, a.seed)                       # (n,3,224,224) float32, ORT 보정과 같은 함수
    assert x.shape == (a.n, 3, 224, 224) and x.dtype == np.float32 and -1.0001 <= x.min() and x.max() <= 1.0001
    nhwc = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
    out = Path(S.os.environ["SIGLIP2_ARTIFACTS"]) / "dfc" / f"calib_highres{a.n}_s{a.seed}_nhwc.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, nhwc)
    print(out, nhwc.shape, hashlib.sha256(out.read_bytes()).hexdigest())
