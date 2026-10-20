"""
make_xplat_bundle.py — 다른 기기(ARM Orin, 다른 x86) 측정용 묶음 만들기 (10/15 준용 요청, GitHub pre-release siglip2-xplat-20261015)
=========================================================================================================
만드는 것: $SIGLIP2_ARTIFACTS/share/<이름>/
  models/   후보 ONNX 복사 (FP32, fc2L2 S8, fc2L2 U8 + reduce_range)
  dev200/   dev 6종 각 앞 K장: <set>.npz (x 전처리 입력 float32, y 정답, clf FP32 분류기, ref FP32 이미지 임베딩)
  portable_eval.py (scripts/portable_eval.py 복사), SHA256SUMS.txt
README.md는 손으로 썼다 (같은 폴더). 10/15 묶음은 이 스크립트와 같은 내용으로 만들었다 (독립 검토 지적: 만든 코드를 저장소에 남김).
사용: "$SIGLIP2_PYTHON" scripts/make_xplat_bundle.py --name 20261015_xplat --k 200
"""
import argparse
import hashlib
import os
import shutil
from pathlib import Path

import numpy as np

SETS = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
MODELS = ["siglip2_image_fp32.onnx",
          "siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16_fc2L2.onnx",
          "siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--models", nargs="+", default=MODELS, help="(G8) 넣을 ONNX 이름 ($SIGLIP2_ARTIFACTS/onnx 아래). 기본은 10/15 묶음과 같음")
    a = ap.parse_args()
    A = Path(os.environ["SIGLIP2_ARTIFACTS"])
    C, O = A / "cache", A / "onnx"
    B = A / "share" / a.name
    (B / "models").mkdir(parents=True, exist_ok=True)
    (B / "dev200").mkdir(exist_ok=True)
    for f in a.models:
        if not (B / "models" / f).exists():
            shutil.copy2(O / f, B / "models" / f)
    for d in SETS:
        x = np.asarray(np.load(C / f"{d}_x.npy", mmap_mode="r")[:a.k])
        np.savez(B / "dev200" / f"{d}.npz", x=x, y=np.load(C / f"{d}_y.npy")[:a.k],
                 clf=np.load(C / f"{d}_clf.npy"), ref=np.load(C / f"{d}_ref_fp32.npy")[:a.k])
        print(d, x.dtype, x.shape)
    shutil.copy2(Path(__file__).with_name("portable_eval.py"), B / "portable_eval.py")
    lines = [f"{sha(p)}  {p.relative_to(B).as_posix()}" for p in sorted(B.rglob("*")) if p.is_file() and p.name not in ("SHA256SUMS.txt", "README.md")]
    (B / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
