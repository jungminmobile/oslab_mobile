"""
s2_merge_dfc_windows.py — DFC 스크리닝을 구간(--start)으로 나눠 돌린 사진별 예측을 dev 전체 하나로 합친다 (G4 dev 전체 평가)
================================================================================================================
입력: 첫 구간 [0, k) npz(보통 G4_dfc_screen의 pred_<tag>.npz) + 나머지 구간 npz들 (pred_<tag>_s<start>.npz)
확인: 각 데이터셋의 합친 길이 = dev 캐시 길이, 정답 순서 = 캐시 정답, 모든 구간의 har_sha256이 같음 (screen.jsonl)
출력: <out>/pred_<tag>_full.npz → s2_compare_dfc_ort.py --k 0 으로 ORT dev 전체와 짝 비교

사용:
  "$SIGLIP2_PYTHON" s2_merge_dfc_windows.py --tag D256L1_official_b9fc2 \
      --first ../results/siglip2/G4_dfc_screen --windows ../results/siglip2/G4_dfc_full --starts 200 400 600 800 1000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S  # noqa: E402

RC = S.RC
DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]


def har_of(folder, tag):
    rows = [json.loads(l) for l in open(Path(folder) / "screen.jsonl", encoding="utf-8")]
    hs = {r["har_sha256"] for r in rows if r["tag"] == tag}
    assert len(hs) == 1, (folder, tag, hs)
    return hs.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--first", required=True, help="첫 구간 [0,k) npz가 있는 폴더 (screen.jsonl 포함)")
    ap.add_argument("--windows", required=True, help="나머지 구간 npz 폴더 (screen.jsonl 포함), 출력도 여기에")
    ap.add_argument("--starts", nargs="+", type=int, required=True)
    a = ap.parse_args()
    h0 = har_of(a.first, a.tag)
    for st in a.starts:
        assert har_of(a.windows, f"{a.tag}_s{st}") == h0, f"구간 {st}의 HAR이 다름"
    parts = [np.load(Path(a.first) / f"pred_{a.tag}.npz")] + [np.load(Path(a.windows) / f"pred_{a.tag}_s{st}.npz") for st in a.starts]
    out = {}
    for d in DATA:
        y = np.load(RC.CACHE_DIR / f"{d}_y.npy")
        for k in ("pred", "cos", "y"):
            out[f"{d}_{k}"] = np.concatenate([p[f"{d}_{k}"] for p in parts])
        assert len(out[f"{d}_pred"]) == len(y), (d, len(out[f"{d}_pred"]), len(y))
        assert np.array_equal(out[f"{d}_y"], y), d
        print(d, len(y), "top1", round(100 * float((out[f"{d}_pred"] == y).mean()), 2), "cos", round(float(out[f"{d}_cos"].mean()), 4))
    dst = Path(a.windows) / f"pred_{a.tag}_full.npz"
    np.savez_compressed(dst, **out)
    print("saved", dst, "har_sha256", h0[:16])


if __name__ == "__main__":
    main()
