"""
s2_merge_dfc_windows.py — DFC 스크리닝을 구간(--start)·데이터셋 묶음(--sets)으로 나눠 돌린 사진별 예측을 dev 전체 하나로 합친다 (G4 dev 전체 평가)
================================================================================================================================
입력: --parts 폴더:tag 여러 개. 각 tag의 npz(<폴더>/pred_<tag>.npz)와 screen.jsonl 행(start, k, har_sha256).
      한 npz에 일부 데이터셋만 있어도 된다 (--sets로 나눠 돌린 구간, 10/15 start=800 OOM 재실행).
확인: 모든 part의 har_sha256이 같음, 데이터셋마다 구간이 [0, n)을 빈틈·겹침 없이 덮음,
      합친 길이 = dev 캐시 길이, 정답 순서 = 캐시 정답
출력: <out>/pred_<name>_full.npz → s2_compare_dfc_ort.py --k 0 으로 ORT dev 전체와 짝 비교

사용:
  "$SIGLIP2_PYTHON" s2_merge_dfc_windows.py --name D256L1_official_b9fc2 --out ../results/siglip2/G4_dfc_full \
      --parts ../results/siglip2/G4_dfc_screen:D256L1_official_b9fc2 ../results/siglip2/G4_dfc_full:D256L1_official_b9fc2_s200 ...
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


def row_of(folder, tag):
    rows = [json.loads(l) for l in open(Path(folder) / "screen.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["tag"] == tag]
    assert len({r["har_sha256"] for r in rows}) == 1, (folder, tag)
    return rows[-1]                                    # 같은 tag를 다시 돌렸으면 마지막 행 (npz도 마지막 실행이 덮어씀)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", nargs="+", required=True, help="폴더:tag")
    a = ap.parse_args()
    pieces = {d: [] for d in DATA}                     # 데이터셋 → [(start, pred, cos, y, tag)]
    hars, info = set(), []
    for spec in a.parts:
        folder, tag = spec.rsplit(":", 1)
        r = row_of(folder, tag)
        hars.add(r["har_sha256"])
        st = r.get("start", 0)                         # start 필드가 생기기 전 행은 0부터
        z = np.load(Path(folder) / f"pred_{tag}.npz")
        for d in DATA:
            if f"{d}_pred" in z.files and len(z[f"{d}_pred"]):
                pieces[d].append((st, z[f"{d}_pred"], z[f"{d}_cos"], z[f"{d}_y"], tag))
        info.append({"tag": tag, "start": st, "commit": r["commit"], "dirty": r["dirty"]})
    assert len(hars) == 1, f"HAR이 서로 다름: {hars}"
    out = {}
    for d in DATA:
        y = np.load(RC.CACHE_DIR / f"{d}_y.npy")
        ps = sorted(pieces[d], key=lambda t: t[0])
        pos = 0
        for st, pred, _, _, tag in ps:
            assert st == pos, f"{d}: 구간 빈틈·겹침 ({tag} start={st}, 기대 {pos})"
            pos += len(pred)
        assert pos == len(y), (d, pos, len(y))
        for i, k in ((1, "pred"), (2, "cos"), (3, "y")):
            out[f"{d}_{k}"] = np.concatenate([p[i] for p in ps])
        assert np.array_equal(out[f"{d}_y"], y), d
        print(d, len(y), "parts", len(ps), "top1", round(100 * float((out[f"{d}_pred"] == y).mean()), 2),
              "cos", round(float(out[f"{d}_cos"].mean()), 4))
    dst = Path(a.out) / f"pred_{a.name}_full.npz"
    np.savez_compressed(dst, **out)
    with open(Path(a.out) / "merge.json", "w", encoding="utf-8") as f:
        json.dump({"name": a.name, "har_sha256": hars.pop(), "parts": info, "out": dst.name}, f, ensure_ascii=False, indent=1)
    print("saved", dst)


if __name__ == "__main__":
    main()
