"""
s2_compare_dfc_ort.py — G4 DFC 에뮬레이션 구성과 G3 ORT 정수 구성을 같은 사진으로 짝지어 비교 (ROADMAP G4)
=================================================================================================

[목적]
  DFC 스크리닝(hailo/s2_dfc_screen.py)은 dev 6종 각 앞 K장(기본 200)의 사진별 예측을 pred_<tag>.npz로 남긴다.
  ORT 정수 구성(G3 누적 full/)은 dev 전체의 사진별 예측을 pred_<모델>.npz로 남겼다. dev 캐시 순서가 같으므로
  ORT 쪽의 앞 K장을 잘라 같은 사진끼리 짝지을 수 있다. 여기서는 계산만 한다(모델을 다시 돌리지 않음, 가벼운 작업).

[출력]  $RN50_RESULTS/compare_dfc_ort.jsonl
  구성별: 고화질 Δtop-1(FP32 대비, %p)과 짝 부트스트랩 95% 구간(PRD 5절, 데이터셋별 층화), 고화질 cos, CIFAR A Δtop-1
  구성 쌍별: 차이(b − a)와 95% 구간, 예측 일치율(같은 사진에서 같은 클래스를 고른 비율)
  ※ K장 표본이라 dev 전체 결과(G3 full)와 구간 폭이 다르다. 선택은 하지 않는다(팀 결정).

[사용]
  RN50_RESULTS=../results/siglip2/G4_dfc_screen "$SIGLIP2_PYTHON" s2_compare_dfc_ort.py \
      --dfc ../results/siglip2/G4_dfc_screen/pred_D_official_b9fc2.npz ... --ort ../results/siglip2/G3_cumulative/full/pred_<모델>.npz ...
"""
import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S                        # noqa: E402
from s2_paired_ci import DATA, HR, boot_hr  # noqa: E402
from scripts import env_info                # noqa: E402

RC = S.RC


def load(path, k):
    """npz → {데이터셋: {pred, cos}} (앞 k장). DFC npz는 <d>_pred·<d>_cos·<d>_y, ORT npz는 <d>_pred·<d>_correct·<d>_cos."""
    z = np.load(path)
    out = {}
    for d in DATA:
        pred, cos = z[f"{d}_pred"][:k], z[f"{d}_cos"][:k]
        assert len(pred) == k, (path, d, len(pred))
        y = np.load(RC.CACHE_DIR / f"{d}_y.npy")[:k]
        if f"{d}_y" in z.files:                                     # DFC: 정답이 들어 있으면 캐시 정답과 같은지 확인
            assert np.array_equal(z[f"{d}_y"][:k], y), (path, d)
        if f"{d}_correct" in z.files:                               # ORT: 정답 여부가 캐시 정답 순서와 맞는지 확인 (독립 검토 지적)
            assert np.array_equal(z[f"{d}_correct"][:k], pred == y), (path, d)
        out[d] = {"pred": pred, "cos": cos}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfc", nargs="*", default=[], help="DFC pred_<tag>.npz")
    ap.add_argument("--ort", nargs="*", default=[], help="ORT pred_<모델>.npz (dev 전체, 앞 k장만 씀)")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    y = {d: np.load(RC.CACHE_DIR / f"{d}_y.npy")[:a.k] for d in DATA}
    fp = {}
    for d in DATA:
        ref, W = np.load(RC.CACHE_DIR / f"{d}_ref_fp32.npy")[:a.k], np.load(RC.CACHE_DIR / f"{d}_clf.npy")
        fp[d] = {"pred": np.argmax(ref @ W, 1), "cos": np.ones(a.k, np.float32)}
    cfg = {}
    for p in a.dfc:
        cfg["dfc:" + Path(p).stem.removeprefix("pred_")] = load(p, a.k)
    for p in a.ort:
        cfg["ort:" + Path(p).stem.removeprefix("pred_siglip2_image_")] = load(p, a.k)
    base = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"], "dirty": env["git"]["dirty"],
            "k": a.k, "n_boot": a.n_boot, "files": a.dfc + a.ort}
    rows = []
    corr = lambda c, d: (c[d]["pred"] == y[d]).astype(float)
    for name, c in cfg.items():
        diff = {d: corr(c, d) - corr(fp, d) for d in HR}
        hr_raw = float(np.mean([100 * v.mean() for v in diff.values()]))
        ci = boot_hr(diff, a.n_boot)
        r = {**base, "config": name, "hr_dtop1": round(hr_raw, 2), "hr_dtop1_ci95": ci, "significant": not (ci[0] <= 0 <= ci[1]),
             "hr_cos": round(float(np.mean([c[d]["cos"].mean() for d in HR])), 4),
             "cifar_A_dtop1": round(100 * float(corr(c, "cifar_A").mean() - corr(fp, "cifar_A").mean()), 2),
             "dtop1": {d: round(100 * float(corr(c, d).mean() - corr(fp, d).mean()), 2) for d in DATA},
             "cos": {d: round(float(c[d]["cos"].mean()), 4) for d in DATA}}
        rows.append(r)
        print(json.dumps({k: r[k] for k in ("config", "hr_dtop1", "hr_dtop1_ci95", "hr_cos", "cifar_A_dtop1")}, ensure_ascii=False), flush=True)
    for (n1, c1), (n2, c2) in itertools.combinations(cfg.items(), 2):
        diff = {d: corr(c2, d) - corr(c1, d) for d in HR}
        ci = boot_hr(diff, a.n_boot)
        agree = float(np.mean([(c1[d]["pred"] == c2[d]["pred"]).mean() for d in HR]))
        rows.append({**base, "pair": [n1, n2], "hr_diff_b_minus_a": round(float(np.mean([100 * v.mean() for v in diff.values()])), 2),
                     "ci95": ci, "tie": ci[0] <= 0 <= ci[1], "hr_pred_agreement": round(agree, 4)})
        print("pair", n1, n2, rows[-1]["hr_diff_b_minus_a"], ci, f"agree {agree:.3f}", flush=True)
    with open(out / "compare_dfc_ort.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
