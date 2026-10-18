"""
s2_xplat_compare.py — 같은 ONNX·같은 dev200 사진을 여러 CPU에서 돌린 portable_eval 결과를 사진별로 짝지어 비교 (10/18, 준용 PR #8)
=========================================================================================================
입력: --runs 이름=<portable_eval 출력 폴더> 여러 개 (pred_fp32/s8/u8rr.npz: <set>_pred, <set>_cos, <set>_y)
출력: $RN50_RESULTS/xplat_compare.json + 표준 출력
  - 환경·모델별: 고화질 5종 Δtop-1(FP32 대비) 과 짝 부트스트랩 95% 구간(데이터셋별 층화, 시드 0, 1000회 — PRD 5절)
  - 환경 쌍·모델별: 예측 일치율, 사진별 cos 차이(평균·최대), 정답률 차이와 95% 구간
모델 실행 없음 (저장된 예측만, 몇 초).
"""
import argparse
import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S                 # noqa: E402
from s2_paired_ci import boot_hr     # noqa: E402
from scripts import env_info         # noqa: E402

HR = ["pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
MODELS = ["s8", "u8rr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="이름=폴더")
    ap.add_argument("--models", nargs="+", default=MODELS, help="pred_<이름>.npz의 모델 이름들 (기본 s8 u8rr, fp32는 항상 포함)")
    ap.add_argument("--vs", default="", help="(기본 꺼짐) 같은 환경 안에서 이 모델 대비 cos 차이·top-1 짝 차이도 낸다 (예: r0_410d, G3Q)")
    ap.add_argument("--out-name", default="xplat_compare.json")
    a = ap.parse_args()
    models = list(a.models)
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    runs = {s.split("=", 1)[0]: Path(s.split("=", 1)[1]) for s in a.runs}
    P = {r: {m: np.load(d / f"pred_{m}.npz") for m in ["fp32"] + models} for r, d in runs.items()}
    res = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"], "dirty": env["git"]["dirty"],
           "runs": {r: str(d) for r, d in runs.items()}, "per_run": {}, "pairs": {}}
    for r in runs:
        res["per_run"][r] = {}
        for m in models:
            diffs = {d: (P[r][m][f"{d}_pred"] == P[r][m][f"{d}_y"]).astype(float) - (P[r]["fp32"][f"{d}_pred"] == P[r]["fp32"][f"{d}_y"]).astype(float) for d in HR}
            res["per_run"][r][m] = {"hr_dtop1": round(float(np.mean([100 * v.mean() for v in diffs.values()])), 2), "hr_dtop1_ci95": boot_hr(diffs),
                                    "hr_cos": round(float(np.mean([P[r][m][f"{d}_cos"].mean() for d in HR])), 4)}
            print(f"{r:12s} {m:5s} Δ {res['per_run'][r][m]['hr_dtop1']:+.2f} {res['per_run'][r][m]['hr_dtop1_ci95']}  cos {res['per_run'][r][m]['hr_cos']}")
    for r1, r2 in itertools.combinations(runs, 2):
        for m in ["fp32"] + models:
            agree = np.mean([np.mean(P[r1][m][f"{d}_pred"] == P[r2][m][f"{d}_pred"]) for d in HR])
            cd = np.concatenate([np.abs(P[r1][m][f"{d}_cos"] - P[r2][m][f"{d}_cos"]) for d in HR])
            diffs = {d: (P[r2][m][f"{d}_pred"] == P[r2][m][f"{d}_y"]).astype(float) - (P[r1][m][f"{d}_pred"] == P[r1][m][f"{d}_y"]).astype(float) for d in HR}
            row = {"pred_agree": round(float(agree), 4), "cos_absdiff_mean": float(cd.mean()), "cos_absdiff_max": float(cd.max()),
                   "top1_diff": round(float(np.mean([100 * v.mean() for v in diffs.values()])), 2), "top1_diff_ci95": boot_hr(diffs)}
            res["pairs"][f"{r1}->{r2}:{m}"] = row
            print(f"{r1}->{r2} {m:5s} 일치 {row['pred_agree']:.4f}  cos차 평균 {row['cos_absdiff_mean']:.2e} 최대 {row['cos_absdiff_max']:.2e}  top-1 차 {row['top1_diff']:+.2f} {row['top1_diff_ci95']}")
    if a.vs:
        res["vs"] = {}
        for r in runs:
            for m in models:
                if m == a.vs:
                    continue
                diffs = {d: (P[r][m][f"{d}_pred"] == P[r][m][f"{d}_y"]).astype(float) - (P[r][a.vs][f"{d}_pred"] == P[r][a.vs][f"{d}_y"]).astype(float) for d in HR}
                row = {"hr_cos_diff": round(float(np.mean([P[r][m][f"{d}_cos"].mean() - P[r][a.vs][f"{d}_cos"].mean() for d in HR])), 4),
                       "top1_diff": round(float(np.mean([100 * v.mean() for v in diffs.values()])), 2), "top1_diff_ci95": boot_hr(diffs)}
                res["vs"][f"{r}:{m}-{a.vs}"] = row
                print(f"{r:12s} {m}-{a.vs}  cos 차 {row['hr_cos_diff']:+.4f}  top-1 차 {row['top1_diff']:+.2f} {row['top1_diff_ci95']}")
    with open(out / a.out_name, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
