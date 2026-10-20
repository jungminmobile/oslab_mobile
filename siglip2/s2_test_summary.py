"""
s2_test_summary.py — G6 test 결과 요약: 고화질 5종 평균 Δtop-1과 짝 부트스트랩 95% 구간 (PRD 5절과 같은 방법)
================================================================================================
입력: $RN50_RESULTS/test_<set>.npz (s2_test_eval.py 출력: y, fp32_pred, <모델>_pred, <모델>_cos)
출력: $RN50_RESULTS/test_summary.json, 표준 출력 표. 모델 실행 없음(저장된 예측만, 몇 초).
방법: s2_paired_ci.boot_hr와 같은 층화 부트스트랩(시드 0, 1000회). CIFAR-100·ImageNetV2는 따로 표기 (D-017).
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S                       # noqa: E402
from s2_paired_ci import boot_hr           # noqa: E402
from scripts import env_info               # noqa: E402

HR = ["pets", "flowers", "cars", "food", "caltech"]
OTHER = ["cifar_test", "imagenetv2"]


def main():
    out = S.require_results_dir()
    env = env_info.collect()
    z = {s: np.load(out / f"test_{s}.npz") for s in HR + OTHER}
    models = sorted({k[:-5] for k in z["pets"].files if k.endswith("_pred") and k != "fp32_pred"})
    res = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"], "dirty": env["git"]["dirty"],
           "n_boot": 1000, "seed": 0, "sets": {s: int(len(z[s]["y"])) for s in z}, "models": {}}
    for m in models:
        r = {}
        for s in HR + OTHER:
            y = z[s]["y"]
            fp, q = z[s]["fp32_pred"] == y, z[s][f"{m}_pred"] == y
            r[s] = {"fp32_top1": round(100 * float(fp.mean()), 2), "top1": round(100 * float(q.mean()), 2),
                    "dtop1": round(100 * float(q.mean() - fp.mean()), 2), "cos": round(float(z[s][f"{m}_cos"].mean()), 4),
                    "dtop1_ci95": boot_hr({s: q.astype(float) - fp.astype(float)})}
        diffs = {s: (z[s][f"{m}_pred"] == z[s]["y"]).astype(float) - (z[s]["fp32_pred"] == z[s]["y"]).astype(float) for s in HR}
        r["hr"] = {"fp32_top1": round(float(np.mean([r[s]["fp32_top1"] for s in HR])), 2),
                   "top1": round(float(np.mean([r[s]["top1"] for s in HR])), 2),
                   "dtop1": round(float(np.mean([100 * d.mean() for d in diffs.values()])), 2),
                   "dtop1_ci95": boot_hr(diffs), "cos": round(float(np.mean([r[s]["cos"] for s in HR])), 4)}
        res["models"][m] = r
        print(m)
        for s in HR + ["hr"] + OTHER:
            print(f"  {s:11s} FP32 {r[s]['fp32_top1']:6.2f}  모델 {r[s]['top1']:6.2f}  Δ {r[s]['dtop1']:+.2f} {r[s]['dtop1_ci95']}  cos {r[s]['cos']}")
    with open(out / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
