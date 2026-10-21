"""pairs.py — E11b: 저장된 pred_*.npz로 R0(노트북 E5a 실행) 대비 짝 부트스트랩 (s2_paired_ci.boot_hr와 같은 방식, 1000회 층화)
사용: python pairs.py --base C:\\s2art\\E5a\\pred_<R0>.npz --pred C:\\s2art\\E11b\\pred_<A>.npz ... --out C:\\s2art\\E11b\\pairs.jsonl"""
import argparse
import json
from pathlib import Path

import numpy as np

HR = ["pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]


def boot(diff, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = [np.mean([100 * d[rng.integers(0, len(d), len(d))].mean() for d in diff.values()]) for _ in range(n)]
    return [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]


ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--pred", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
b = np.load(a.base)
for p in a.pred:
    z = np.load(p)
    diff = {d: z[f"{d}_correct"].astype(float) - b[f"{d}_correct"] for d in HR}
    row = {"base": Path(a.base).stem, "model": Path(p).stem, "hr_diff": round(float(np.mean([100 * v.mean() for v in diff.values()])), 2),
           "ci95": boot(diff), "per_data": {d: round(100 * float(v.mean()), 2) for d, v in diff.items()},
           "cifar_diff": round(100 * float(z["cifar_A_correct"].mean() - b["cifar_A_correct"].mean()), 2),
           "hr_cos": round(float(np.mean([z[f"{d}_cos"].mean() for d in HR])), 4),
           "hr_top1": round(float(np.mean([100 * z[f"{d}_correct"].mean() for d in HR])), 2)}
    print(json.dumps(row, ensure_ascii=False))
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
