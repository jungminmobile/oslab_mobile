"""portable_eval 결과 비교 (준용 10/23): 285K(x86) · AGX(ARM) · 준혁 i7(result.json만).
사진별 비교는 pred_*.npz가 있는 285K와 AGX 사이만 가능하다 (i7은 데이터셋별 값만 공개됨).
사용: python compare_junyong.py  (이 파일이 있는 폴더 기준) → compare_junyong.json 출력"""
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
SETS = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HR = SETS[1:]
ENV = {"i7": H / "i7-1165G7/result.json", "285k": H / "out_junyong_285k/result.json", "agx": H / "out_junyong_agx/result.json"}
MODELS = ["fp32", "oab", "m2rot"]

res = {k: json.load(open(p)) for k, p in ENV.items()}
out = {"dataset": {}, "pair_285k_agx": {}, "per_env_cos": {}, "i7_lower_bound": {}}
for m in MODELS:
    out["dataset"][m] = {e: {**{d: res[e]["models"][m][d]["top1"] for d in SETS},
                             "hr_dtop1": res[e]["models"][m]["hr_dtop1"], "hr_cos": res[e]["models"][m]["hr_cos"],
                             "cos_mean": {d: res[e]["models"][m][d]["cos_mean"] for d in SETS}} for e in ENV}
    # i7과 정답 수가 다르면 그 차이만큼은 사진별 top-1이 달라야 한다 (하한)
    for e in ("285k", "agx"):
        lb = {d: round(abs(res[e]["models"][m][d]["top1"] - res["i7"]["models"][m][d]["top1"]) * res[e]["models"][m][d]["n"] / 100)
              for d in SETS}
        out["i7_lower_bound"].setdefault(m, {})[e] = {**lb, "total": sum(lb.values())}
    z = {e: np.load(H / f"out_junyong_{e}/pred_{m}.npz") for e in ("285k", "agx")}
    row, cosr = {}, {}
    for d in SETS:
        a, b = z["285k"], z["agx"]
        assert (a[f"{d}_y"] == b[f"{d}_y"]).all()
        y = a[f"{d}_y"]
        diff = a[f"{d}_pred"] != b[f"{d}_pred"]
        corr = (a[f"{d}_pred"] == y) != (b[f"{d}_pred"] == y)
        dc = np.abs(a[f"{d}_cos"].astype(np.float64) - b[f"{d}_cos"].astype(np.float64))
        row[d] = {"n": int(len(y)), "pred_diff": int(diff.sum()), "correct_diff": int(corr.sum()),
                  "dcos_max": float(dc.max()), "dcos_mean": float(dc.mean())}
    tot = {"n": sum(r["n"] for r in row.values()), "pred_diff": sum(r["pred_diff"] for r in row.values()),
           "correct_diff": sum(r["correct_diff"] for r in row.values()),
           "dcos_max": max(r["dcos_max"] for r in row.values())}
    out["pair_285k_agx"][m] = {**row, "total": tot}
    for e in ("285k", "agx"):
        c = np.concatenate([z[e][f"{d}_cos"] for d in SETS]).astype(np.float64)
        ch = np.concatenate([z[e][f"{d}_cos"] for d in HR]).astype(np.float64)
        out["per_env_cos"].setdefault(m, {})[e] = {"all_min": float(c.min()), "all_mean": float(c.mean()),
                                                   "hr_min": float(ch.min()), "hr_mean": float(ch.mean())}
json.dump(out, open(H / "compare_junyong.json", "w"), indent=1)
print(json.dumps(out, indent=1))
