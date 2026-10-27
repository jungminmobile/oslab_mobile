"""E14 ARM 확인 (준용 10/27): 285K(x86) ↔ AGX(ARM) portable_eval 비교, 모델 fp32·oab·f2·p2.
compare_junyong.py(10/23)와 같은 방식 + oab가 10/23 결과와 비트 단위로 같은지 + 판정(실행 전에 정한 기준).
사용: python compare_e14.py  (이 파일이 있는 폴더 기준) → compare_e14.json 출력"""
import json
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
OLD = H.parent  # 10/23 결과 (out_junyong_285k, out_junyong_agx_np244)
SETS = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HR = SETS[1:]
ENVS = ("285k", "agx")
MODELS = ["fp32", "oab", "f2", "p2"]

res = {e: json.load(open(H / f"out_junyong_{e}_e14/result.json")) for e in ENVS}
out = {"dataset": {}, "pair_285k_agx": {}, "per_env_cos": {}, "oab_vs_1023": {}, "verdict": {}}
for m in MODELS:
    out["dataset"][m] = {e: {**{d: res[e]["models"][m][d]["top1"] for d in SETS},
                             "hr_dtop1": res[e]["models"][m]["hr_dtop1"], "hr_cos": res[e]["models"][m]["hr_cos"],
                             "cos_mean": {d: res[e]["models"][m][d]["cos_mean"] for d in SETS}} for e in ENVS}
    z = {e: np.load(H / f"out_junyong_{e}_e14/pred_{m}.npz") for e in ENVS}
    row = {}
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
    for e in ENVS:
        c = np.concatenate([z[e][f"{d}_cos"] for d in SETS]).astype(np.float64)
        ch = np.concatenate([z[e][f"{d}_cos"] for d in HR]).astype(np.float64)
        out["per_env_cos"].setdefault(m, {})[e] = {"all_min": float(c.min()), "all_mean": float(c.mean()),
                                                   "hr_min": float(ch.min()), "hr_mean": float(ch.mean())}

# oab·fp32: 10/23 같은 환경 결과와 npz 모든 배열이 비트 단위로 같은지 (AGX는 같은 venv_ort125_np244)
for m in ("fp32", "oab"):
    for e, old in (("285k", "out_junyong_285k"), ("agx", "out_junyong_agx_np244")):
        a, b = np.load(H / f"out_junyong_{e}_e14/pred_{m}.npz"), np.load(OLD / old / f"pred_{m}.npz")
        same = sorted(a.files) == sorted(b.files) and all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]) for k in a.files)
        out["oab_vs_1023"].setdefault(m, {})[e] = {"ref": old, "bit_identical": bool(same)}

# 판정 (실행 전에 정함): AGX P2 고화질 Δ가 285K P2와 ±0.5%p 이내 + 다른 사진 수가 같은 실행 O-AB의 2배 이하
d_hr = abs(res["agx"]["models"]["p2"]["hr_dtop1"] - res["285k"]["models"]["p2"]["hr_dtop1"])
n_p2, n_oab = out["pair_285k_agx"]["p2"]["total"]["pred_diff"], out["pair_285k_agx"]["oab"]["total"]["pred_diff"]
ok = d_hr <= 0.5 + 1e-9 and n_p2 <= 2 * n_oab
out["verdict"] = {"p2_hr_dtop1_gap": round(d_hr, 4), "p2_pred_diff": n_p2, "oab_pred_diff": n_oab,
                  "ratio": round(n_p2 / n_oab, 3) if n_oab else None, "result": "통과" if ok else "ARM 주의"}
json.dump(out, open(H / "compare_e14.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ("pair_285k_agx", "oab_vs_1023", "verdict")}, indent=1, ensure_ascii=False))
