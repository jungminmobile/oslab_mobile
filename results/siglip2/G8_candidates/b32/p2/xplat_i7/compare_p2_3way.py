"""compare_p2_3way.py — P2·O-AB portable_eval(dev200) 사진별 예측: 이 PC i7 vs 준용 285K·AGX (origin/exp/E14-arm-junyong) (2026-10-27)
test 실행 중이라 siglip2/ 밖에 둔다. 사용: "$SIGLIP2_PYTHON" results/siglip2/G8_candidates/b32/p2/xplat_i7/compare_p2_3way.py
"""
import io, json, subprocess
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
BR = "origin/exp/E14-arm-junyong"
REL = "results/siglip2/G8_candidates/b32/xplat/E14"
def gnpz(p):
    return dict(np.load(io.BytesIO(subprocess.run(["git", "-C", str(ROOT), "show", f"{BR}:{p}"], capture_output=True, check=True).stdout)))
agx_dir = [l for l in subprocess.run(["git", "-C", str(ROOT), "ls-tree", "-d", "--name-only", f"{BR}:{REL}"], capture_output=True, text=True).stdout.split() if "agx" in l]
res = {"branch": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", BR], capture_output=True, text=True).stdout.strip(), "agx_dir": agx_dir, "models": {}}
for m in ("oab", "p2"):
    d = {"i7": dict(np.load(HERE / f"pred_{m}.npz")), "285k": gnpz(f"{REL}/out_junyong_285k_e14/pred_{m}.npz"), "agx": gnpz(f"{REL}/{agx_dir[0]}/pred_{m}.npz")}
    keys = sorted(k for k in d["i7"] if k.endswith("_pred"))
    out = {}
    for a, b in (("i7", "285k"), ("i7", "agx"), ("285k", "agx")):
        per = {k[:-5]: int((d[a][k] != d[b][k]).sum()) for k in keys}
        flip = sum(int(((d[a][k] == d[a][k[:-5] + "_y"]) != (d[b][k] == d[b][k[:-5] + "_y"])).sum()) for k in keys)
        cmax = max(float(np.abs(d[a][k[:-5] + "_cos"] - d[b][k[:-5] + "_cos"]).max()) for k in keys)
        out[f"{a}_vs_{b}"] = {"pred_diff": sum(per.values()), "correct_flip": flip, "max_abs_cos_diff": cmax, "per_set": per}
    res["models"][m] = out
json.dump(res, open(HERE / "compare_p2_3way.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for m, o in res["models"].items():
    for k, v in o.items(): print(m, k, v["pred_diff"], v["correct_flip"], round(v["max_abs_cos_diff"], 4), v["per_set"])
