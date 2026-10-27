"""xplat_compare3.py — 같은 Release 묶음(dev 6종 × 200장)을 세 CPU에서 portable_eval로 돌린 사진별 예측 비교 (2026-10-24)
i7-1165G7(이 PC, AVX-512 VNNI, Windows) / Ultra 9 285K(준용, avx_vnni, WSL2) / Jetson AGX Orin(준용, ARM dotprod)
준용 npz는 origin/siglip2/b32-portable-junyong 에서 git show로 읽는다. 무거운 계산 없음.
"""
import io, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
X = ROOT / "results/siglip2/G8_candidates/b32/xplat"
OUT = X / "compare_3way"
BR = "origin/siglip2/b32-portable-junyong"


def git_npz(rel):
    b = subprocess.run(["git", "-C", str(ROOT), "show", f"{BR}:{rel}"], capture_output=True, check=True).stdout
    return dict(np.load(io.BytesIO(b)))


rel = "results/siglip2/G8_candidates/b32/xplat"
env = {"i7": lambda m: dict(np.load(X / f"i7-1165G7/pred_{m}.npz")),
       "285k": lambda m: git_npz(f"{rel}/out_junyong_285k/pred_{m}.npz"),
       "agx": lambda m: git_npz(f"{rel}/out_junyong_agx/pred_{m}.npz")}
res = {"branch_junyong": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", BR], capture_output=True, text=True).stdout.strip(),
       "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(), "models": {}}
for m in ("fp32", "oab", "m2rot"):
    d = {k: f(m) for k, f in env.items()}
    keys = sorted(k for k in d["i7"] if k.endswith("_pred"))
    assert all(sorted(k for k in d[e] if k.endswith("_pred")) == keys for e in d), "키 다름"
    out = {}
    for a, b in (("i7", "285k"), ("i7", "agx"), ("285k", "agx")):
        diff, flip, per, n, cmax = 0, 0, {}, 0, 0.0
        for k in keys:
            pa, pb = d[a][k], d[b][k]
            y = d[a][k.replace("_pred", "_y")] if k.replace("_pred", "_y") in d[a] else None
            assert pa.shape == pb.shape
            n += len(pa)
            dk = int((pa != pb).sum())
            per[k.removesuffix("_pred")] = dk
            diff += dk
            cmax = max(cmax, float(np.abs(d[a][k.replace("_pred", "_cos")] - d[b][k.replace("_pred", "_cos")]).max()))
            if y is not None:
                flip += int(((pa == y) != (pb == y)).sum())
        out[f"{a}_vs_{b}"] = {"n": n, "pred_diff": diff, "correct_flip": flip, "per_set": per, "max_abs_cos_diff": cmax,
                              "bit_identical_pred": diff == 0}
    res["models"][m] = out
    res["keys"] = list(d["i7"].keys())
OUT.mkdir(parents=True, exist_ok=True)
json.dump(res, open(OUT / "compare_3way.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
