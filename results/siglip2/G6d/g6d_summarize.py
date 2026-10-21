"""g6d_summarize.py — G6d (G6 g6_summarize.py에서 PAIRS만 바꿈: 각 구성 − Z·M_both, Vrot − V, 원인 분해 S1−V·S2−S1·M_both−S2) 실칩 dev 전체 임베딩 → HEF별 지표 + 쌍 비교 (README 1·5-2). 계산은 G5b g5b_devfull_summarize.py와 같다.
구간: s2_paired_ci.boot_hr 그대로 import (층화 짝 부트스트랩, 시드 0, 1000회). FP32 기준: s2_g6.py fp32 (HF FP32, Zoo 전처리).
사용: python g6_summarize.py --chip <AGX out 사본> --ref ~/siglip2_work/g6/ref --s2dir ../../../siglip2 --out results_chip.jsonl
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HR = DATA[1:]
MODEL_OF = lambda h: h.split("_")[0]          # b32_M_z16 → b32, b16_M2rot2 → b16
NEW = ["S1", "S2", "Vrot", "G", "Dmax2", "F"]
PAIRS = {"b32": [(x, "Z") for x in NEW] + [(x, "M_both") for x in NEW] + [("Vrot", "V"), ("S1", "V"), ("S2", "S1"), ("M_both", "S2"), ("M_both", "Z"), ("S1", "Vrot")],   # S1−Vrot: 기록용 (회전을 뺀 그래프 출처)
         "s1b16": [("R", "Z"), ("R", "V"), ("V", "Z"), ("R", "D"), ("D", "V")]}
ap = argparse.ArgumentParser()
ap.add_argument("--chip", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--s2dir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, a.s2dir)
from s2_paired_ci import boot_hr          # noqa: E402
l2n = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
repo = Path(__file__).resolve().parents[3]
git = lambda *x: subprocess.run(["git", "-C", str(repo), *x], capture_output=True, text=True).stdout.strip()
base = {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "s2_paired_ci_sha256": hashlib.sha256((Path(a.s2dir) / "s2_paired_ci.py").read_bytes()).hexdigest()[:16], "n_boot": 1000, "seed": 0}
chip, R = Path(a.chip), Path(a.ref)
temps = [json.loads(l) for l in (chip / "temps.jsonl").read_text().splitlines() if l.strip()]
st = {t["label"][5:]: round(max(t["ts0_c"], t["ts1_c"]), 1) for t in temps if t["label"].startswith("wait_") and "ts0_c" in t}
hefs = sorted(p.name[5:] for p in chip.glob("eval_*") if (p / "eval_info.json").exists())
P, rows = {}, []
for h in hefs:
    m = MODEL_OF(h)
    info = json.loads((chip / f"eval_{h}" / "eval_info.json").read_text())["models"][h]
    z = np.load(chip / f"eval_{h}" / f"emb_{h}.npz")
    r, P[h] = {}, {}
    for d in DATA:
        y, W, ref = np.load(R / f"{m}_{d}_y.npy"), np.load(R / f"{m}_{d}_clf.npy"), l2n(np.load(R / f"{m}_{d}_ref_fp32.npy"))
        e = l2n(z[f"{d}_emb"].astype(np.float32)); assert e.shape == ref.shape, (h, d, e.shape, ref.shape)
        pred, fp = np.argmax(e @ W, 1), np.argmax(ref @ W, 1)
        c = (e * ref).sum(1)
        P[h][d] = {"ok": (pred == y).astype(float), "fp": (fp == y).astype(float), "pred": pred}
        r[d] = {"n": int(len(y)), "top1": round(100 * float((pred == y).mean()), 2), "fp32_top1": round(100 * float((fp == y).mean()), 2),
                "cos_mean": round(float(c.mean()), 4), "cos_min": round(float(c.min()), 4)}
    rows.append({"what": "g6_chip", "hef": h, "model": m, **base, "hef_sha256": info["hef_sha256"], "input_domain": info["input_domain"],
                 "in_shape": info["in_shape"], "quant_in": info["quant_in"], "start_temp_c": st.get(f"eval_{h}"),
                 "hr_n": sum(r[d]["n"] for d in HR), "hr_top1": round(float(np.mean([100 * P[h][d]["ok"].mean() for d in HR])), 2),
                 "hr_fp32_top1": round(float(np.mean([100 * P[h][d]["fp"].mean() for d in HR])), 2),
                 "hr_dtop1": round(float(np.mean([100 * (P[h][d]["ok"].mean() - P[h][d]["fp"].mean()) for d in HR])), 2),   # 반올림 전 값으로 평균 (독립 검토)
                 "hr_dtop1_ci95": boot_hr({d: P[h][d]["ok"] - P[h][d]["fp"] for d in HR}),
                 "hr_cos": round(float(np.mean([r[d]["cos_mean"] for d in HR])), 4), "hr_cos_min": min(r[d]["cos_min"] for d in HR),
                 "cifar_A_dtop1": round(100 * float(P[h]["cifar_A"]["ok"].mean() - P[h]["cifar_A"]["fp"].mean()), 2), "cifar_A_cos": r["cifar_A"]["cos_mean"],
                 "per_set": r})
for m, pairs in PAIRS.items():
    for x, y in pairs:
        hx, hy = f"{m}_{x}", f"{m}_{y}"
        if hx not in P or hy not in P:
            continue
        diff = {d: P[hx][d]["ok"] - P[hy][d]["ok"] for d in HR}
        dm = round(float(np.mean([100 * v.mean() for v in diff.values()])), 2); ci = boot_hr(diff)
        verdict = "우위" if ci[0] > 0 else ("열위" if ci[1] < 0 else "동률")
        rows.append({"what": "g6_pair", "model": m, "a": x, "b": y, "pair": f"{x}-{y}", **base, "hr_diff": dm, "ci95": ci, "verdict": verdict,
                     "cifar_A_diff": round(100 * float((P[hx]["cifar_A"]["ok"] - P[hy]["cifar_A"]["ok"]).mean()), 2),
                     "pred_agree_hr": round(float(np.mean([np.mean(P[hx][d]["pred"] == P[hy][d]["pred"]) for d in HR])), 4)})
with open(a.out, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for r in rows:
    if r["what"] == "g6_chip":
        print(r["hef"], "hr", r["hr_top1"], "Δ", r["hr_dtop1"], r["hr_dtop1_ci95"], "cos", r["hr_cos"], "min", r["hr_cos_min"], "cifarΔ", r["cifar_A_dtop1"])
    else:
        print(r["model"], r["pair"], r["hr_diff"], r["ci95"], r["verdict"], "cifar", r["cifar_A_diff"])
