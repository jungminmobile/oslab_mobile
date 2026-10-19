"""g5b_devfull_summarize.py — 작업 B: dev 전체 실칩 임베딩(M2rot2, 벤더) → 지표 + 짝 비교 (s2_paired_ci.boot_hr 그대로 import: 층화 짝 부트스트랩, 시드 0, 1000회)
정확도 계산은 s2_hailo_chip_eval.py(--k 0)와 같다. 예전 M2rot2 dev 전체 실칩 임베딩(--old-m2rot)이 있으면 새 실행과 같은지 확인만 한다.
사용: python g5b_devfull_summarize.py --chip <out_devfull 사본> --cache <cache> --s2dir ~/oslab_m2rot/siglip2 --out results_devfull.jsonl [--old-m2rot emb.npz]
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HR = DATA[1:]
ap = argparse.ArgumentParser()
ap.add_argument("--chip", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--s2dir", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--old-m2rot", default="")
a = ap.parse_args()
sys.path.insert(0, a.s2dir)
from s2_paired_ci import boot_hr          # noqa: E402  (기존 함수 그대로)
l2n = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
repo = Path(__file__).resolve().parents[3]
git = lambda *x: subprocess.run(["git", "-C", str(repo), *x], capture_output=True, text=True).stdout.strip()
base = {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "s2_paired_ci_sha256": hashlib.sha256((Path(a.s2dir) / "s2_paired_ci.py").read_bytes()).hexdigest()[:16], "n_boot": 1000, "seed": 0}
C, chip = Path(a.cache), Path(a.chip)
temps = [json.loads(l) for l in (chip / "temps.jsonl").read_text().splitlines() if l.strip()]
st = {t["label"][5:]: round(max(t["ts0_c"], t["ts1_c"]), 1) for t in temps if t["label"].startswith("wait_")}
M = ["M2rot2_D256L1", "vendor_v500"]
P, rows = {}, []
for m in M:
    info = json.loads((chip / f"eval_{m}" / "eval_info.json").read_text())
    z = np.load(chip / f"eval_{m}" / f"emb_{m}.npz")
    r, P[m] = {}, {}
    for d in DATA:
        y, W, ref = np.load(C / f"{d}_y.npy"), np.load(C / f"{d}_clf.npy"), l2n(np.load(C / f"{d}_ref_fp32.npy"))
        e = l2n(z[f"{d}_emb"].astype(np.float32)); assert e.shape == ref.shape, (m, d, e.shape, ref.shape)
        pred, fp = np.argmax(e @ W, 1), np.argmax(ref @ W, 1)
        P[m][d] = {"ok": (pred == y).astype(float), "fp": (fp == y).astype(float), "pred": pred, "e": e}
        r[d] = {"n": int(len(y)), "top1": round(100 * float((pred == y).mean()), 2), "fp32_top1": round(100 * float((fp == y).mean()), 2),
                "cos_mean": round(float((e * ref).sum(1).mean()), 4)}
    rows.append({"what": "g5b_devfull", "model": m, **base, "hef_sha256": info["models"][m]["hef_sha256"], "input_domain": info["models"][m]["input_domain"],
                 "quant_in": info["models"][m]["quant_in"], "hailortcli": info["hailortcli"], "dfc": "5.4.0", "start_temp_c": st.get(f"eval_{m}"),
                 "hr_n": sum(r[d]["n"] for d in HR), "hr_dtop1": round(float(np.mean([r[d]["top1"] - r[d]["fp32_top1"] for d in HR])), 2),
                 "hr_dtop1_ci95": boot_hr({d: P[m][d]["ok"] - P[m][d]["fp"] for d in HR}),
                 "hr_cos": round(float(np.mean([r[d]["cos_mean"] for d in HR])), 4),
                 "cifar_A_dtop1": round(r["cifar_A"]["top1"] - r["cifar_A"]["fp32_top1"], 2), "per_set": r})
diff = {d: P["M2rot2_D256L1"][d]["ok"] - P["vendor_v500"][d]["ok"] for d in HR}
d_mean = round(float(np.mean([100 * v.mean() for v in diff.values()])), 2); ci = boot_hr(diff)
verdict = "M2rot2가 유의하게 높다" if ci[0] > 0 else ("벤더가 높다" if ci[1] < 0 else "동률")
vs = {"what": "g5b_devfull_vs", "a": "M2rot2_D256L1", "b": "vendor_v500", **base, "m2rot_minus_vendor_hr_top1": d_mean, "ci95": ci, "verdict": verdict,
      "cifar_A_diff": round(rows[0]["per_set"]["cifar_A"]["top1"] - rows[1]["per_set"]["cifar_A"]["top1"], 2),
      "pred_agree_hr": round(float(np.mean([np.mean(P["M2rot2_D256L1"][d]["pred"] == P["vendor_v500"][d]["pred"]) for d in HR])), 4)}
if a.old_m2rot:
    o = np.load(a.old_m2rot)
    vs["old_m2rot_same"] = {d: bool(np.array_equal(o[f"{d}_emb"], np.load(chip / "eval_M2rot2_D256L1" / "emb_M2rot2_D256L1.npz")[f"{d}_emb"])) for d in DATA}
with open(a.out, "w") as f:
    for r in rows + [vs]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
for r in rows:
    print(r["model"], "hr_dtop1", r["hr_dtop1"], r["hr_dtop1_ci95"], "hr_cos", r["hr_cos"], "cifarA", r["cifar_A_dtop1"],
          {d: (v["top1"], v["fp32_top1"], v["cos_mean"]) for d, v in r["per_set"].items()})
print(json.dumps(vs, ensure_ascii=False))
