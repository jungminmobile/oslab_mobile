"""g5c_summarize.py — G5c 결과 정리: 구조(parse-hef) · 지연(3세션 중앙값) · 출력 일치(cos·top-1) → 판정(4-2) → results.jsonl·표
- FP32 기준·클래스 임베딩: g5c_siglip1_refs.py 출력 (<set>_ref_s1.npy, <set>_clf_s1.npy). 정답 y: dev 캐시 앞 200장.
- 고화질 = pets·flowers·cars·food·caltech 5종 평균 (데이터셋별 평균의 평균, s2_hailo_chip_eval.py와 같음). CIFAR A 따로.
사용: python g5c_summarize.py --chip <AGX out 사본> --refs <refs 폴더> --cache <cache> --sizes sizes.json --out results.jsonl
"""
import argparse, json, re, subprocess
from pathlib import Path
import numpy as np

DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HR = DATA[1:]
K = 200
l2n = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
ap = argparse.ArgumentParser()
ap.add_argument("--chip", required=True); ap.add_argument("--refs", required=True); ap.add_argument("--cache", required=True)
ap.add_argument("--sizes", required=True, help='{"zoo_public": bytes, ...}'); ap.add_argument("--out", required=True)
ap.add_argument("--models", nargs="+", default=["zoo_public", "R_our", "R_coco"])
a = ap.parse_args()
chip, R, C = Path(a.chip), Path(a.refs), Path(a.cache)
repo = Path(__file__).resolve().parents[3]
git = lambda *x: subprocess.run(["git", "-C", str(repo), *x], capture_output=True, text=True).stdout.strip()
base = {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "dfc": "5.4.0"}
sizes = json.loads(Path(a.sizes).read_text())
temps = [json.loads(l) for l in (chip / "temps.jsonl").read_text().splitlines() if l.strip()]
st = {t["label"][5:]: round(max(t["ts0_c"], t["ts1_c"]), 1) for t in temps if t["label"].startswith("wait_")}
M = [m for m in a.models if (chip / f"eval_{m}").exists()]
E, rows = {}, {}
for m in M:
    info = json.loads((chip / f"eval_{m}" / "eval_info.json").read_text())
    z = np.load(chip / f"eval_{m}" / f"emb_{m}.npz")
    ph = (chip / f"parsehef_{m}.txt").read_text()
    ctx = int(re.search(r"Number of contexts: (\d+)", ph).group(1))
    io = re.findall(r"(Input|Output)\s+\S+\s+(\w+), (\w+)\(([^)]*)\)", ph)
    r, E[m] = {}, {}
    for d in DATA:
        y = np.load(C / f"{d}_y.npy")[:K]; W = np.load(R / f"{d}_clf_s1.npy"); ref = l2n(np.load(R / f"{d}_ref_s1.npy")[:K])
        e = l2n(z[f"{d}_emb"].astype(np.float32)); assert e.shape == ref.shape, (m, d, e.shape)
        E[m][d] = e
        pred, fp = np.argmax(e @ W, 1), np.argmax(ref @ W, 1)
        c = (e * ref).sum(1)
        r[d] = {"top1": round(100 * float((pred == y).mean()), 2), "fp32_top1": round(100 * float((fp == y).mean()), 2),
                "cos_fp32_mean": round(float(c.mean()), 4), "cos_fp32_min": round(float(c.min()), 4)}
    lat = {}
    for b in (1, 8):
        runs = []
        for s in (1, 2, 3):
            f = chip / f"bench_b{b}_s{s}" / "bench.json"
            if f.exists():
                runs += [dict(x, session=s) for x in json.loads(f.read_text())["runs"] if x["model"] == m]
        if runs:
            mm = float(np.median([x["mean_ms"] for x in runs]))
            lat[f"b{b}"] = {"mean_ms_med3": round(mm, 3), "median_ms_med3": round(float(np.median([x["median_ms"] for x in runs])), 3),
                            "img_per_s": round(1000 * b / mm, 2), "sessions_mean_ms": [x["mean_ms"] for x in runs],
                            "start_temp_c": [st.get(f"bench_b{b}_s{s}") for s in (1, 2, 3)]}
    run2 = {}
    for b in (1, 8):
        mt = re.search(rf"run2 b{b} {m} .*fps: ([0-9.]+)", (chip / "run.log").read_text())
        run2[f"b{b}"] = float(mt.group(1)) if mt else None
    rows[m] = {"what": "g5c", "model": m, **base, "hef_sha256": info["models"][m]["hef_sha256"], "size_b": sizes[m], "contexts": ctx,
               "io": io, "quant_in": info["models"][m]["quant_in"], "input_domain": info["models"][m]["input_domain"], "hailortcli": info["hailortcli"],
               "eval_start_temp_c": st.get(f"eval_{m}"), "k": K,
               "hr_cos_fp32": round(float(np.mean([r[d]["cos_fp32_mean"] for d in HR])), 4), "cifar_A_cos_fp32": r["cifar_A"]["cos_fp32_mean"],
               "hr_dtop1": round(float(np.mean([r[d]["top1"] - r[d]["fp32_top1"] for d in HR])), 2),
               "cifar_A_dtop1": round(r["cifar_A"]["top1"] - r["cifar_A"]["fp32_top1"], 2), "per_set": r, "latency": lat, "run2_fps": run2}
pub = rows["zoo_public"]
for m in M:
    if m == "zoo_public":
        continue
    o = rows[m]
    vs = {}
    for d in DATA:
        c = (E[m][d] * E["zoo_public"][d]).sum(1)
        vs[d] = {"mean": round(float(c.mean()), 5), "min": round(float(c.min()), 5)}
    o["vs_public_cos"] = vs
    o["vs_public_cos_hr_mean"] = round(float(np.mean([vs[d]["mean"] for d in HR])), 5)
    o["vs_public_cos_hr_min"] = round(float(min(vs[d]["min"] for d in HR)), 5)
    o["vs_public_pred_agree_hr"] = round(float(np.mean([np.mean(np.argmax(E[m][d] @ np.load(R / f"{d}_clf_s1.npy"), 1)
                                                               == np.argmax(E["zoo_public"][d] @ np.load(R / f"{d}_clf_s1.npy"), 1)) for d in HR])), 4)
    g = {"contexts_same": o["contexts"] == pub["contexts"],
         "size_diff_pct": round(100 * (o["size_b"] - pub["size_b"]) / pub["size_b"], 2),
         "b1_diff_pct": round(100 * (o["latency"]["b1"]["mean_ms_med3"] - pub["latency"]["b1"]["mean_ms_med3"]) / pub["latency"]["b1"]["mean_ms_med3"], 2)
         if "b1" in o["latency"] and "b1" in pub["latency"] else None,
         "hr_cos_diff": round(o["hr_cos_fp32"] - pub["hr_cos_fp32"], 4)}
    g["size_ok"] = abs(g["size_diff_pct"]) <= 2.0
    g["b1_ok"] = g["b1_diff_pct"] is not None and abs(g["b1_diff_pct"]) <= 3.0
    g["cos_ok"] = abs(g["hr_cos_diff"]) <= 0.005
    struct_ok = g["contexts_same"] and g["size_ok"] and g["b1_ok"]
    g["verdict"] = "재현됨" if struct_ok and g["cos_ok"] else ("구조·속도만 재현" if struct_ok else "재현 안 됨")
    o["gate"] = g
with open(a.out, "w") as f:
    for m in M:
        f.write(json.dumps(rows[m], ensure_ascii=False) + "\n")
print("| HEF | 크기 MB | 컨텍스트 | b1 ms | b8 ms (장/초) | run2 b1/b8 FPS | 고화질 cos(FP32) | CIFAR A cos(FP32) | 고화질 Δtop-1 | CIFAR A Δ | vs 공개 cos 고화질 평균(최소) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for m in M:
    o = rows[m]; L = o["latency"]
    vp = f"{o['vs_public_cos_hr_mean']:.4f} ({o['vs_public_cos_hr_min']:.4f})" if "vs_public_cos" in o else "—"
    print(f"| {m} | {o['size_b'] / 1e6:.1f} | {o['contexts']} | {L.get('b1', {}).get('mean_ms_med3', '—')} | {L.get('b8', {}).get('mean_ms_med3', '—')} ({L.get('b8', {}).get('img_per_s', '—')}) | "
          f"{o['run2_fps']['b1']}/{o['run2_fps']['b8']} | {o['hr_cos_fp32']:.4f} | {o['cifar_A_cos_fp32']:.4f} | {o['hr_dtop1']:+.2f} | {o['cifar_A_dtop1']:+.2f} | {vp} |")
for m in M:
    if "gate" in rows[m]:
        print(m, json.dumps(rows[m]["gate"], ensure_ascii=False))
