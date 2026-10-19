"""g5b_summarize.py — G5b 실칩 결과 정리: 정확도(dev 스크리닝 6×200) + 지연(배치 1·8, 3세션 중앙값) → results.jsonl·표 (G5b 6단계)
정확도 계산은 s2_hailo_chip_eval.py(--k 200)와 같다: L2 정규화 임베딩 · 클래스 임베딩(clf) argmax, FP32 기준 = 캐시 <set>_ref_fp32.npy.
고화질 Δtop-1의 95% CI는 s2_paired_ci.boot_hr와 같은 방식(데이터셋별 층화 재표집 1000회, 시드 0)으로 사진별 (칩 정답 − FP32 정답)에 건다.
사용: python g5b_summarize.py --chip <AGX out 폴더 사본> --cache $SIGLIP2_ARTIFACTS/cache --hefinfo hefinfo.json --out results.jsonl
"""
import argparse, json, subprocess
from pathlib import Path
import numpy as np

DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HR = DATA[1:]
K = 200


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def boot_hr(diff_by_data, n_boot=1000, seed=0):      # s2_paired_ci.boot_hr와 같은 코드
    rng = np.random.default_rng(seed)
    vals = [np.mean([100 * v[rng.integers(0, len(v), len(v))].mean() for v in diff_by_data.values()]) for _ in range(n_boot)]
    return [round(float(np.percentile(vals, 2.5)), 2), round(float(np.percentile(vals, 97.5)), 2)]


ap = argparse.ArgumentParser()
ap.add_argument("--chip", required=True); ap.add_argument("--cache", required=True)
ap.add_argument("--hefinfo", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--models", nargs="+", default=["M2rot2_D256L1", "vendor_v500"])
a = ap.parse_args()
C, chip = Path(a.cache), Path(a.chip)
repo = Path(__file__).resolve().parents[3]
git = lambda *x: subprocess.run(["git", "-C", str(repo), *x], capture_output=True, text=True).stdout.strip()
commit, dirty = git("rev-parse", "--short", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
hefinfo = json.loads(Path(a.hefinfo).read_text())
temps = [json.loads(l) for l in (chip / "temps.jsonl").read_text().splitlines() if l.strip()]
start_temp = {}
for t in temps:          # 세션 시작 온도 = 그 라벨의 마지막 wait_ 기록 (그 값이 기준 미만이라 시작함)
    if t.get("label", "").startswith("wait_"):
        start_temp[t["label"][5:]] = round(max(t["ts0_c"], t["ts1_c"]), 1)

P, rows = {}, {}
for m in a.models:
    info = json.loads((chip / f"eval_{m}" / "eval_info.json").read_text())
    z = np.load(chip / f"eval_{m}" / f"emb_{m}.npz")
    r, P[m] = {}, {}
    for d in DATA:
        y, W, ref = np.load(C / f"{d}_y.npy")[:K], np.load(C / f"{d}_clf.npy"), l2n(np.load(C / f"{d}_ref_fp32.npy")[:K])
        e = l2n(z[f"{d}_emb"].astype(np.float32))
        assert e.shape == ref.shape == (K, ref.shape[1]), (m, d, e.shape, ref.shape)
        pred, fp = np.argmax(e @ W, 1), np.argmax(ref @ W, 1)
        P[m][d] = {"ok": (pred == y).astype(float), "fp_ok": (fp == y).astype(float), "cos": (e * ref).sum(1), "pred": pred}
        r[d] = {"top1": round(100 * float((pred == y).mean()), 2), "fp32_top1": round(100 * float((fp == y).mean()), 2),
                "cos_mean": round(float(P[m][d]["cos"].mean()), 4)}
    hr_d = {d: P[m][d]["ok"] - P[m][d]["fp_ok"] for d in HR}
    lat = {}
    for b in (1, 8):
        runs = []
        for s in (1, 2, 3):
            f = chip / f"bench_b{b}_s{s}" / "bench.json"
            if f.exists():
                runs += [dict(x, session=s) for x in json.loads(f.read_text())["runs"] if x["model"] == m]
        if runs:
            lat[f"b{b}"] = {"mean_ms_med3": round(float(np.median([x["mean_ms"] for x in runs])), 3),
                            "median_ms_med3": round(float(np.median([x["median_ms"] for x in runs])), 3),
                            "img_per_s": round(1000 * b / float(np.median([x["mean_ms"] for x in runs])), 2),
                            "sessions": [{k: x[k] for k in ("session", "mean_ms", "median_ms", "p90_ms")} for x in runs],
                            "start_temp_c": [start_temp.get(f"bench_b{b}_s{s}") for s in (1, 2, 3)]}
    rows[m] = {"what": "g5b", "model": m, "commit": commit, "dirty": dirty, "hef": info["models"][m]["hef"],
               "hef_sha256": info["models"][m]["hef_sha256"], "input_domain": info["models"][m].get("input_domain"),
               "quant_in": info["models"][m]["quant_in"], "hailortcli": info["hailortcli"], "dfc": "5.4.0",
               "eval_start_temp_c": start_temp.get(f"eval_{m}"), "k": K, **hefinfo.get(m, {}),
               "hr_dtop1": round(float(np.mean([r[d]["top1"] - r[d]["fp32_top1"] for d in HR])), 2), "hr_dtop1_ci95": boot_hr(hr_d),
               "hr_cos": round(float(np.mean([r[d]["cos_mean"] for d in HR])), 4),
               "cifar_A_dtop1": round(r["cifar_A"]["top1"] - r["cifar_A"]["fp32_top1"], 2), "per_set": r, "latency": lat}
m0, m1 = a.models
diffs = {d: P[m1][d]["ok"] - P[m0][d]["ok"] for d in HR}
vs = {"what": "g5b_vs", "a": m0, "b": m1, "commit": commit, "dirty": dirty,
      "top1_diff": round(float(np.mean([100 * v.mean() for v in diffs.values()])), 2), "top1_diff_ci95": boot_hr(diffs),
      "hr_cos_diff": round(rows[m1]["hr_cos"] - rows[m0]["hr_cos"], 4),
      "pred_agree_hr": round(float(np.mean([np.mean(P[m1][d]["pred"] == P[m0][d]["pred"]) for d in HR])), 4)}
with open(a.out, "w") as f:
    for m in a.models:
        f.write(json.dumps(rows[m], ensure_ascii=False) + "\n")
    f.write(json.dumps(vs, ensure_ascii=False) + "\n")
print("| HEF | 크기 MB | 컨텍스트 | b1 ms | b1 FPS | b8 ms | b8 FPS | 고화질 Δtop-1 %p [95% CI] | 고화질 cos | CIFAR A Δ %p |")
print("|---|---|---|---|---|---|---|---|---|---|")
for m in a.models:
    r = rows[m]; L = r["latency"]; g = lambda b, k: L.get(b, {}).get(k, "—")
    print(f"| {m} | {r.get('size_mb', '—')} | {r.get('contexts', '—')} | {g('b1', 'mean_ms_med3')} | {g('b1', 'img_per_s')} | "
          f"{g('b8', 'mean_ms_med3')} | {g('b8', 'img_per_s')} | {r['hr_dtop1']:+.2f} [{r['hr_dtop1_ci95'][0]:+.2f}, {r['hr_dtop1_ci95'][1]:+.2f}] | "
          f"{r['hr_cos']:.4f} | {r['cifar_A_dtop1']:+.2f} |")
print(json.dumps(vs, ensure_ascii=False))
