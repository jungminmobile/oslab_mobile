"""O1 결과 요약: 지연(세션별 평균의 중앙값)·메모리·크기·정확도를 모아 판정(README 4절)과 표를 낸다.
사용 (results/siglip2/O1/):  python o1_summarize.py  → o1_summary.json + 표 출력 (markdown)
"""
import json
import statistics
from pathlib import Path

D = Path(__file__).resolve().parent
NAMES = ["fp", "base", "oa", "ob", "oab"]
LABEL = {"fp": "FP32", "base": "기준", "oa": "O-A", "ob": "O-B", "oab": "O-AB"}
THREADS = [1, 2, 4]

# 지연: s2_g6b_bench가 남긴 speed_*.json (세션 3 × 스레드 3)
runs = [json.loads(p.read_text()) for p in sorted((D / "speed").glob("speed_*.json"))]
assert all(not json.loads(l)["dirty"] for l in (D / "speed" / "runs.jsonl").read_text().splitlines())
lat = {}
for r in runs:
    t = r["cond"]["threads"]
    for k, v in r["results"].items():
        lat.setdefault(k, {}).setdefault(t, []).append(v["ms_per_img_mean"])
for k in lat:
    for t in THREADS:
        assert len(lat[k][t]) == 3, (k, t, len(lat[k][t]))
med = {k: {t: statistics.median(v[t]) for t in THREADS} for k, v in lat.items()}
size_mib = {k: v["MiB"] for r in runs[:1] for k, v in r["results"].items()}

# 메모리
mem = {}
for line in (D / "model_stats" / "model_stats.jsonl").read_text().splitlines():
    r = json.loads(line)
    for k, v in r["files"].items():
        mem[k] = {"infer": v["memory"]["peak_during_infer_MiB"], "load": v["memory"]["peak_during_load_MiB"]}

# 정확도
acc, pair = {}, {}
for line in (D / "accuracy" / "ort_accuracy.jsonl").read_text().splitlines():
    r = json.loads(line)
    assert not r["dirty"]
    if r["what"] == "g6b_ort":
        acc[r["model"][2:]] = r
    else:
        pair[(r["int8"][2:], r["fp32"][2:])] = r

out = {}
for b in ("b16", "b32"):
    rows = {}
    for n in NAMES:
        k = b + n
        row = {"label": LABEL[n], "MiB": size_mib[k], "lat": med[k], "lat_sessions": lat[k], "mem": mem.get(k),
               "hr_top1": acc[k]["hr_top1"], "cifar_A": acc[k]["cifar_A_top1"], "sha256": acc[k]["sha256"]}
        if n != "fp":
            p = pair[(k, b + "fp")]
            row["vs_fp"] = {"d": p["hr_dtop1"], "ci": p["ci95"], "cifar": p["cifar_A_dtop1"], "cos": p["hr_cos"]}
            row["speedup_vs_fp"] = {t: med[b + "fp"][t] / med[k][t] for t in THREADS}
        if n not in ("fp", "base"):
            p = pair[(k, b + "base")]
            row["vs_base"] = {"d": p["hr_dtop1"], "ci": p["ci95"], "cifar": p["cifar_A_dtop1"], "cos": p["hr_cos"],
                              "cos_min": p["hr_cos_min"]}
            red = {t: (med[b + "base"][t] - med[k][t]) / med[b + "base"][t] * 100 for t in THREADS}
            row["reduction_pct"] = red
            c1 = all(red[t] >= 3.0 for t in THREADS)
            c2 = row["vs_base"]["ci"][1] >= 0
            c3 = row["vs_base"]["cifar"] >= -1.0
            row["criteria"] = {"1_speed_all_ge3pct": c1, "2_hr_tie_or_better": c2, "3_cifar_ge_-1": c3, "adopt": c1 and c2 and c3}
        rows[n] = row
    adopt = [n for n in ("oa", "ob", "oab") if rows[n]["criteria"]["adopt"]]
    best = min(adopt, key=lambda n: rows[n]["lat"][4]) if adopt else None
    out[b] = {"rows": rows, "adopted": adopt, "pick": best}

(D / "o1_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

f2 = lambda x: f"{x:.2f}"
for b in ("b16", "b32"):
    R = out[b]["rows"]
    print(f"\n### {b.upper()}\n")
    print("| 구성 | 크기 MiB | t1 / t2 / t4 ms | 기준 대비 단축 t1 / t2 / t4 | FP32 대비 배속 t1 / t2 / t4 | peak 추론 MiB | 고화질 top-1 | FP32 대비 Δ [95%] | 기준 대비 Δ [95%] | CIFAR A (기준 대비) | cos (기준 대비) | 판정 ①②③ |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in NAMES:
        r = R[n]
        lat_s = " / ".join(f2(r["lat"][t]) for t in THREADS)
        red = " / ".join(f"{-r['reduction_pct'][t]:+.1f}%" for t in THREADS) if "reduction_pct" in r else ""
        spd = " / ".join(f"{r['speedup_vs_fp'][t]:.2f}×" for t in THREADS) if "speedup_vs_fp" in r else ""
        vf = f"{r['vs_fp']['d']:+.2f} [{r['vs_fp']['ci'][0]:+.2f}, {r['vs_fp']['ci'][1]:+.2f}]" if "vs_fp" in r else "기준"
        vb = f"{r['vs_base']['d']:+.2f} [{r['vs_base']['ci'][0]:+.2f}, {r['vs_base']['ci'][1]:+.2f}]" if "vs_base" in r else ""
        ca = f"{r['cifar_A']:.1f}" + (f" ({r['vs_base']['cifar']:+.1f})" if "vs_base" in r else "")
        cs = f"{r['vs_base']['cos']:.4f}" if "vs_base" in r else ""
        cr = r.get("criteria")
        crs = (" ".join("O" if cr[k] else "X" for k in ("1_speed_all_ge3pct", "2_hr_tie_or_better", "3_cifar_ge_-1"))
               + (" → **채택 후보**" if cr["adopt"] else "")) if cr else ""
        m = r["mem"]["infer"] if r["mem"] else float("nan")
        print(f"| {r['label']} | {r['MiB']:.1f} | {lat_s} | {red} | {spd} | {m:.1f} | {r['hr_top1']:.2f} | {vf} | {vb} | {ca} | {cs} | {crs} |")
    print(f"\n채택 후보: {out[b]['adopted']}, 고른 것(4스레드 최단): {out[b]['pick']}")
