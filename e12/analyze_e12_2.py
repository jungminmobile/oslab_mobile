"""analyze_e12_2.py — E12-2 8단계: 결과 파일 → 11절 표 + 사전 등록 판정 (계획서 5절)

판정 (모두 만족 시 채택):
 ① G0~G3 통과 ② 4T 지연 B1/B0 비율(같은 세션)의 3세션 중앙값 ≤ −3% ③ 1·2T 비율 중앙값 +2% 이내
 ④ 고해상도 dev top-1 B1−B0 95% CI 하한 > −0.3%p ⑤ peak 메모리 +2% 이내
 ②만 미달 → 효과 없음, ④ 또는 ⑤ 미달 → 기각
"""
import json
import statistics as st
import sys
from pathlib import Path

R = Path(sys.argv[1])
rd = lambda f: json.loads((R / f).read_text(encoding="utf-8"))
rl = lambda f: [json.loads(l) for l in (R / f).read_text(encoding="utf-8").splitlines() if l.strip()]

g0, diag, fuse, eq = rd("g0_sha.json"), rd("diag_ln.json"), rd("fuse_log.json"), rd("fp32_equal.json")
ql = {("B1" if "E12-2_B1" in r["model"] else "B0"): r for r in rl("quantize_log.jsonl")}
ci = rl("paired_ci.jsonl")
mem = [r for r in rl("memory.jsonl") if r["type"] == "summary"]
sp = [r for r in rl("speed.jsonl") if r["session"] in (1, 2, 3)]

tag = lambda m: "B1" if "E12-2_B1" in m else "B0" if "E12-2_B0" in m else "O"
acc = {tag(r["model"]): r for r in ci if "model" in r}
pair = next(r for r in ci if "pair" in r and tag(r["pair"][0]) == "B0" and tag(r["pair"][1]) == "B1")
pair_o = next(r for r in ci if "pair" in r and tag(r["pair"][0]) == "B0" and tag(r["pair"][1]) == "O")
peak = {r["model"]: r["median"]["peak_MiB"] for r in mem}

lat = {}
for r in sp:
    lat.setdefault((r["session"], r["model"], r["threads"]), r["p50_ms"])
ratio = {}
for th in (1, 2, 4):
    for m in ("O", "B1"):
        ratio[(m, th)] = [100 * (lat[(s, m, th)] / lat[(s, "B0", th)] - 1) for s in (1, 2, 3)]
med = lambda v: round(st.median(v), 2)
lat_med = {(m, th): round(st.median([lat[(s, m, th)] for s in (1, 2, 3)]), 2) for m in ("O", "B0", "B1") for th in (1, 2, 4)}

G = {"G0": g0["G0_pass"], "G1": True, "G2": fuse["G2a_fused_ln_ge_12"] and eq["G2b_pass"],
     "G3": all(ql[k]["ort_optimized_graph"]["int_kernel_ops"].get("QLinearMatMul") == 79 for k in ("B0", "B1"))
     and abs(ql["B1"]["MiB"] / ql["B0"]["MiB"] - 1) <= 0.01}
c = {"1_gates": all(G.values()), "2_4T_le_-3pct": med(ratio[("B1", 4)]) <= -3.0,
     "3_1T2T_within_+2pct": med(ratio[("B1", 1)]) <= 2.0 and med(ratio[("B1", 2)]) <= 2.0,
     "4_ci_low_gt_-0.3": pair["ci95"][0] > -0.3,
     "5_peak_within_+2pct": 100 * (peak["B1"] / peak["B0"] - 1) <= 2.0}
if not (c["4_ci_low_gt_-0.3"] and c["5_peak_within_+2pct"]):
    verdict = "기각"
elif all(c.values()):
    verdict = "채택"
elif not c["2_4T_le_-3pct"] and c["1_gates"] and c["3_1T2T_within_+2pct"]:
    verdict = "효과 없음"
else:
    verdict = "판정 규칙 밖 (보고 필요)"

opt_sln = lambda k: ql[k]["ort_optimized_graph"]["ops"].get("com.microsoft.SkipLayerNormalization", 0)
rows = []
hdr = "| 모델 | 합쳐진 LN | QLinearMatMul | 크기 (MiB) | 고해상도 top-1 | Δ vs B0 [95% CI] | CIFAR | cos | 4T 지연 (ms) | Δ vs B0 | 1T / 2T Δ | peak (MiB) |"
rows += [hdr, "|" + "---|" * 12]
hr = lambda k: round(st.mean([acc[k]["top1"][d] for d in ("pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev")]), 2)
for k, name in (("O", "O (공식)"), ("B0", "B0 (노트북 재양자화)"), ("B1", "B1 (합치기)")):
    sln = diag["optimized"]["n_SkipLayerNormalization"] if k == "O" else opt_sln(k)
    mib = 91.72 if k == "O" else ql[k]["MiB"]
    d = "—" if k == "B0" else (f"{pair['hr_diff_b_minus_a']:+.2f} [{pair['ci95'][0]}, {pair['ci95'][1]}]" if k == "B1"
                               else f"{pair_o['hr_diff_b_minus_a']:+.2f} [{pair_o['ci95'][0]}, {pair_o['ci95'][1]}]")
    dl = "—" if k == "B0" else f"{med(ratio[(k, 4)]):+.2f}%"
    d12 = "—" if k == "B0" else f"{med(ratio[(k, 1)]):+.2f}% / {med(ratio[(k, 2)]):+.2f}%"
    pk = peak.get(k, "(미측정)")
    rows.append(f"| {name} | {sln} | 79 | {mib} | {hr(k)} | {d} | {acc[k]['top1']['cifar_A']} | {acc[k]['hr_cos']} | "
                f"{lat_med[(k, 4)]} | {dl} | {d12} | {pk} |")
res = {"gates": G, "criteria": c, "verdict": verdict,
       "ratio_vs_B0_pct_per_session": {f"{m}_{th}T": [round(v, 2) for v in ratio[(m, th)]] for (m, th) in ratio},
       "ratio_vs_B0_pct_median": {f"{m}_{th}T": med(ratio[(m, th)]) for (m, th) in ratio},
       "p50_ms_median": {f"{m}_{th}T": v for (m, th), v in lat_med.items()},
       "peak_MiB": peak, "peak_delta_pct": round(100 * (peak["B1"] / peak["B0"] - 1), 2),
       "acc_pair_B1_minus_B0": {"hr": pair["hr_diff_b_minus_a"], "ci95": pair["ci95"]},
       "acc_pair_O_minus_B0": {"hr": pair_o["hr_diff_b_minus_a"], "ci95": pair_o["ci95"]},
       "table_md": rows}
(R / "verdict.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("\n".join(rows))
print(json.dumps({k: res[k] for k in ("gates", "criteria", "verdict", "ratio_vs_B0_pct_per_session", "peak_delta_pct")}, ensure_ascii=False, indent=1))
