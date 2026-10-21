"""analyze_e12_1.py — E12-1 판정 계산 (계획서 6절 사전 등록 규칙 그대로) → judgment.json + 11절 표(markdown) 출력

채택: ① G1 통과 ② M2rot M-peak C0 대비 −5% 이상, 3회 모두 감소 ③ 4T 지연 +5% 이내 ④ 1·2T 지연 +10% 이내
부분 채택: ①② 만족, ③ 또는 ④ 불만족 / 효과 없음: M-peak 감소 5% 미만
여러 조합 채택 → M-peak 최소, 차이 2%p 이내면 더 빠른(4T) 조합. FP32는 판정에 쓰지 않음.
속도 비율 = 같은 세션 안 (조합 p50 / C0 p50), 3세션 중앙값.
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(sys.argv[1])          # results 폴더
eq = json.loads((R / "equal.json").read_text(encoding="utf-8"))
mem = [json.loads(l) for l in open(R / "memory.jsonl", encoding="utf-8")]
spd = [json.loads(l) for l in open(R / "speed.jsonl", encoding="utf-8")] if (R / "speed.jsonl").exists() else []
COMBOS = ["C0", "C1", "C2", "C3", "C4"]
summ = {(r["model"], r["combo"]): r for r in mem if r["type"] == "summary"}


def speed_ratio(model, combo, th):
    rs = []
    for s in sorted({r["session"] for r in spd}):
        d = {r["combo"]: r["p50_ms"] for r in spd if r["session"] == s and r["model"] == model and r["threads"] == th}
        if combo in d and "C0" in d:
            rs.append(d[combo] / d["C0"])
    return (float(np.median(rs)), rs) if rs else (None, [])


def p50_median(model, combo, th):
    v = [r["p50_ms"] for r in spd if r["model"] == model and r["combo"] == combo and r["threads"] == th]
    return float(np.median(v)) if v else None


res = {"G1_pass": eq["G1_pass"], "models": {}}
for model in sorted({m for m, _ in summ}):
    rows = {}
    for c in COMBOS:
        s = summ.get((model, c))
        if s is None:
            continue
        r = {"M_peak": s["median"]["peak_MiB"], "dPeak_pct": s["peak_delta_vs_C0_pct"], "all_lower": s["peak_lower_all_reps"],
             "M_create": s["median"]["create_peak_MiB"], "M_infer": s["median"]["infer_wset_max_MiB"],
             "M_private": s["median"]["infer_private_max_MiB"], "base": s["median"]["base_MiB"]}
        for th in (1, 2, 4):
            ratio, per = speed_ratio(model, c, th)
            r[f"p50_{th}T"] = p50_median(model, c, th)
            r[f"ratio_{th}T"] = None if ratio is None else round(ratio, 4)
            r[f"ratio_{th}T_sessions"] = [round(x, 4) for x in per]
        if c != "C0" and model == "m2rot":
            g2 = r["dPeak_pct"] is not None and r["dPeak_pct"] <= -5 and bool(r["all_lower"])
            g3 = r["ratio_4T"] is not None and r["ratio_4T"] <= 1.05
            g4 = all(r[f"ratio_{t}T"] is not None and r[f"ratio_{t}T"] <= 1.10 for t in (1, 2))
            if not (r["dPeak_pct"] is not None and r["dPeak_pct"] <= -5):
                v = "효과 없음"
            elif eq["G1_pass"] and g2 and g3 and g4:
                v = "채택"
            elif eq["G1_pass"] and g2:
                v = "부분 채택"
            else:
                v = "보류(①② 불만족: 정민 보고)"
            r.update({"cond2": g2, "cond3": g3, "cond4": g4, "verdict": v})
        rows[c] = r
    res["models"][model] = rows

m2 = res["models"].get("m2rot", {})
ad = [c for c, r in m2.items() if r.get("verdict") == "채택"]
if ad:
    best = min(ad, key=lambda c: m2[c]["M_peak"])
    near = [c for c in ad if m2[c]["M_peak"] - m2[best]["M_peak"] <= 0.02 * m2["C0"]["M_peak"]]
    res["selected"] = min(near, key=lambda c: m2[c]["ratio_4T"])
else:
    res["selected"] = None
(R / "judgment.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def f(v, d=1, pct=False):
    if v is None:
        return "—"
    return f"{v:+.{d}f}%" if pct else f"{v:.{d}f}"


print("| 모델 | 설정 | M-peak (MiB) | Δ vs C0 | 3회 모두↓ | M-create | M-infer | M-private | 4T p50 (ms) | 4T Δ | 1T Δ | 2T Δ | 판정 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for model, rows in res["models"].items():
    for c, r in rows.items():
        rt = {t: None if r[f"ratio_{t}T"] is None else 100 * (r[f"ratio_{t}T"] - 1) for t in (1, 2, 4)}
        print(f"| {model} | {c} | {f(r['M_peak'])} | {f(r['dPeak_pct'], 1, True)} | {r['all_lower'] if c != 'C0' else ''} | "
              f"{f(r['M_create'])} | {f(r['M_infer'])} | {f(r['M_private'])} | {f(r['p50_4T'], 2)} | {f(rt[4], 1, True)} | "
              f"{f(rt[1], 1, True)} | {f(rt[2], 1, True)} | {r.get('verdict', '')} |")
print("\nG1_pass:", res["G1_pass"], " selected:", res["selected"])
