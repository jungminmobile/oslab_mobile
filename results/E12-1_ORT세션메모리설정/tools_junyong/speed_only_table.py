"""speed_only_table.py — 메모리 없이(WSL) 속도만 정리. e12/analyze_e12_1.py의 speed_ratio·p50_median과 같은 공식.
속도 비율 = 같은 세션 안 (조합 p50 / C0 p50), 3세션 중앙값. 사용: python speed_only_table.py <results 폴더>
"""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(sys.argv[1])
spd = [json.loads(l) for l in open(R / "speed.jsonl", encoding="utf-8")]
eq = json.loads((R / "equal.json").read_text(encoding="utf-8")) if (R / "equal.json").exists() else {}
COMBOS = ["C0", "C1", "C2", "C3", "C4"]


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


print("| 모델 | 설정 | 1T p50 (ms) | 1T Δ | 2T p50 (ms) | 2T Δ | 4T p50 (ms) | 4T Δ | 4T Δ 세션별 |")
print("|---|---|---|---|---|---|---|---|---|")
out = {}
for model in sorted({r["model"] for r in spd}):
    for c in COMBOS:
        row, cells = {}, []
        for th in (1, 2, 4):
            ratio, per = speed_ratio(model, c, th)
            p = p50_median(model, c, th)
            row[f"p50_{th}T"], row[f"ratio_{th}T"], row[f"ratio_{th}T_sessions"] = p, ratio, per
            cells += [f"{p:.2f}", "—" if c == "C0" else f"{100 * (ratio - 1):+.1f}%"]
        ses = "" if c == "C0" else " / ".join(f"{100 * (x - 1):+.1f}" for x in row["ratio_4T_sessions"])
        print(f"| {model} | {c} | " + " | ".join(cells) + f" | {ses} |")
        out[f"{model}/{c}"] = row
print("\nG1_pass:", eq.get("G1_pass"))
(R / "speed_only_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
