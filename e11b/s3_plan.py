"""s3_plan.py — E11b S3 2단계: 비용표 → 다중 예산 0/1 배낭(동적계획) → 구성 목록 (E10과 같은 예산 비율)
- 가치 s_u(cos 증가), 무게 t_u(4T 지연 증가, 0.05 ms 단위, 음수는 0), 예산 = R0 4T(비용표 측정의 중앙값) × (r − 1)
- 출력: s3_plan.json  {"configs": [{"tag": "S3P1", "r":…, "units":[…], "exclude": "...", "pred_dt":…, "pred_ds":…}]}  (같은 구성은 한 번)
사용: python s3_plan.py C:\\s2art\\E11b\\s3_cost.jsonl C:\\s2art\\E11b\\s3_plan.json"""
import itertools
import json
import statistics
import sys

import numpy as np

R = [1.013, 1.052, 1.091, 1.130, 1.169, 1.247, 1.325]
BASE_EX = "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/"
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
assert len(rows) == 51, len(rows)
t0 = statistics.median(r["lat_R0_ms"] for r in rows)
w = [max(0, int(round(r["t_u"] / 0.05))) for r in rows]
v = [r["s_u"] for r in rows]


def knap(cap):
    n = len(w)
    best = np.full(cap + 1, 0.0)
    keep = np.zeros((n, cap + 1), bool)
    for i in range(n):
        if v[i] <= 0:
            continue                                 # cos를 낮추는 단위는 고르지 않는다
        for c in range(cap, w[i] - 1, -1):
            if best[c - w[i]] + v[i] > best[c]:
                best[c] = best[c - w[i]] + v[i]
                keep[i, c] = True
    sel, c = [], cap
    for i in range(n - 1, -1, -1):
        if keep[i, c]:
            sel.append(i)
            c -= w[i]
    return sorted(sel)


# 작은 문제로 동적계획 검증: 앞 14개 단위 전수 탐색
for cap in (20, 60):
    sub = list(range(14))
    bv = max((sum(v[i] for i in s), s) for k in range(15) for s in itertools.combinations(sub, k)
             if sum(w[i] for i in s) <= cap and all(v[i] > 0 for i in s))[0]
    vs, ws = v, w
    globals()["v"], globals()["w"] = v[:14], w[:14]
    got = sum(v[i] for i in knap(cap))
    globals()["v"], globals()["w"] = vs, ws
    assert abs(got - bv) < 1e-12, (cap, got, bv)

configs, seen = [], set()
for k, r in enumerate(R, 1):
    cap = int((r - 1) * t0 / 0.05)
    sel = knap(cap)
    key = tuple(sel)
    if not sel or key in seen:
        continue
    seen.add(key)
    configs.append({"tag": f"S3P{k}", "r": r, "budget_ms": round((r - 1) * t0, 2), "units": [rows[i]["unit"] for i in sel],
                    "exclude": BASE_EX + "," + ",".join(rows[i]["pattern"] for i in sel),
                    "pred_dt_ms": round(sum(rows[i]["t_u"] for i in sel), 2), "pred_ds": round(sum(v[i] for i in sel), 4),
                    "pred_dMiB": round(sum(rows[i]["m_u_MiB"] for i in sel), 1)})
json.dump({"t0_ms": t0, "R": R, "configs": configs}, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({"t0_ms": t0, "configs": configs}, ensure_ascii=False, indent=1))
