"""
ilp_alloc.py — E10 2단계: 비용표 → 다중선택 배낭 동적계획(지연 예산 아래 cos 회복 합 최대) + 전수 탐색 검증 (numpy만)
사전 등록: results/E10_ilp/PREREG.md (0e99334)

  그룹마다 {INT8(아무것도 안 함), 선택지들 중 하나를 FP32로} 고른다. X.0 그룹은 선택지 2개(ds/full), 나머지는 1개.
  가치 v = s_u (B0 대비 cos 증가분), 무게 w = round(max(t_u, 0) / 0.05 ms). 예산 W = floor((r−1) × B0_4T / 0.05).
사용: python ilp_alloc.py --cost ../results/E10_ilp/cost_rn50.jsonl --out ../results/E10_ilp/ilp_plan.json
"""
import argparse
import itertools
import json
from statistics import median

import numpy as np

R = [1.013, 1.052, 1.091, 1.130, 1.169, 1.247, 1.325]
STEP = 0.05


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    base, units = rows[0], rows[1:]
    groups = {}
    for u in units:
        groups.setdefault(u["group"], []).append(u)
    return base, groups


def dp(groups, W):
    g = list(groups)
    best = np.full(W + 1, -np.inf)
    best[0] = 0.0
    back = []
    for name in g:
        new, ch = best.copy(), np.full(W + 1, -1)
        for j, o in enumerate(groups[name]):
            w = int(round(max(o["t_u"], 0) / STEP))
            if w > W:
                continue
            cand = np.full(W + 1, -np.inf)
            cand[w:] = best[:W + 1 - w] + o["s_u"]
            m = cand > new
            new[m], ch[m] = cand[m], j
        back.append(ch)
        best = new
    c = int(np.argmax(best))
    val, pick = float(best[c]), []
    for name, ch in zip(reversed(g), reversed(back)):
        j = int(ch[c])
        if j >= 0:
            o = groups[name][j]
            pick.append(o["unit"])
            c -= int(round(max(o["t_u"], 0) / STEP))
    return val, sorted(pick)


def brute(groups, W):
    """전수 탐색: 선택지 1개 그룹은 비트 벡터로, 2개짜리(X.0) 그룹은 3^k 조합을 바깥 반복으로."""
    singles = [o[0] for o in groups.values() if len(o) == 1]
    multi = [o for o in groups.values() if len(o) > 1]
    n = len(singles)
    bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    v1 = bits @ np.array([o["s_u"] for o in singles])
    w1 = bits @ np.array([int(round(max(o["t_u"], 0) / STEP)) for o in singles])
    best = -np.inf
    for combo in itertools.product(*[[None] + m for m in multi]):
        cv = sum(o["s_u"] for o in combo if o)
        cw = sum(int(round(max(o["t_u"], 0) / STEP)) for o in combo if o)
        ok = w1 + cw <= W
        if ok.any():
            best = max(best, float(v1[ok].max()) + cv)
    return best, 2 ** n * 3 ** len(multi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-brute", action="store_true")
    a = ap.parse_args()
    base, groups = load(a.cost)
    b0 = median(base["b0_4T_runs_ms"])
    allu = {o["unit"]: o for opts in groups.values() for o in opts}
    plans, seen = [], {}
    for r in R:
        W = int((r - 1) * b0 / STEP)
        val, pick = dp(groups, W)
        bv, n = (None, None) if a.no_brute else brute(groups, W)
        key = ",".join(pick)
        cid = seen.setdefault(key, f"P{len(seen) + 1}")
        p = {"r": r, "budget_ms": round((r - 1) * b0, 3), "W": W, "config": cid, "fp32_units": pick,
             "pred_dt_ms": round(sum(max(allu[u]["t_u"], 0) for u in pick), 3),
             "pred_dt_ms_raw": round(sum(allu[u]["t_u"] for u in pick), 3),
             "pred_4T_ms": round(b0 + sum(max(allu[u]["t_u"], 0) for u in pick), 3),
             "pred_cos_gain": val, "pred_cos": base["cos"] + val,
             "pred_MiB": round(base["MiB"] + sum(allu[u]["m_u"] for u in pick), 2),
             "brute_best": bv, "brute_n": n, "brute_match": None if bv is None else abs(bv - val) < 1e-12}
        plans.append(p)
        print(f"r={r:.3f} budget {p['budget_ms']:.2f} ms -> {cid} pred 4T {p['pred_4T_ms']:.2f} cos {p['pred_cos']:.4f} "
              f"MiB {p['pred_MiB']:.1f} brute={p['brute_match']} FP32: {pick}", flush=True)
    out = {"b0_4T_ms": b0, "b0_cos": base["cos"], "b0_MiB": base["MiB"], "step_ms": STEP, "budgets_r": R, "plans": plans,
           "configs": {cid: k.split(",") if k else [] for k, cid in seen.items()}}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
