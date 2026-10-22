"""e12_3_summary.py — E12-3 집계·판정 (계획서 4절 집계 정의, 5절 결정 규칙)

분모 = 장당 model_run 평균(워밍업 제외 100장). 묶음 ms = 묶음 커널 시간 합 / 100.
오버헤드 = 프로파일 켠 실행 perf_counter p50 − 무프로파일 perf_counter p50 (음수면 0으로 처리, 표시).
커널 밖(보정) = 커널 밖 평균 − 오버헤드. 세션별로 계산 후 3세션 중앙값.
"""
import argparse
import json
import statistics as st
from pathlib import Path

BUCKETS = {
    "matmul": ("QLinearMatMul", "MatMulInteger", "DynamicQuantizeMatMul", "MatMulIntegerToFloat", "QGemm", "QLinearConv",
               "MatMul", "FusedMatMul", "Gemm"),
    "qdq": ("QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear"),
    "addmulsub": ("Add", "Mul", "Sub", "Div"),
    "softmax": ("Softmax",),
    "gelu": ("Gelu", "FastGelu", "BiasGelu", "Erf", "QuickGelu", "Tanh", "Pow"),   # SigLIP2 GELU(tanh)는 Mul·Add·Tanh·Pow로 분해돼 있음: Tanh·Pow만 여기, Mul·Add는 addmulsub
    "ln": ("LayerNormalization", "SkipLayerNormalization", "SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization"),
    "shape": ("Transpose", "Reshape", "Concat", "Slice", "Gather", "Squeeze", "Unsqueeze", "Shape", "Expand", "Split",
              "Flatten", "Identity", "Cast"),
}
INT_MM = ("QLinearMatMul", "MatMulIntegerToFloat", "MatMulInteger", "DynamicQuantizeMatMul", "QGemm")
INT_KERNEL_OF = {"Add": "QLinearAdd", "Mul": "QLinearMul", "Sub": "QLinearSub", "Gelu": "QLinearGelu",
                 "LayerNormalization": "QLinearLayerNormalization", "SkipLayerNormalization": "QLinearLayerNormalization"}
ELEMWISE = ("Add", "Mul", "Sub", "Gelu", "FastGelu", "BiasGelu", "LayerNormalization", "SkipLayerNormalization")


def bucket(op):
    for b, ops in BUCKETS.items():
        if op in ops:
            return b
    return "other"


def metrics(row):
    p, npf = row["prof"], row.get("noprof")
    mr = p["model_run_mean_ms"]
    bms = {}
    for op, v in p["by_op_ms"].items():
        bms[bucket(op)] = bms.get(bucket(op), 0) + v
    calls = {}
    for op, v in p["node_calls_per_img"].items():
        calls[bucket(op)] = calls.get(bucket(op), 0) + v
    ov_raw = p["wall_p50_ms"] - npf["wall_p50_ms"] if npf else None
    ov = max(0.0, ov_raw) if ov_raw is not None else 0.0
    out_c = p["outside_mean_ms"] - ov
    return {"model_run_ms": mr, "model_run_p50_ms": p["model_run_p50_ms"], "kernel_ms": p["kernel_mean_ms"],
            "outside_ms": p["outside_mean_ms"], "overhead_raw_ms": ov_raw, "overhead_ms": ov, "outside_corr_ms": out_c,
            "outside_pct": 100 * p["outside_mean_ms"] / mr, "outside_corr_pct": 100 * out_c / mr,
            "overhead_pct_of_mr": 100 * ov / mr,
            "outside_corr_pct_altden": 100 * out_c / (mr - ov) if mr > ov else None,
            "wall_prof_p50": p["wall_p50_ms"], "wall_noprof_p50": npf["wall_p50_ms"] if npf else None,
            "bucket_ms": bms, "bucket_pct": {b: 100 * v / mr for b, v in bms.items()}, "bucket_calls": calls,
            "op_ms": p["by_op_ms"], "op_pct": {o: 100 * v / mr for o, v in p["by_op_ms"].items()}, "top_nodes": p["top_nodes"]}


def med_dict(ds):
    keys = set().union(*[d.keys() for d in ds])
    return {k: st.median([d.get(k, 0.0) for d in ds]) for k in keys}


def combine(ms):
    """세션별 metrics 리스트 → 중앙값."""
    out = {}
    for k in ("model_run_ms", "model_run_p50_ms", "kernel_ms", "outside_ms", "overhead_ms", "outside_corr_ms", "outside_pct",
              "outside_corr_pct", "overhead_pct_of_mr", "wall_prof_p50", "wall_noprof_p50"):
        vals = [m[k] for m in ms if m[k] is not None]
        out[k] = st.median(vals) if vals else None
    out["overhead_raw_ms"] = [m["overhead_raw_ms"] for m in ms]
    for k in ("bucket_ms", "bucket_pct", "bucket_calls", "op_ms", "op_pct"):
        out[k] = med_dict([m[k] for m in ms])
    mid = sorted(ms, key=lambda m: m["model_run_ms"])[len(ms) // 2]
    out["top_nodes"] = mid["top_nodes"]
    out["n_sessions"] = len(ms)
    return out


def eff(one, four):
    return {b: (one["bucket_ms"][b] / (four["bucket_ms"][b] * 4)) if four["bucket_ms"].get(b) else None
            for b in four["bucket_ms"] if b in one["bucket_ms"]}


def rule1(m4, e):
    low = [b for b, v in e.items() if v is not None and v < 0.5]
    low_pct = sum(m4["bucket_pct"][b] for b in low)
    return {"outside_corr_pct": m4["outside_corr_pct"], "a_outside_ge15": m4["outside_corr_pct"] >= 15,
            "low_eff_buckets": low, "low_eff_pct": low_pct, "b_loweff_ge20": low_pct >= 20,
            "holds": m4["outside_corr_pct"] >= 15 or low_pct >= 20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--affinity", required=True)
    ap.add_argument("--env", required=True, help="check_env.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    aff = [json.loads(l) for l in Path(a.affinity).read_text(encoding="utf-8").splitlines() if l.strip()]
    env = json.loads(Path(a.env).read_text(encoding="utf-8"))
    avail = {k: v["available"] for k, v in env["int_kernels"].items()}
    rows = [r for r in rows if r["session"] in ("1", "2", "3")]

    agg = {}
    for model in ("S", "F"):
        for th in (1, 2, 4):
            ms = [metrics(r) for r in rows if r["model"] == model and r["threads"] == th]
            agg[f"{model}_{th}T"] = combine(ms)
    per_session = {f"{r['session']}_{r['model']}_{r['threads']}T": {k: v for k, v in metrics(r).items() if k not in ("op_ms", "op_pct", "top_nodes")}
                   for r in rows}
    eS = eff(agg["S_1T"], agg["S_4T"])
    eF = eff(agg["F_1T"], agg["F_4T"])
    S4 = agg["S_4T"]

    # 규칙 1 (고정 없음, 3세션 중앙값)
    r1 = rule1(S4, eS)
    # 2b 코어 고정
    A = {r["session"]: metrics(r) for r in aff}
    pin = {}
    if "affinity_p4" in A and "affinity_p1" in A:
        ep = eff(A["affinity_p1"], A["affinity_p4"])
        pin = {"none_4T": {k: A["affinity_none"][k] for k in ("model_run_ms", "model_run_p50_ms", "outside_corr_pct", "wall_noprof_p50")} if "affinity_none" in A else None,
               "p4": {k: A["affinity_p4"][k] for k in ("model_run_ms", "model_run_p50_ms", "outside_corr_pct", "wall_noprof_p50")},
               "p1_model_run_ms": A["affinity_p1"]["model_run_ms"], "eff_pinned": ep, "rule1_pinned": rule1(A["affinity_p4"], ep),
               "bucket_pct_p4": A["affinity_p4"]["bucket_pct"]}
        rp = pin["rule1_pinned"]
        pin["near_threshold"] = abs(rp["outside_corr_pct"] - 15) <= 2 or abs(rp["low_eff_pct"] - 20) <= 2
    r1_final = r1["holds"] and bool(pin) and pin["rule1_pinned"]["holds"]

    # 규칙 2: Q/DQ + 정수 커널 있는 원소별 연산
    integ = [o for o in ELEMWISE if avail.get(INT_KERNEL_OF.get(o, "-"), False)]
    excluded = [o for o in ELEMWISE if o not in integ]
    r2_pct = S4["bucket_pct"].get("qdq", 0) + sum(S4["op_pct"].get(o, 0) for o in integ)
    # 규칙 3
    r3_pct = sum(S4["op_pct"].get(o, 0) for o in INT_MM)

    if r1_final:
        decision = "E12-4"
    elif r2_pct >= 20:
        decision = "E12-5 검토"
    elif r3_pct >= 70:
        decision = "보류"
    elif r1["holds"] and pin and not pin["rule1_pinned"]["holds"]:
        decision = "측정 방법 개선 (조건 1이 P코어 고정에서 사라짐)"
    else:
        decision = "보류 (규칙 1·2 불성립, 규칙 3 미달: 사전 등록 규칙에 해당 칸 없음 → 정민 판단)"

    res = {"agg": agg, "per_session": per_session, "eff_S": eS, "eff_F": eF, "rule1": r1, "pinned": pin, "rule1_final": r1_final,
           "rule2": {"integerizable_ops": integ, "excluded_ops": excluded, "pct": r2_pct, "holds": r2_pct >= 20},
           "rule3": {"int_matmul_pct": r3_pct, "holds": r3_pct >= 70}, "decision": decision}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    F4 = agg["F_4T"]
    print("| 묶음 | S 1T ms | S 4T % | S 4T ms | S eff | F 4T ms | F 4T % | S-F ms | S 호출수 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for b in list(BUCKETS) + ["other"]:
        s1, s4, f4 = agg["S_1T"]["bucket_ms"].get(b, 0), S4["bucket_ms"].get(b, 0), F4["bucket_ms"].get(b, 0)
        e = eS.get(b)
        print(f"| {b} | {s1:.2f} | {S4['bucket_pct'].get(b, 0):.1f} | {s4:.2f} | {e:.2f} | {f4:.2f} | {F4['bucket_pct'].get(b, 0):.1f} | {s4 - f4:+.2f} | {S4['bucket_calls'].get(b, 0):.0f} |"
              if e is not None else f"| {b} | {s1:.2f} | {S4['bucket_pct'].get(b, 0):.1f} | {s4:.2f} | - | {f4:.2f} | {F4['bucket_pct'].get(b, 0):.1f} | {s4 - f4:+.2f} | {S4['bucket_calls'].get(b, 0):.0f} |")
    for k in ("S_1T", "S_2T", "S_4T", "F_1T", "F_2T", "F_4T"):
        g = agg[k]
        print(f"{k}: model_run {g['model_run_ms']:.2f} kernel {g['kernel_ms']:.2f} outside {g['outside_ms']:.2f} ({g['outside_pct']:.1f}%) "
              f"overhead {g['overhead_ms']:.2f} raw {[round(x, 2) for x in g['overhead_raw_ms']]} outside_corr {g['outside_corr_pct']:.1f}% "
              f"wall prof {g['wall_prof_p50']:.2f} noprof {g['wall_noprof_p50']:.2f}")
    print("rule1", json.dumps(r1, ensure_ascii=False))
    print("pinned", json.dumps(pin, ensure_ascii=False, default=float)[:1500])
    print("rule2", json.dumps(res["rule2"], ensure_ascii=False), "rule3", json.dumps(res["rule3"]))
    print("DECISION", decision)
    print("S 4T op %:", {o: round(v, 2) for o, v in sorted(S4["op_pct"].items(), key=lambda kv: -kv[1])})


if __name__ == "__main__":
    main()
