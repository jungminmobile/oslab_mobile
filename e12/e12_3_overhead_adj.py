"""e12_3_overhead_adj.py — E12-3 사후 민감도 분석 (사전 등록 아님, 판정에 쓰지 않음)

관찰: 프로파일 오버헤드(4T 약 6.6 ms)가 커널 밖(약 1.6 ms)보다 커서, 오버헤드 대부분이 노드 kernel_time 안에 들어감.
→ 노드마다 같은 오버헤드가 붙는다고 가정하고 빼 본다:
   노드당 오버헤드 = (프로파일 perf_counter p50 − 무프로파일 p50) / 장당 노드 수
   보정 op ms = op ms − 호출 수 × 노드당 오버헤드 (0 아래면 0),  분모 = 무프로파일 p50
이 보정으로도 규칙 1(b)·2·3 결론이 같은지 본다. 세션별 계산 후 3세션 중앙값.
"""
import argparse
import json
import statistics as st
from pathlib import Path

import e12_3_summary as SM


def adj(row):
    p, n = row["prof"], row["noprof"]
    calls = p["node_calls_per_img"]
    total = sum(calls.values())
    ov = max(0.0, p["wall_p50_ms"] - n["wall_p50_ms"])
    per = ov / total
    den = n["wall_p50_ms"]
    op = {o: max(0.0, v - calls.get(o, 0) * per) for o, v in p["by_op_ms"].items()}
    b = {}
    for o, v in op.items():
        b[SM.bucket(o)] = b.get(SM.bucket(o), 0) + v
    return {"per_node_us": per * 1000, "den": den, "op_ms": op, "bucket_ms": b, "kernel_adj": sum(op.values()),
            "outside_est_ms": den - sum(op.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--affinity", required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in Path(a.inp).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["session"] in ("1", "2", "3")]
    aff = {r["session"]: r for r in (json.loads(l) for l in Path(a.affinity).read_text(encoding="utf-8").splitlines() if l.strip())}
    env = json.loads(Path(a.env).read_text(encoding="utf-8"))
    avail = {k: v["available"] for k, v in env["int_kernels"].items()}
    integ = [o for o in SM.ELEMWISE if avail.get(SM.INT_KERNEL_OF.get(o, "-"), False)]

    res = {}
    for model in ("S", "F"):
        for th in (1, 2, 4):
            xs = [adj(r) for r in rows if r["model"] == model and r["threads"] == th]
            res[f"{model}_{th}T"] = {"per_node_us": st.median(x["per_node_us"] for x in xs), "den": st.median(x["den"] for x in xs),
                                     "outside_est_ms": st.median(x["outside_est_ms"] for x in xs),
                                     "bucket_ms": SM.med_dict([x["bucket_ms"] for x in xs]), "op_ms": SM.med_dict([x["op_ms"] for x in xs])}

    def rules(r1, r4):
        e = {b: r1["bucket_ms"][b] / (r4["bucket_ms"][b] * 4) for b in r4["bucket_ms"] if r4["bucket_ms"][b] > 0 and b in r1["bucket_ms"]}
        pct = {b: 100 * v / r4["den"] for b, v in r4["bucket_ms"].items()}
        low = [b for b, v in e.items() if v < 0.5]
        r2 = pct.get("qdq", 0) + sum(100 * r4["op_ms"].get(o, 0) / r4["den"] for o in integ)
        r3 = sum(100 * r4["op_ms"].get(o, 0) / r4["den"] for o in SM.INT_MM)
        return {"eff": e, "pct": pct, "outside_est_pct": 100 * r4["outside_est_ms"] / r4["den"], "low_eff_buckets": low,
                "low_eff_pct": sum(pct[b] for b in low), "rule2_pct": r2, "rule3_pct": r3}

    out = {"note": "사후 민감도, 판정에 쓰지 않음", "agg": res, "S": rules(res["S_1T"], res["S_4T"]), "F": rules(res["F_1T"], res["F_4T"])}
    if "affinity_p4" in aff:
        p4, p1 = adj(aff["affinity_p4"]), None
        # 2b 고정 1T는 무프로파일 기준이 있으므로 같은 방식
        p1 = adj(aff["affinity_p1"])
        out["S_pinned"] = rules({"bucket_ms": p1["bucket_ms"]}, {**p4})
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for k in ("S", "F", "S_pinned"):
        if k in out:
            r = out[k]
            print(k, "eff", {b: round(v, 2) for b, v in r["eff"].items()}, "pct", {b: round(v, 1) for b, v in r["pct"].items()},
                  "outside_est%", round(r["outside_est_pct"], 1), "low", r["low_eff_buckets"], round(r["low_eff_pct"], 1),
                  "rule2%", round(r["rule2_pct"], 1), "rule3%", round(r["rule3_pct"], 1))
    for k, v in res.items():
        print(k, "per_node_us", round(v["per_node_us"], 2), "den", v["den"], "outside_est", round(v["outside_est_ms"], 2))


if __name__ == "__main__":
    main()
