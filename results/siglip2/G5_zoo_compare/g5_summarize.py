"""g5_summarize.py — G5 실칩 raw(out/)에서 results.jsonl과 요약표를 만든다 (데스크탑, 읽기 전용 입력)
[사용] python3 -I g5_summarize.py <out 폴더> <커밋 해시> <dirty 0/1>
  - 세션 시작 온도: temps.jsonl에서 'wait_bench_b{B}_s{S}' 마지막 행 (run.log 'cool …' 직전에 읽은 값)
  - 보고 값: 세션별 mean_ms의 3세션 중앙값 (README 2절), 세션별 median_ms의 3세션 중앙값도 같이
"""
import json
import statistics as st
import sys
from pathlib import Path

CTX = {"M2rot2_D256L1": 21, "rothad2_D256L1": 17, "siglip_b_16_image_encoder": 17}


def main(out, commit, dirty):
    out = Path(out)
    temps = [json.loads(l) for l in open(out / "temps.jsonl") if l.startswith("{")]
    last = {}
    for t in temps:
        last[t["label"]] = t
    rows = []
    for b in (1, 8):
        for s in (1, 2, 3):
            j = json.load(open(out / f"bench_b{b}_s{s}" / "bench.json"))
            t0 = last.get(f"wait_bench_b{b}_s{s}", {})
            for r in j["runs"]:
                rows.append({"exp": "G5", "date": j["date"], "commit": commit, "dirty": bool(int(dirty)), "hailortcli": j["hailortcli"],
                             "host": j["platform"], "batch": b, "session": s, "session_start_temp_c": max(t0.get("ts0_c", -1), t0.get("ts1_c", -1)),
                             "contexts": CTX[r["model"]], **{k: r[k] for k in ("model", "hef_sha256", "input_domain", "mean_ms", "median_ms", "p90_ms", "min_ms", "max_ms")},
                             "img_per_s_from_mean": round(1000 * b / r["mean_ms"], 2), "warmup": j["warmup"], "n": j["n"]})
    with open(out.parent / "results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {}
    for b in (1, 8):
        for m in CTX:
            rr = [r for r in rows if r["batch"] == b and r["model"] == m]
            summ[(b, m)] = {"mean_ms_sessions": [r["mean_ms"] for r in rr], "median_ms_sessions": [r["median_ms"] for r in rr],
                            "rep_mean_ms": st.median(r["mean_ms"] for r in rr), "rep_median_ms": st.median(r["median_ms"] for r in rr),
                            "p90_sessions": [r["p90_ms"] for r in rr], "temps": [r["session_start_temp_c"] for r in rr]}
    for (b, m), v in summ.items():
        ips = 1000 * b / v["rep_mean_ms"]
        print(f"b{b} {m:28s} C={CTX[m]:2d} mean/session {v['mean_ms_sessions']} → {v['rep_mean_ms']:.3f} ms ({ips:.2f} img/s) | "
              f"median/session {v['median_ms_sessions']} → {v['rep_median_ms']:.3f} | p90 {v['p90_sessions']} | T0 {v['temps']}")
    z = summ[(1, "siglip_b_16_image_encoder")]["rep_mean_ms"]
    for m in ("M2rot2_D256L1", "rothad2_D256L1"):
        d = summ[(1, m)]["rep_mean_ms"] - z
        pred = (CTX[m] - 17) * 2.0
        tol = max(1.0, 0.25 * abs(d))
        print(f"{m} − Zoo: 실측 Δ {d:+.3f} ms, 예측 Δ {pred:+.1f} ms, |차| {abs(d - pred):.3f} ≤ {tol:.3f}? {'설명됨' if abs(d - pred) <= tol else '설명 안 됨'}")
    m2, ro = summ[(1, "M2rot2_D256L1")]["rep_mean_ms"], summ[(1, "rothad2_D256L1")]["rep_mean_ms"]
    print(f"(기록) M2rot2 − rothad2: {m2 - ro:+.3f} ms, ΔC 4 → 컨텍스트당 {(m2 - ro) / 4:.3f} ms")


if __name__ == "__main__":
    main(*sys.argv[1:4])
