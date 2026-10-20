"""
g8_metrics.py — G8 A4: 기존 raw(속도 json·model_stats)에서 보고서 양식 2·6장 숫자 계산 (모델 실행 없음)
  지연: 세션(speed_a/b/c)마다 장당 평균 → 3세션 중앙값, FPS = 1000 / 그 값 (배치 1), p50·p95는 세션별 값의 중앙값
  GFLOPs = 2 × GMACs (곱셈-누산 1번 = 연산 2번, fvcore 기준). 크기 MB(10^6 B)와 MiB를 함께
사용: python scripts/g8_metrics.py --speed <폴더> --stats <model_stats.jsonl> --out <json>
"""
import argparse
import glob
import json
import os
import statistics as st
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--speed", required=True, help="speed_a·speed_b·speed_c가 있는 폴더")
ap.add_argument("--stats", required=True)
ap.add_argument("--onnx-dir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = {}
for sess in sorted(glob.glob(os.path.join(a.speed, "speed_*/"))):
    for f in sorted(glob.glob(os.path.join(sess, "speed_*.json"))):
        d = json.load(open(f, encoding="utf-8"))
        t = d["cond"]["threads"]
        for m, r in d["results"].items():
            x = rows.setdefault(m, {}).setdefault(t, {"mean": [], "p50": [], "p95": [], "file": r["file"]})
            x["mean"].append(r["ms_per_img_mean"]); x["p50"].append(r["lat_p50"]); x["p95"].append(r["lat_p95"])
stats = [json.loads(l) for l in open(a.stats, encoding="utf-8") if '"params"' in l][-1]
out = {"latency": {}, "stats": {}}
for m, byt in rows.items():
    out["latency"][m] = {}
    for t, x in sorted(byt.items()):
        assert len(x["mean"]) == 3, (m, t, len(x["mean"]))
        med = st.median(x["mean"])
        out["latency"][m][t] = {"sessions_ms": x["mean"], "median_ms": round(med, 2), "fps_b1": round(1000 / med, 2),
                                "p50_ms": round(st.median(x["p50"]), 2), "p95_ms": round(st.median(x["p95"]), 2)}
        out["latency"][m][t]["file"] = x["file"]
for m, byt in out["latency"].items():
    f = Path(a.onnx_dir) / next(iter(byt.values()))["file"]
    out["stats"][m] = {"bytes": f.stat().st_size, "MB": round(f.stat().st_size / 1e6, 2), "MiB": round(f.stat().st_size / 2**20, 2)}
g = stats["gmacs"]["fvcore"]["total_GMACs"]
out["params_M"] = round(stats["params"]["onnx_initializers"] / 1e6, 2)
out["GMACs_fvcore"] = round(g, 3); out["GFLOPs"] = round(2 * g, 2)
out["peak_mem_MiB"] = {k: v["memory"]["peak_during_infer_MiB"] for k, v in stats["files"].items() if "memory" in v}   # 모델마다 새 프로세스, 추론 중 최대
out["peak_mem_MB"] = {k: round(v * 2**20 / 1e6, 1) for k, v in out["peak_mem_MiB"].items()}
json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)[:3000])
