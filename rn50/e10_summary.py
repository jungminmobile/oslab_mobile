"""E10 4단계 정리: 3단계 속도(3세션 중앙값)·메모리·정확도 → 가법성 표·판정표 (PREREG 2·3절). 출력 summary.json + 표 출력."""
import json
from statistics import median

import common as C

D = C.RESULTS_DIR
plan = json.loads((D / "ilp_plan.json").read_text(encoding="utf-8"))
cost = [json.loads(l) for l in (D / "cost_rn50.jsonl").read_text(encoding="utf-8").splitlines()]
tu = {r["unit"]: r["t_u"] for r in cost[1:]}
b0 = plan["b0_4T_ms"]
cfgs = list(plan["configs"])
names = ["fp32", "B2"] + cfgs
stem = lambda k: "rn50_image_int8_q1_percentile99999_highres256_s0_" + ("B2" if k == "B2" else f"E10_ilp_{k}")

# 속도: 3단계 실행만 (모델에 fp32와 B2가 함께 있는 것), 스레드별 시간순 = 세션 1·2·3
lat = {t: {k: [] for k in names} for t in (1, 2, 4)}
for f in sorted(D.glob("speed_*T.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    res = d["results"]
    if not {"fp32", "B2"} <= set(res):
        continue
    t = d["cond"]["threads"]
    for k in names:
        lat[t][k].append(res[k]["ms_per_img_mean"])
med = {t: {k: median(v) for k, v in lat[t].items()} for t in lat}
assert all(len(v) == 3 for t in lat for v in lat[t].values()), {t: {k: len(v) for k, v in lat[t].items()} for t in lat}

mem = {}
for l in (D / "memory.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(l)
    mem[r["name"]] = r
boot = {}
for l in (D / "bootstrap.jsonl").read_text(encoding="utf-8").splitlines():
    r = json.loads(l)
    boot[(r["a"], r["b"])] = r
size = {k: (C.onnx_path(stem(k) + ".onnx").stat().st_size if k != "fp32" else C.onnx_path("rn50_image_fp32_q1.onnx").stat().st_size) / 2**20 for k in names}

pred_b2 = b0 + sum(max(tu[u], 0) for u in ("stem.c1", "stem.c2", "stem.c3", "L1.0.full"))
rows = []
for k in cfgs:
    p = next(p for p in plan["plans"] if p["config"] == k)
    vs_b2 = boot[(stem("B2"), stem(k))]
    vs_fp = boot[("fp32", stem(k))]
    r = {"config": k, "fp32_units": plan["configs"][k], "MiB": round(size[k], 1),
         "lat_ms": {t: round(med[t][k], 2) for t in (1, 2, 4)}, "lat_sessions": {t: lat[t][k] for t in (1, 2, 4)},
         "peak_rss_MiB": mem.get(k, {}).get("peak_rss_MiB"), "acc_vs_fp32": vs_fp["diff_pp"], "acc_vs_fp32_ci": vs_fp["ci95"],
         "acc_vs_B2": vs_b2["diff_pp"], "acc_vs_B2_ci": vs_b2["ci95"],
         "pred_4T": p["pred_4T_ms"], "pred_minus_B2": round(p["pred_4T_ms"] - pred_b2, 2),
         "meas_minus_B2_4T": round(med[4][k] - med[4]["B2"], 2)}
    r["additivity_err"] = round(abs(r["pred_4T"] - med[4][k]) / med[4][k], 3)
    r["c1"] = vs_b2["ci95"][1] >= 0
    r["c2"] = all(med[t][k] <= med[t]["B2"] for t in (1, 2, 4))
    g4 = 1 - med[4][k] / med[4]["B2"]
    grss = (1 - mem[k]["peak_rss_MiB"] / mem["B2"]["peak_rss_MiB"]) if k in mem and "B2" in mem else None
    gsz = 1 - size[k] / size["B2"]
    r["gain_4T"], r["gain_rss"], r["gain_size"] = round(g4, 3), None if grss is None else round(grss, 3), round(gsz, 3)
    r["c3"] = g4 >= 0.05 or (grss is not None and grss >= 0.05) or gsz >= 0.10
    r["c4"] = all(med[t][k] < med[t]["fp32"] for t in (1, 2, 4))
    r["adopt"] = r["c1"] and r["c2"] and r["c3"] and r["c4"]
    rows.append(r)
out = {"b0_4T_cost_stage": b0, "pred_B2_4T": round(pred_b2, 2),
       "ref": {k: {"lat_ms": {t: round(med[t][k], 2) for t in (1, 2, 4)}, "MiB": round(size[k], 1),
                   "peak_rss_MiB": mem.get(k, {}).get("peak_rss_MiB")} for k in ("fp32", "B2")},
       "B2_vs_fp32": boot[("fp32", stem("B2"))]["diff_pp"], "rows": rows,
       "adopted": sorted([r for r in rows if r["adopt"]], key=lambda r: r["lat_ms"][4])[:1]}
(D / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out["ref"], ensure_ascii=False), "pred_B2_4T", out["pred_B2_4T"])
for r in rows:
    print(r["config"], r["MiB"], r["lat_ms"], "rss", r["peak_rss_MiB"], "vsB2", r["acc_vs_B2"], r["acc_vs_B2_ci"],
          "pred4T", r["pred_4T"], "adderr", r["additivity_err"], "c1-4", r["c1"], r["c2"], r["c3"], r["c4"], "gain4T", r["gain_4T"], "ADOPT" if r["adopt"] else "")
print("adopted:", [r["config"] for r in out["adopted"]])
