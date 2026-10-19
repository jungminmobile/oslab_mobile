"""g5_profiler_extract.py — Hailo DFC 프로파일러 html에서 층별 컨텍스트·비트 수와 컨텍스트별 실측 시간을 뽑는다 (G5, 10/19, 읽기 전용)
=========================================================================================================================
html 안의 JSON 덩어리(작은따옴표 문자열로 박힌 것)를 json.raw_decode로 꺼낸다.
  - csv_data: 층마다 한 행 (행 구분자 ';'), 열 layer_name·layer_type·context·input/output_activation_bits·weights_bits 등
  - runs (런타임 데이터가 있는 html만, 예: Zoo의 *_compiled_runtime_data.html): 컨텍스트마다 context_time·overhead·configurations (클럭 틱)
[사용]  python3 -I g5_profiler_extract.py <profiler.html> <출력 접두어>
  → <접두어>_layers.csv, <접두어>_contexts.csv, <접두어>_summary.json
"""
import csv
import io
import json
import re
import sys
from collections import OrderedDict, defaultdict

KEEP = ["layer_name", "layer_type", "context", "input_activation_bits", "output_activation_bits", "weights_bits", "macs", "fps", "latency"]


def blobs(s):
    dec, pos, out = json.JSONDecoder(), 0, []
    for m in re.finditer(r'\{"', s):
        if m.start() < pos:
            continue
        try:
            obj, end = dec.raw_decode(s, m.start())
        except ValueError:
            continue
        if end - m.start() > 10000 and isinstance(obj, dict):
            out.append(obj)
            pos = end
    return out


def ctx_index(name):
    m = re.search(r"context_(\d+)$", name or "")
    return int(m.group(1)) if m else -1


def main(html, prefix):
    s = open(html, encoding="utf-8", errors="replace").read()
    bs = blobs(s)
    main_b = next(b for b in bs if "csv_data" in b)
    rows = [r for r in main_b["csv_data"].split(";") if r.strip()]
    i_layer = next(i for i, r in enumerate(rows) if r.startswith("layer_name,"))
    model_hdr = next(csv.reader(io.StringIO(rows[0])))
    model_row = next(csv.reader(io.StringIO(rows[1])))
    model = dict(zip(model_hdr, model_row))
    hdr = next(csv.reader(io.StringIO(rows[i_layer])))
    layers = []
    for r in rows[i_layer + 1:]:
        vals = next(csv.reader(io.StringIO(r)))
        if len(vals) != len(hdr):
            continue
        d = dict(zip(hdr, vals))
        if not d.get("context", "").count("context_"):
            continue
        layers.append({k: d.get(k, "") for k in KEEP})
    with open(prefix + "_layers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=KEEP)
        w.writeheader()
        w.writerows(layers)

    # 컨텍스트별 층 묶음
    per = defaultdict(list)
    for d in layers:
        per[ctx_index(d["context"])].append(d)
    # 런타임 데이터 (있으면)
    run_b = next((b for b in bs if "runs" in b and "clock_MHz" in b), None)
    rt = {}
    if run_b:
        mhz = float(run_b["clock_MHz"])
        run = run_b["runs"][0]
        for c in run["contexts"]:
            k = ctx_index(c.get("label", ""))
            if not c.get("context_time") or k in rt:   # 마지막 항목은 다음 프레임의 context_0 (context_time 없음) → 건너뜀
                continue
            t = lambda x: (x["end"] - x["start"]) / mhz / 1000.0 if x else None  # 틱 → ms
            rt[k] = {"context_ms": t(c.get("context_time")), "overhead_ms": t(c.get("overhead")),
                     "config_ms": t(c.get("configurations"))}
        rt["_batch_size"] = run.get("batch_size")
        rt["_runtime_percentages"] = run.get("runtime_percentages")
        rt["_preliminary_ms"] = (run["preliminary"]["context_time"]["end"] - run["preliminary"]["context_time"]["start"]) / mhz / 1000.0
    ctx_rows = []
    for k in sorted(per):
        ls = per[k]
        b16 = [d["layer_name"] for d in ls if d["output_activation_bits"] == "16" or d["weights_bits"] == "16"]
        ew = [d["layer_name"] for d in ls if re.fullmatch(r"ew_add\d+", d["layer_name"])]
        mm = [d["layer_name"] for d in ls if d["layer_type"] in ("conv", "dense", "matmul")]
        row = OrderedDict(context=k, n_layers=len(ls), ew_add=" ".join(ew), n_conv_mm=len(mm),
                          n_16bit=len(b16), layers_16bit=" ".join(b16))
        if k in rt:
            row.update({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in rt[k].items()})
        ctx_rows.append(row)
    fields = list(OrderedDict.fromkeys(f for r in ctx_rows for f in r))
    with open(prefix + "_contexts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(ctx_rows)
    summ = {"html": html, "model_name": model.get("model_name"), "hw_arch": model.get("hw_arch"),
            "number_of_contexts": model.get("number_of_contexts"), "layers": len(layers), "profiler_fps": model.get("fps"),
            "profiler_latency": model.get("latency"), "optimization_level": model.get("optimization_level"),
            "normalization": model.get("normalization"), "has_runtime_data": bool(run_b),
            "n_16bit_layers": sum(1 for d in layers if d["output_activation_bits"] == "16" or d["weights_bits"] == "16")}
    if run_b:
        cm = [rt[k]["context_ms"] for k in rt if isinstance(k, int)]
        om = [rt[k]["overhead_ms"] for k in rt if isinstance(k, int) and rt[k]["overhead_ms"] is not None]
        summ.update({"rt_batch_size": rt["_batch_size"], "rt_sum_context_ms": round(sum(cm), 3), "rt_n_contexts": len(cm),
                     "rt_sum_overhead_ms": round(sum(om), 3), "rt_preliminary_ms": round(rt["_preliminary_ms"], 3),
                     "rt_runtime_percentages": rt["_runtime_percentages"]})
        for b in bs:
            st = b.get("stats", {}).get("performance_details")
            if st:
                summ["stats_performance_details"] = st
    json.dump(summ, open(prefix + "_summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
