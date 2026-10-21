"""diag_ln.py — E12-2 1단계: LayerNorm이 왜 SkipLayerNormalization으로 안 합쳐지는지 진단 (계획서 4절 1단계)

입력 모델(기본: 공식 M2rot_ort)을 C0(ORT_ENABLE_ALL) 세션으로 열어 최적화 그래프를 저장하고,
남아 있는 LayerNormalization마다 바로 앞 노드(입력 생산자)와 그 노드의 출력 소비자 수, 입력 모양을 기록한다.
같은 진단을 원본(최적화 전) 그래프에도 한다 (--also-raw).
SkipLayerNorm 합치기 조건(ORT 코어/파이썬 optimizer 공통): LN 입력이 Add(a, b)이고 a·b 모양이 같음(3D).
코어 fusion은 Add 출력 소비자가 LN 하나뿐이어야 하고, 파이썬 optimizer는 잔차로 더 쓰이면 4번째 출력으로 처리한다.
"""
import argparse
import collections
import json
import sys
from datetime import datetime
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_opts as SO   # noqa: E402


def analyze(model_path, shape_infer=False):
    m = onnx.load(str(model_path), load_external_data=False)
    if shape_infer:
        try:
            from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
            m = SymbolicShapeInference.infer_shapes(onnx.load(str(model_path)), auto_merge=True)
        except Exception as e:   # noqa: BLE001
            print("shape infer 실패:", repr(e)[:300])
    g = m.graph
    shapes = {}
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        t = vi.type.tensor_type
        if t.HasField("shape"):
            shapes[vi.name] = [d.dim_param or d.dim_value for d in t.shape.dim]
    prod = {o: n for n in g.node for o in n.output}
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    inits = {i.name for i in g.initializer}
    ops = collections.Counter(f"{n.domain + '.' if n.domain else ''}{n.op_type}" for n in g.node)
    rows, pat = [], collections.Counter()
    for n in g.node:
        if n.op_type != "LayerNormalization":
            continue
        p = prod.get(n.input[0])
        r = {"ln": n.name, "in_shape": shapes.get(n.input[0]), "producer": p.op_type if p else "graph_input",
             "producer_name": p.name if p else None}
        if p is not None and p.op_type == "Add":
            a, b = p.input[0], p.input[1]
            r["add_inputs"] = [{"name": x, "init": x in inits, "shape": shapes.get(x),
                                "from": prod[x].op_type if x in prod else ("init" if x in inits else "graph_input")} for x in (a, b)]
            r["add_out_consumers"] = [c.op_type for c in cons[p.output[0]]]
            sa, sb = shapes.get(a), shapes.get(b)
            r["same_shape"] = (sa == sb) if (sa is not None and sb is not None) else None
            key = f"Add(from {r['add_inputs'][0]['from']},{r['add_inputs'][1]['from']}) consumers={len(r['add_out_consumers'])} same_shape={r['same_shape']}"
        else:
            key = f"producer={r['producer']}"
        r["pattern"] = key
        pat[key] += 1
        rows.append(r)
    return {"ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])), "n_LayerNormalization": ops.get("LayerNormalization", 0),
            "n_SkipLayerNormalization": ops.get("com.microsoft.SkipLayerNormalization", 0),
            "ln_patterns": dict(pat), "ln_nodes": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--also-raw", nargs="*", default=[], help="최적화 전 그래프도 분석할 모델 (shape inference 포함)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tmp", default=r"C:\s2art\tmp_e12_2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort
    opt = tmp / (Path(a.model).stem + "_C0opt.onnx")
    so = SO.make_options("C0", a.threads)
    so.optimized_model_filepath = str(opt)
    ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
    res = {"date": datetime.now().isoformat(timespec="seconds"), "model": Path(a.model).name, "combo": "C0",
           "ort": ort.__version__, "git": SO.git_state(), "optimized": analyze(opt)}
    res["raw"] = {Path(r).name: analyze(r, shape_infer=True) for r in a.also_raw}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("optimized LN", res["optimized"]["n_LayerNormalization"], "SLN", res["optimized"]["n_SkipLayerNormalization"])
    print(json.dumps(res["optimized"]["ln_patterns"], ensure_ascii=False, indent=1))
    for k, v in res["raw"].items():
        print("RAW", k, "LN", v["n_LayerNormalization"], json.dumps(v["ln_patterns"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
