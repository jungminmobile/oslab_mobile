"""G8 게이트: (1) S-const 연산 개수·제외 목록, (2) 가중치 MatMul 53개의 가중치 INT8·scale·zp와 활성값 scale·zp가 S-all과 비트 단위로 같은지.
사용 (siglip2/): RN50_RESULTS=../results/siglip2/G8_constb/gate "$SIGLIP2_PYTHON" scripts/g8_gate_check.py S_ALL.onnx S_CONST.onnx"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import s2common as S          # noqa: E402
from scripts import env_info  # noqa: E402


def qdq_info(path):
    """원래 MatMul 이름 → {입력/출력 텐서별 (scale, zp, int 값)}. QDQ 그래프에서 MatMul 입력은 DQ 출력이다."""
    g = onnx.load(str(path)).graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    out = {}
    for n in g.node:
        if n.op_type != "MatMul":
            continue
        rec = {}
        for k, t in (("A", n.input[0]), ("B", n.input[1])):
            dq = prod.get(t)
            if dq is None or dq.op_type != "DequantizeLinear":
                rec[k] = None
                continue
            x = dq.input[0]
            if x in init:                                   # 가중치: INT8 값이 initializer
                rec[k] = ("w", init[x], init[dq.input[1]], init[dq.input[2]] if len(dq.input) > 2 else None)
            else:                                            # 활성값: Q 노드의 scale·zp
                q = prod.get(x)
                rec[k] = ("a", None, init[q.input[1]], init[q.input[2]] if len(q.input) > 2 else None)
        qs = [c for c in cons.get(n.output[0], []) if c.op_type == "QuantizeLinear"]
        rec["Y"] = ("a", None, init[qs[0].input[1]], init[qs[0].input[2]] if len(qs[0].input) > 2 else None) if qs else None
        out[n.name] = rec
    return out


def ops(path):
    g = onnx.load(str(path), load_external_data=False).graph
    c = {}
    for n in g.node:
        c[n.op_type] = c.get(n.op_type, 0) + 1
    return c


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return all((x is None and y is None) or (x is not None and y is not None and x.dtype == y.dtype and x.shape == y.shape
               and x.tobytes() == y.tobytes()) for x, y in zip(a[1:], b[1:])) and a[0] == b[0]


def main():
    d = S.RC.ONNX_DIR
    sa, sc = d / sys.argv[1], d / sys.argv[2]
    ia, ic = qdq_info(sa), qdq_info(sc)
    assert set(ia) == set(ic), "MatMul 이름 집합이 다름"
    w_mm = [m for m in ic if ic[m]["B"] is not None and ic[m]["B"][0] == "w"]
    float_mm = [m for m in ic if ic[m]["A"] is None and ic[m]["B"] is None]
    diff = {m: [k for k in ("A", "B", "Y") if not same(ia[m][k], ic[m][k])] for m in w_mm}
    diff = {m: v for m, v in diff.items() if v}
    sa_int = [m for m in ia if ia[m]["A"] is not None]
    row = {"date": datetime.now().isoformat(timespec="seconds"), **{k: v for k, v in env_info.collect()["git"].items() if k in ("commit", "dirty")},
           **env_info.flat(env_info.collect()), "s_all": sa.name, "s_const": sc.name,
           "s_all_quantized_matmul": len(sa_int), "s_const_weight_matmul_int8": len(w_mm), "s_const_float_matmul": sorted(float_mm),
           "n_float_matmul": len(float_mm), "bit_identical_weight_matmul": not diff, "diff": diff,
           "saved_graph_ops": {"s_all": ops(sa), "s_const": ops(sc)}}
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "gate.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"S-all 양자화 MatMul {len(sa_int)}, S-const 가중치 INT8 MatMul {len(w_mm)}, float MatMul {len(float_mm)}")
    print("가중치 MatMul 53개 비트 동일:", not diff, diff if diff else "")
    for m in sorted(float_mm):
        print("  float:", m)


if __name__ == "__main__":
    main()
