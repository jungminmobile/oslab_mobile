import onnx, numpy as np, sys
from onnx import numpy_helper as nh
O = r"C:\s2art\onnx\e12_3\siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx"
m = onnx.load(O)
g = m.graph
init = {i.name: i for i in g.initializer}
prod = {o: n for n in g.node for o in n.output}
cons = {}
for n in g.node:
    for i in n.input:
        cons.setdefault(i, []).append(n)
print("nodes", len(g.node), "inputs", [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in g.input], "outputs", [o.name for o in g.output])
from collections import Counter
print(Counter(n.op_type for n in g.node).most_common())
def desc(t):
    if t in init: a = nh.to_array(init[t]); return f"INIT{a.shape}{a.dtype}"
    p = prod.get(t); return f"{p.op_type}:{p.name}" if p else "INPUT"
for s in [n for n in g.node if n.op_type == "Sub"]:
    c = cons.get(s.output[0], [])
    print("SUB", s.name, [ (t, desc(t)) for t in s.input], "->", [(x.op_type, x.name) for x in c])
    for mul in c:
        print("   MUL", [(t, desc(t)) for t in mul.input], "->", [(x.op_type, x.name, [(t, desc(t)) for t in x.input]) for x in cons.get(mul.output[0], [])])
