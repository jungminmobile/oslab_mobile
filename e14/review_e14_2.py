"""review_e14_2.py — E14-2 계획 검토용 사전 확인 (측정 아님): GELU 13곳 패턴, ORT 최적화 후 fc1 bias 처리, CPU FastGelu 커널·수치."""
import numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh, helper, TensorProto
from collections import Counter
O = r"C:\s2art\onnx\e12_3\siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx"
OPT = r"C:\s2art_b32\e14\e14_1_O_ortopt.onnx"
m = onnx.load(O); g = m.graph
init = {i.name: nh.to_array(i) for i in g.initializer}
prod = {o: n for n in g.node for o in n.output}
cons = {}
for n in g.node:
    for i in n.input: cons.setdefault(i, []).append(n)
def val(t): return init[t].tolist() if t in init and init[t].size == 1 else (f"INIT{init[t].shape}" if t in init else prod[t].op_type if t in prod else "IN")
pats = Counter()
for t in [n for n in g.node if n.op_type == "Tanh"]:
    # 거꾸로: Tanh 입력 Mul(·c1) ← Add(x + ...) 
    m3 = prod[t.input[0]]; add = prod[[i for i in m3.input if i not in init][0]]
    x = [i for i in add.input if prod.get(i) is None or prod[i].op_type != "Mul" or i in init]
    xin = add.input
    xprod = [prod[i].op_type if i in prod else "INIT" for i in xin]
    c1 = [init[i].item() for i in m3.input if i in init]
    # x 찾기: Add 입력 중 cube Mul 경로가 아닌 쪽
    X = [i for i in xin if not (i in prod and prod[i].op_type == "Mul" and any(j in init for j in prod[i].input))][0]
    xp = prod[X]
    bias_add = xp.op_type == "Add" and any(i in init for i in xp.input)
    mm = prod[[i for i in xp.input if i not in init][0]] if bias_add else None
    users = Counter(c.op_type for c in cons[X])
    after = cons[t.output[0]]
    chain = []
    cur = t
    for _ in range(3):
        nx = cons[cur.output[0]]
        if len(nx) != 1: break
        cur = nx[0]; chain.append(cur.op_type + str([val(i) for i in cur.input if i in init]))
    end_cons = [c.op_type for c in cons[cur.output[0]]]
    key = (bias_add, mm.op_type if mm else None, tuple(sorted(users.items())), tuple(c1), tuple(chain), tuple(end_cons))
    pats[key] += 1
for k, v in pats.items(): print(v, "곳:", k)
# ORT 최적화 후 그래프: Tanh 앞뒤, MatMulIntegerToFloat bias 입력 여부
mo = onnx.load(OPT); go = mo.graph
pr = {o: n for n in go.node for o in n.output}
oi = {i.name for i in go.initializer}
print("opt ops:", Counter(n.op_type for n in go.node).most_common(12))
for t in [n for n in go.node if n.op_type == "Tanh"][:2]:
    # 최적화 그래프에서 GELU 입력 x를 거슬러 올라가기
    seen = []; cur = pr[t.input[0]]
    for _ in range(4):
        seen.append(cur.op_type)
        nxt = [i for i in cur.input if i in pr]
        if not nxt: break
        cur = pr[nxt[0]]
    print("Tanh 위쪽:", seen)
mitf = [n for n in go.node if n.op_type == "MatMulIntegerToFloat"]
print("MatMulIntegerToFloat 입력 개수 분포:", Counter(len(n.input) for n in mitf), " bias 입력(6번) 있는 것:", sum(len(n.input) > 6 and n.input[6] != "" for n in mitf))
# CPU FastGelu 커널 확인
x = (np.random.default_rng(0).standard_normal((1, 64, 3072)) * 3).astype(np.float32)
b = (np.random.default_rng(1).standard_normal(3072) * 0.5).astype(np.float32)
def run(nodes, inputs, inits):
    gr = helper.make_graph(nodes, "t", [helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in inputs],
                           [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)], [nh.from_array(v, k) for k, v in inits.items()])
    md = helper.make_model(gr, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)])
    md.ir_version = 9
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(md.SerializeToString(), so, providers=["CPUExecutionProvider"])
    return s.run(None, {"x": x})[0]
c = {"a": np.float32(0.044715), "k": np.float32(0.7978845608028654), "one": np.float32(1), "h": np.float32(0.5), "b": b}
dec = [helper.make_node("Add", ["x", "b"], ["xb"]), helper.make_node("Mul", ["xb", "xb"], ["s"]), helper.make_node("Mul", ["s", "xb"], ["cu"]),
       helper.make_node("Mul", ["cu", "a"], ["ca"]), helper.make_node("Add", ["xb", "ca"], ["in"]), helper.make_node("Mul", ["in", "k"], ["ik"]),
       helper.make_node("Tanh", ["ik"], ["th"]), helper.make_node("Add", ["th", "one"], ["t1"]), helper.make_node("Mul", ["xb", "t1"], ["xt"]),
       helper.make_node("Mul", ["xt", "h"], ["y"])]
y0 = run(dec, ["x"], c)
y1 = run([helper.make_node("Add", ["x", "b"], ["xb"]), helper.make_node("FastGelu", ["xb"], ["y"], domain="com.microsoft")], ["x"], {"b": b})
y2 = run([helper.make_node("FastGelu", ["x", "b"], ["y"], domain="com.microsoft")], ["x"], {"b": b})
for nm, y in (("F1 FastGelu(x)", y1), ("F2 FastGelu(x,b)", y2)):
    d = np.abs(y - y0)
    rel = d / np.maximum(np.abs(y0), 1e-30)
    print(nm, "max|Δ|", float(d.max()), "max|Δ|/max|y|", float(d.max() / np.abs(y0).max()), "max 상대오차(원소별)", float(rel.max()),
          "|y0|<1e-3인 원소에서 max 상대오차", float(rel[np.abs(y0) < 1e-3].max()), "비트 같은 비율", float((y == y0).mean()))
