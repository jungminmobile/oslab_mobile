"""rewrite_patch.py — E14-4 패치 임베딩 Conv(k32 s32) → 자르기 + MatMul (계획서 results/E14-4_패치임베딩MatMul/E14-4_패치임베딩MatMul.md 3절)

P1  : image → Reshape[-1,3,8,32,8,32] → Transpose(0,2,4,1,3,5) → Reshape[-1,64,3072] → MatMul(FP32 Wm) → /trunk/Add(pos + b)
P2  : image → Add(+1) → Q(U8, 2/255, 0) → DQ → 자르기(float) → MatMul ← DQ(INT8 Wq) → /trunk/Add(pos + b')
P2b : image → Add(+1) → Q → 자르기(uint8) → DQ → MatMul ← DQ(INT8 Wq) → /trunk/Add(pos + b')
Wm = W.reshape(768,3072).T,  b' = b − W_dq.sum(axis=0) (float64 → float32).  bias는 위치 임베딩 상수에 미리 더한다 (공통 규칙 1).
새 노드 이름은 모두 "/e14_4/"로 시작한다 (Q/DQ 지문에서 뺄 때 씀).
"""
import copy
import hashlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper as nh

PFX = "/e14_4/"
CONV = "/trunk/patch_embed/proj/Conv"
MM_OUT = PFX + "patch_mm"
T_OUT = "/trunk/patch_embed/Transpose_output_0"
ADD = "/trunk/Add"
SCALE_IN = np.float32(2.0 / 255.0)


def _maps(g):
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    return prod, cons


def inspect(model):
    """G0 ②③: Conv·뒤 연결·위치 임베딩 확인."""
    g = model.graph
    init = {i.name: i for i in g.initializer}
    prod, cons = _maps(g)
    convs = [n for n in g.node if n.op_type == "Conv"]
    r = {"n_conv": len(convs)}
    c = next(n for n in convs if n.name == CONV)
    at = {a.name: helper.get_attribute_value(a) for a in c.attribute}
    W, b = nh.to_array(init[c.input[1]]), nh.to_array(init[c.input[2]])
    r.update({"conv_input": c.input[0], "attrs": {k: (v.decode() if isinstance(v, bytes) else v) for k, v in at.items()},
              "W_shape": list(W.shape), "W_absmax": float(np.abs(W).max()), "b_shape": list(b.shape), "b_absmax": float(np.abs(b).max())})
    r["attrs_ok"] = (at.get("kernel_shape") == [32, 32] and at.get("strides") == [32, 32] and at.get("pads", [0] * 4) == [0, 0, 0, 0]
                     and at.get("group", 1) == 1 and at.get("dilations", [1, 1]) == [1, 1] and at.get("auto_pad", b"NOTSET") in (b"NOTSET", "NOTSET"))
    rs = cons[c.output[0]]
    tr = cons[rs[0].output[0]] if len(rs) == 1 else []
    ad = cons[tr[0].output[0]] if len(tr) == 1 else []
    r["chain"] = [x[0].op_type if len(x) == 1 else [n.op_type for n in x] for x in (rs, tr, ad)]
    r["reshape_shape"] = nh.to_array(init[rs[0].input[1]]).tolist()
    r["transpose_perm"] = helper.get_attribute_value(tr[0].attribute[0])
    pos_name = [i for i in ad[0].input if i != tr[0].output[0]][0]
    r["pos_name"], r["pos_is_init"] = pos_name, pos_name in init
    r["pos_n_consumers"] = len(cons.get(pos_name, []))
    if pos_name in init:
        r["pos_shape"] = list(nh.to_array(init[pos_name]).shape)
    r["add_name"] = ad[0].name
    r["graph_input_direct"] = c.input[0] in {i.name for i in g.input}
    r["ok"] = bool(r["n_conv"] == 1 and r["attrs_ok"] and r["chain"] == ["Reshape", "Transpose", "Add"] and list(W.shape) == [768, 3, 32, 32]
                   and r["transpose_perm"] == [0, 2, 1] and r["pos_is_init"] and r["pos_n_consumers"] == 1 and r["graph_input_direct"])
    return r


def weight_format(model):
    """G0 ④: 기존 MatMul 가중치 DQ 형식 (모든 initializer 입력 DQ 요약)."""
    g = model.graph
    init = {i.name: i for i in g.initializer}
    rows = []
    for n in g.node:
        if n.op_type == "DequantizeLinear" and n.input[0] in init:
            w = nh.to_array(init[n.input[0]])
            s = nh.to_array(init[n.input[1]])
            z = nh.to_array(init[n.input[2]]) if len(n.input) > 2 and n.input[2] else None
            ax = {a.name: helper.get_attribute_value(a) for a in n.attribute}.get("axis", 1)
            rows.append((str(w.dtype), int(w.min()), int(w.max()), list(s.shape) == [w.shape[1]], int(ax),
                         None if z is None else (str(z.dtype), bool((z == 0).all()))))
    from collections import Counter
    return {"n_weight_dq": len(rows), "kinds": {str(k): v for k, v in Counter((r[0], r[3], r[4], r[5]) for r in rows).items()},
            "int_min": min(r[1] for r in rows), "int_max": max(r[2] for r in rows)}


def quant_weight(Wm, qmax=64):
    """채널별(출력 축=1) |max| 대칭, int8, zp 0, 범위 ±qmax (기존 rr 형식). ORT quantizer와 같은 계산: scale = absmax/qmax."""
    absmax = np.abs(Wm).max(axis=0).astype(np.float32)
    scale = (absmax / np.float32(qmax)).astype(np.float32)
    q = np.clip(np.round(Wm / scale), -qmax, qmax).astype(np.int8)
    return q, scale, np.zeros(Wm.shape[1], np.int8)


def rewrite(model, variant):
    assert variant in ("P1", "P2", "P2b")
    m = copy.deepcopy(model)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    prod, cons = _maps(g)
    c = next(n for n in g.node if n.name == CONV)
    rs = cons[c.output[0]][0]
    tr = cons[rs.output[0]][0]
    ad = cons[tr.output[0]][0]
    assert ad.name == ADD
    pos_name = [i for i in ad.input if i != tr.output[0]][0]
    W = nh.to_array(init[c.input[1]]).astype(np.float32)
    b = nh.to_array(init[c.input[2]]).astype(np.float32)
    pos = nh.to_array(init[pos_name]).astype(np.float32)
    Wm = np.ascontiguousarray(W.reshape(768, 3072).T)
    new_nodes, new_inits = [], []
    I = lambda name, arr: new_inits.append(nh.from_array(np.asarray(arr), PFX + name)) or PFX + name
    sh6 = I("shape6", np.array([-1, 3, 8, 32, 8, 32], np.int64))
    sh3 = I("shape3", np.array([-1, 64, 3072], np.int64))
    N = lambda op, ins, outs, name, **kw: new_nodes.append(helper.make_node(op, ins, outs, name=PFX + name, **kw))

    def crop(src, tag):
        N("Reshape", [src, sh6], [PFX + f"r6{tag}"], f"Reshape6{tag}")
        N("Transpose", [PFX + f"r6{tag}"], [PFX + f"tp{tag}"], f"Transpose{tag}", perm=[0, 2, 4, 1, 3, 5])
        N("Reshape", [PFX + f"tp{tag}", sh3], [PFX + f"r3{tag}"], f"Reshape3{tag}")
        return PFX + f"r3{tag}"

    extra = {}
    if variant == "P1":
        a = crop(c.input[0], "")
        wname = I("Wm", Wm)
        N("MatMul", [a, wname], [MM_OUT], "MatMul")
        bias_used = b
    else:
        q, ws, wz = quant_weight(Wm)
        W_dq = q.astype(np.float32) * ws[None, :]
        bp = (b.astype(np.float64) - W_dq.astype(np.float64).sum(axis=0)).astype(np.float32)
        one = I("one", np.array(1.0, np.float32))
        s_in, z_in = I("in_scale", SCALE_IN), I("in_zp", np.array(0, np.uint8))
        N("Add", [c.input[0], one], [PFX + "x1"], "AddOne")
        N("QuantizeLinear", [PFX + "x1", s_in, z_in], [PFX + "xq"], "QuantizeLinear")
        if variant == "P2":
            N("DequantizeLinear", [PFX + "xq", s_in, z_in], [PFX + "xdq"], "DequantizeLinear")
            a = crop(PFX + "xdq", "")
        else:
            aq = crop(PFX + "xq", "")
            N("DequantizeLinear", [aq, s_in, z_in], [PFX + "adq"], "DequantizeLinear")
            a = PFX + "adq"
        wq, wsn, wzn = I("Wq", q), I("Wq_scale", ws), I("Wq_zp", wz)
        N("DequantizeLinear", [wq, wsn, wzn], [PFX + "Wdq"], "DequantizeLinear_W", axis=1)
        N("MatMul", [a, PFX + "Wdq"], [MM_OUT], "MatMul")
        bias_used = bp
        extra = {"w_int_min": int(q.min()), "w_int_max": int(q.max()), "w_rel_err_max": float(np.abs(W_dq - Wm).max() / np.abs(Wm).max()),
                 "bprime_absmax": float(np.abs(bp).max())}
    new_pos = (pos + bias_used[None, None, :]).astype(np.float32)
    pos_new_name = I("pos_embed_plus_bias", new_pos)
    ad.input[:] = [MM_OUT if i == tr.output[0] else (pos_new_name if i == pos_name else i) for i in ad.input]
    # 옛 노드·상수 제거, 새 노드는 Conv 자리(그래프 맨 앞쪽)에 넣어 위상 순서를 지킨다
    drop = {c.name, rs.name, tr.name}
    idx = [k for k, n in enumerate(g.node) if n.name == c.name][0]
    kept = [n for n in g.node if n.name not in drop]
    nodes = kept[:idx] + new_nodes + kept[idx:]
    del g.node[:]
    g.node.extend(nodes)
    still = {i for n in g.node for i in n.input}
    removed = [x.name for x in g.initializer if x.name not in still]
    keep_init = [x for x in g.initializer if x.name in still]
    del g.initializer[:]
    g.initializer.extend(keep_init + new_inits)
    vi = [v for v in g.value_info if v.name not in {c.output[0], rs.output[0], tr.output[0]}]
    del g.value_info[:]
    g.value_info.extend(vi)
    onnx.checker.check_model(m)
    extra.update({"removed_inits": removed, "new_init_bytes": int(sum(nh.to_array(x).nbytes for x in new_inits)),
                  "bias_used": bias_used, "new_nodes": [n.op_type for n in new_nodes]})
    return m, extra


def qdq_fingerprint_excl(model):
    """rewrite_mask.qdq_fingerprint와 같은 계산, 단 /e14_4/ 새 노드는 뺀다 (G1 ②: 패치 임베딩 밖 Q/DQ)."""
    g = model.graph
    init = {i.name: i for i in g.initializer}
    h, hw = hashlib.sha256(), hashlib.sha256()
    cnt = {"QuantizeLinear": 0, "DequantizeLinear": 0}
    for n in g.node:
        if n.op_type in cnt and not n.name.startswith(PFX):
            cnt[n.op_type] += 1
            for t in n.input[1:]:
                h.update(t.encode())
                if t in init:
                    a = nh.to_array(init[t])
                    h.update((str(a.dtype) + str(a.shape)).encode() + a.tobytes())
            if n.op_type == "DequantizeLinear" and n.input[0] in init:
                hw.update(nh.to_array(init[n.input[0]]).tobytes())
    return {"counts": cnt, "scale_zp_sha256": h.hexdigest(), "dq_weight_sha256": hw.hexdigest()}


def add_outputs(model, names):
    """측정용 사본: 중간 텐서를 그래프 출력으로 추가 (Q 출력은 zp dtype)."""
    m = copy.deepcopy(model)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    have = {o.name for o in g.output}
    for n in names:
        if n in prod and n not in have:
            et = TensorProto.FLOAT
            p = prod[n]
            if p.op_type == "QuantizeLinear" and len(p.input) > 2 and p.input[2] in init:
                et = init[p.input[2]].data_type
            g.output.append(helper.make_tensor_value_info(n, et, None))
    return m
