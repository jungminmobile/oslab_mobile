"""rewrite_gelu.py — E14-2 GELU 융합 (계획서 results/E14-2_GELU융합/E14-2_GELU융합.md 3절)

O의 GELU(tanh 근사) 9노드
  x → Mul(x,x)=s → Mul(s,x)=cu → Mul(cu,0.044715)=ca → Add(x,ca)=in → Mul(in,√(2/π))=ik → Tanh=th
    → Add(th,1)=t1 → Mul(x,t1)=xt → Mul(xt,0.5)=y
를 찾아
  F1: y = FastGelu(x)                 (x = fc1/Add(bias) 출력, bias Add 그대로)
  F2: y = FastGelu(mm, bias)          (bias Add까지 흡수, x를 GELU 밖에서 안 쓸 때만)
로 바꾼다. 상수는 float32로 비교. Q/DQ와 다른 연산은 건드리지 않는다.
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper as nh

CONST = {"a": 0.044715, "k": float(np.sqrt(2 / np.pi)), "one": 1.0, "half": 0.5}


def _maps(g):
    init = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    return init, prod, cons


def _scalar(init, t):
    if t not in init:
        return None
    a = nh.to_array(init[t])
    return float(a.reshape(-1)[0]) if a.size == 1 else None


def _isc(init, t, key):
    v = _scalar(init, t)
    return v is not None and np.float32(v) == np.float32(CONST[key])


def _other(n, t):
    o = [i for i in n.input if i != t]
    return o[0] if len(o) == 1 else (t if list(n.input) == [t, t] else None)


def find_sites(model):
    g = model.graph
    init, prod, cons = _maps(g)
    outs = {o.name for o in g.output}
    sites = []
    for th in [n for n in g.node if n.op_type == "Tanh"]:
        r = {"tanh": th.name, "ok": False, "why": ""}
        try:
            ik = prod[th.input[0]]; assert ik.op_type == "Mul"
            kin = [i for i in ik.input if _isc(init, i, "k")]; assert len(kin) == 1, "√(2/π) 상수"
            add = prod[_other(ik, kin[0])]; assert add.op_type == "Add"
            # add = x + ca
            ca_n = [prod[i] for i in add.input if i in prod and prod[i].op_type == "Mul" and any(_isc(init, j, "a") for j in prod[i].input)]
            assert len(ca_n) == 1, "0.044715 Mul"
            ca = ca_n[0]
            x = _other(add, ca.output[0]); assert x is not None
            cu = prod[_other(ca, [j for j in ca.input if _isc(init, j, "a")][0])]; assert cu.op_type == "Mul"
            assert x in cu.input, "x³의 x"
            s = prod[_other(cu, x)]; assert s.op_type == "Mul" and list(s.input) == [x, x], "x·x"
            t1 = cons[th.output[0]]; assert len(t1) == 1 and t1[0].op_type == "Add"; t1 = t1[0]
            assert _isc(init, _other(t1, th.output[0]), "one"), "1+tanh"
            xt = cons[t1.output[0]]; assert len(xt) == 1 and xt[0].op_type == "Mul"; xt = xt[0]
            assert _other(xt, t1.output[0]) == x, "x·(1+tanh)"
            y = cons[xt.output[0]]; assert len(y) == 1 and y[0].op_type == "Mul"; y = y[0]
            assert _isc(init, _other(y, xt.output[0]), "half"), "·0.5"
            gelu = [s, cu, ca, add, ik, th, t1, xt, y]
            for n in gelu[:-1]:                       # 중간 출력은 GELU 안에서만 쓰여야 함
                assert n.output[0] not in outs
                assert all(c in gelu for c in cons.get(n.output[0], [])), f"{n.name} 출력이 밖에서 쓰임"
            x_users = cons[x]
            xb = prod.get(x)
            bias_ok = (xb is not None and xb.op_type == "Add" and sum(i in init for i in xb.input) == 1
                       and all(c in gelu for c in x_users) and x not in outs)
            r.update(ok=True, x=x, y=y.output[0], nodes=[n.name for n in gelu], y_node=y.name,
                     y_to=[c.op_type for c in cons.get(y.output[0], [])], x_from=xb.op_type if xb else "INPUT",
                     x_users=len(x_users), bias_ok=bias_ok)
            if bias_ok:
                b = [i for i in xb.input if i in init][0]
                mm = [i for i in xb.input if i not in init][0]
                r.update(bias=b, bias_shape=list(nh.to_array(init[b]).shape), mm=mm, mm_from=prod[mm].op_type if mm in prod else "INPUT",
                         bias_add=xb.name)
        except (AssertionError, KeyError, IndexError, TypeError) as e:
            r["why"] = str(e) or type(e).__name__
        sites.append(r)
    return sites


def rewrite(model, variant):
    m2 = onnx.ModelProto()
    m2.CopyFrom(model)
    g = m2.graph
    init, prod, cons = _maps(g)
    sites = [r for r in find_sites(model) if r["ok"] and (variant == "F1" or r["bias_ok"])]
    drop, repl, new_inits = set(), {}, []
    for i, r in enumerate(sites):
        tag = f"e14_{variant}_{i:02d}"
        drop |= set(r["nodes"])
        if variant == "F1":
            node = helper.make_node("FastGelu", [r["x"]], [r["y"]], name=f"{tag}_FastGelu", domain="com.microsoft")
        else:
            b = nh.to_array(init[r["bias"]]).astype(np.float32)
            bname = r["bias"]
            if b.ndim != 1:                            # FastGelu bias는 1차원 (마지막 축)
                bname = f"{tag}_bias1d"
                new_inits.append(nh.from_array(b.reshape(-1), bname))
            node = helper.make_node("FastGelu", [r["mm"], bname], [r["y"]], name=f"{tag}_FastGelu", domain="com.microsoft")
            drop.add(r["bias_add"])
        repl[r["y_node"]] = [node]
        r["new_node"] = node.name
    kept = []
    for n in g.node:
        if n.name in repl:
            kept.extend(repl[n.name])
        elif n.name not in drop:
            kept.append(n)
    del g.node[:]
    g.node.extend(kept)
    g.initializer.extend(new_inits)
    used = {t for n in g.node for t in n.input}
    removed = [x.name for x in g.initializer if x.name not in used]
    keep = [x for x in g.initializer if x.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    produced = {o for n in g.node for o in n.output}
    for vi in list(g.value_info):
        if vi.name not in produced:
            g.value_info.remove(vi)
    if not any(o.domain == "com.microsoft" for o in m2.opset_import):
        m2.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(m2)
    return m2, sites, {"new_init_bytes": int(sum(nh.to_array(x).nbytes for x in new_inits)), "removed_inits": removed}
