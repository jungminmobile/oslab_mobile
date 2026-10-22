"""rewrite_mask.py — E14-1 M2 마스크 식 정리 (계획서 results/E14-1_M2마스크정리/E14-1_M2마스크정리.md 3절)

O의 M2 마스크 `Y + m·(c − X)` (Sub(c, X) → Mul(m, ·) → Add(Y, ·)) 를 찾아
  W1: Y·(1−m) + (m·c)          (Mul·Add, 상수는 미리 계산)
  W2: Where(m != 0, c, Y)       (Where 1개)
로 바꾼다. 바꾸는 조건 (계획서 3절 규칙 1):
  - 단순: Y == X (같은 텐서 이름)
  - 체인: Y가 같은 X를 쓰는 앞 마스크 자리의 출력이고, 두 마스크의 1인 자리가 겹치지 않음 (상수로 확인)
    → m=1 자리에서 Y = X 가 정확히 성립하므로 단순과 같은 선택 연산이 된다. G2(c)로 2장 실행 확인.
  - 그 밖: 그대로 둔다.
Q/DQ 노드와 그 상수, 다른 연산은 건드리지 않는다. 더 이상 쓰이지 않는 상수만 지운다.
"""
import hashlib

import numpy as np
import onnx
from onnx import helper, numpy_helper as nh


def _maps(g):
    init = {i.name: i for i in g.initializer}
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    outs = {o.name for o in g.output}
    return init, prod, cons, outs


def find_sites(model):
    """마스크 자리 목록. 각 자리: sub·mul·add 이름, c·m 이름, X·Y·출력 이름, 종류(simple/chain/other)."""
    g = model.graph
    init, prod, cons, outs = _maps(g)
    sites = []
    for s in [n for n in g.node if n.op_type == "Sub"]:
        c, X = s.input
        r = {"sub": s.name, "c": c, "X": X, "kind": "other", "why": ""}
        cs = cons.get(s.output[0], [])
        if c not in init or len(cs) != 1 or cs[0].op_type != "Mul" or s.output[0] in outs:
            r["why"] = "Sub 모양이 다름"; sites.append(r); continue
        mul = cs[0]
        mname = [t for t in mul.input if t != s.output[0]]
        ca = cons.get(mul.output[0], [])
        if len(mname) != 1 or mname[0] not in init or len(ca) != 1 or ca[0].op_type != "Add" or mul.output[0] in outs:
            r["why"] = "Mul 모양이 다름"; sites.append(r); continue
        add = ca[0]
        Y = [t for t in add.input if t != mul.output[0]]
        if len(Y) != 1:
            r["why"] = "Add 모양이 다름"; sites.append(r); continue
        m = nh.to_array(init[mname[0]])
        r.update(mul=mul.name, add=add.name, m=mname[0], Y=Y[0], out=add.output[0],
                 m_shape=list(m.shape), c_shape=list(nh.to_array(init[c]).shape),
                 m_binary=bool(np.isin(m, [0.0, 1.0]).all()), m_ones=int(m.sum()),
                 m_pos=[int(i) for i in np.flatnonzero(m.reshape(-1))])
        sites.append(r)
    by_out = {r["out"]: r for r in sites if "out" in r}
    for r in sites:
        if "out" not in r:
            continue
        if not r["m_binary"]:
            r["why"] = "마스크가 0/1이 아님"; continue
        if r["Y"] == r["X"]:
            r["kind"] = "simple"; continue
        up = by_out.get(r["Y"])
        if up is None:
            r["why"] = f"Y({r['Y']})가 X도 아니고 앞 마스크 출력도 아님"; continue
        chain, cur, ok = [], up, True
        while cur is not None:          # 체인을 끝까지 따라가 모든 앞 마스크가 같은 X를 쓰는지
            chain.append(cur)
            if cur["X"] != r["X"]:
                ok = False; break
            cur = by_out.get(cur["Y"]) if cur["Y"] != cur["X"] else None
        if not ok:
            r["why"] = "앞 마스크가 다른 X를 씀"; continue
        m = nh.to_array(init[r["m"]])
        overlap = 0
        for u in chain:
            mu = nh.to_array(init[u["m"]])
            sh = np.broadcast_shapes(m.shape, mu.shape)
            overlap += int(((np.broadcast_to(m, sh) > 0) & (np.broadcast_to(mu, sh) > 0)).sum())
        r["chain_up"] = [u["sub"] for u in chain]
        r["overlap"] = overlap
        if overlap:
            r["why"] = "앞 마스크와 1인 자리가 겹침"; continue
        r["kind"] = "chain"
    return sites


def rewrite(model, variant):
    """variant W1/W2. 새 ModelProto와 바꾼 자리 목록을 돌려준다."""
    m2 = onnx.ModelProto()
    m2.CopyFrom(model)
    g = m2.graph
    init, prod, cons, outs = _maps(g)
    sites = [r for r in find_sites(model) if r["kind"] in ("simple", "chain")]
    drop, new_inits, repl = set(), [], {}
    for i, r in enumerate(sites):
        c = nh.to_array(init[r["c"]]).astype(np.float32)
        m = nh.to_array(init[r["m"]]).astype(np.float32)
        tag = f"e14_{variant}_{i:02d}"
        if variant == "W1":
            new_inits += [nh.from_array((1.0 - m).astype(np.float32), f"{tag}_inv"),
                          nh.from_array((m * c).astype(np.float32), f"{tag}_mc")]
            nodes = [helper.make_node("Mul", [r["Y"], f"{tag}_inv"], [f"{tag}_mul"], name=f"{tag}_Mul"),
                     helper.make_node("Add", [f"{tag}_mul", f"{tag}_mc"], [r["out"]], name=f"{tag}_Add")]
        else:
            new_inits += [nh.from_array(m != 0, f"{tag}_mask")]
            nodes = [helper.make_node("Where", [f"{tag}_mask", r["c"], r["Y"]], [r["out"]], name=f"{tag}_Where")]
        r["new_nodes"] = [n.name for n in nodes]
        repl[r["add"]] = nodes
        drop |= {r["sub"], r["mul"]}
    kept = []
    for n in g.node:
        if n.name in drop:
            continue
        kept.extend(repl.get(n.name, [n]))
    del g.node[:]
    g.node.extend(kept)
    g.initializer.extend(new_inits)
    used = {t for n in g.node for t in n.input}
    removed = [x.name for x in g.initializer if x.name not in used]
    keep_init = [x for x in g.initializer if x.name in used]
    del g.initializer[:]
    g.initializer.extend(keep_init)
    produced = {o for n in g.node for o in n.output}
    for vi in list(g.value_info):        # 지운 노드 출력의 shape 정보 제거
        if vi.name not in produced:
            g.value_info.remove(vi)
    onnx.checker.check_model(m2)
    new_bytes = int(sum(nh.to_array(x).nbytes for x in new_inits))
    return m2, sites, {"new_init_bytes": new_bytes, "removed_inits": removed}


def qdq_fingerprint(model):
    """Q/DQ 노드 수와 각 노드 scale·zero point 상수의 sha256 (노드 순서대로) + DQ 정수 가중치 sha256."""
    g = model.graph
    init = {i.name: i for i in g.initializer}
    h, hw = hashlib.sha256(), hashlib.sha256()
    cnt = {"QuantizeLinear": 0, "DequantizeLinear": 0}
    for n in g.node:
        if n.op_type in cnt:
            cnt[n.op_type] += 1
            for t in n.input[1:]:
                h.update(t.encode())
                if t in init:
                    a = nh.to_array(init[t])
                    h.update((str(a.dtype) + str(a.shape)).encode() + a.tobytes())
            if n.op_type == "DequantizeLinear" and n.input[0] in init:
                hw.update(nh.to_array(init[n.input[0]]).tobytes())
    return {"counts": cnt, "scale_zp_sha256": h.hexdigest(), "dq_weight_sha256": hw.hexdigest()}


def file_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
