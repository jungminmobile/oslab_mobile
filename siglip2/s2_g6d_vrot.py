"""
s2_g6d_vrot.py — G6d Vrot: Zoo 공개 B32 ONNX의 initializer 값만 바꿔 우리 회전(rothad)을 적용 (그래프 노드는 그대로)
==========================================================================================================
s2_rotate.rotate_trunk와 같은 변환 (같은 회전 행렬 rotq_rothad.npy, Q·1 = 1)을 HF/Zoo ONNX의 가중치 이름에 맞춰 옮겼다.
행 벡터 표기(ONNX MatMul은 X @ W, W: [입력, 출력])에서 잔차 h → h @ P, P = Qᵀ (s2_rotate의 열 벡터 Q·x와 같다).
  쓰는 쪽 (잔차에 더하는 것, 출력 쪽에 P):
    패치 conv weight[768,3,32,32]·bias, 위치 임베딩 상수[1,64,768], 블록마다 out_proj·fc2 MatMul W(→ W @ P)·bias(→ b @ P)
  읽는 쪽 (LN → Linear, LN γ·β를 접고 입력 쪽에 Pᵀ):  W ← Pᵀ (γ ⊙ W), b ← b + β @ W,  LN γ ← 1, β ← 0
    layer_norm1 → q·k·v 합친 MatMul [768,2304] (+ Split 뒤 q/k/v bias Add 3개)
    layer_norm2 → fc1 MatMul [768,3072] (+ bias)
    post_layernorm → (Transpose) → 헤드 k·v MatMul [768,1536] (+ bias)
  LN₀(h @ P) = LN₀(h) @ P (P·1 = 1이라 평균·노름 보존) → 출력(1393)은 수학적으로 같다.
  그래프 두 번째 출력 `key`(post_layernorm 출력)는 회전된 값이 된다 (parser 끝 노드 1393 밖, Zoo yaml 기준).
검사: 노드 수·연산 종류·노드 순서·initializer 이름이 원본과 같은지, ORT FP32 출력 1393의 cos (README 2절, ≥ 0.9999)
사용 (siglip2/, RN50_RESULTS=../results/siglip2/G6d): python s2_g6d_vrot.py --zoo <zoo.sim.onnx> --q <rotq_rothad.npy> --out <vrot.onnx>
"""
import argparse
import collections
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-npy", default="", help="FP32 동치 확인용 입력 N×3×256×256 (Zoo 전처리 정규화 값)")
    a = ap.parse_args()
    m = onnx.load(a.zoo)
    g = m.graph
    init = {t.name: t for t in g.initializer}
    arr = {k: numpy_helper.to_array(v).astype(np.float64) for k, v in init.items()}
    cons = collections.defaultdict(list)
    for i, n in enumerate(g.node):
        for x in n.input:
            cons[x].append(i)
    node = {n.name: n for n in g.node if n.name}
    Q = np.load(a.q).astype(np.float64)
    assert Q.shape == (768, 768) and np.abs(Q @ Q.T - np.eye(768)).max() < 1e-10 and np.abs(Q @ np.ones(768) - 1).max() < 1e-10
    P = Q.T
    used = set()                     # 값을 바꾼 initializer (각각 정확히 한 번만 바꾼다)

    def W(name):
        assert name in arr, name
        return arr[name]

    def setw(name, v):
        assert name not in used, f"{name}을 두 번 바꾸려 함"
        assert v.shape == arr[name].shape, (name, v.shape, arr[name].shape)
        used.add(name)
        arr[name] = v

    def w_of(n):
        ws = [x for x in n.input if x in arr and arr[x].ndim == 2]
        assert len(ws) == 1, (n.name, n.input)
        return ws[0]

    def b_of(add):
        bs = [x for x in add.input if x in arr and arr[x].ndim == 1]
        assert len(bs) == 1, (add.name, add.input)
        return bs[0]

    def next_ops(n, op):
        return [g.node[i] for i in cons[n.output[0]] if g.node[i].op_type == op]

    # ---- 쓰는 쪽
    conv = node["/model/embeddings/patch_embedding/Conv"]
    cw, cb = conv.input[1], conv.input[2]
    setw(cw, np.tensordot(P.T, W(cw), axes=([1], [0])))          # 출력 채널 o' = Σ_o P[o,o'] W[o]
    setw(cb, W(cb) @ P)
    pe = "/model/embeddings/position_embedding/Gather_output_0"
    setw(pe, W(pe) @ P)
    L = 12
    for l in range(L):
        for nm in (f"/model/encoder/layers.{l}/self_attn/out_proj", f"/model/encoder/layers.{l}/mlp/fc2"):
            mm, add = node[nm + "/MatMul"], node[nm + "/Add"]
            w, b = w_of(mm), b_of(add)
            setw(w, W(w) @ P)
            setw(b, W(b) @ P)

    # ---- 읽는 쪽
    def fold(ln_name, mm, adds_cols):
        ln = node[ln_name]
        gam, bet = ln.input[1], ln.input[2]
        w = w_of(mm)
        Wf = W(w)
        for add, (c0, c1) in adds_cols:
            b = b_of(add)
            setw(b, W(b) + W(bet) @ Wf[:, c0:c1])
        setw(w, P.T @ (W(gam)[:, None] * Wf))
        setw(gam, np.ones(768))
        setw(bet, np.zeros(768))

    for l in range(L):
        ln1 = node[f"/model/encoder/layers.{l}/layer_norm1/LayerNormalization"]
        mm = next_ops(ln1, "MatMul")
        assert len(mm) == 1
        sp = next_ops(mm[0], "Split")
        assert len(sp) == 1 and list(numpy_helper.to_array(init[sp[0].input[1]])) == [768, 768, 768]
        adds = [g.node[cons[o][0]] for o in sp[0].output]
        assert [x.name.split("/")[-2] for x in adds] == ["q_proj", "k_proj", "v_proj"], [x.name for x in adds]
        fold(ln1.name, mm[0], list(zip(adds, [(0, 768), (768, 1536), (1536, 2304)])))
        ln2 = node[f"/model/encoder/layers.{l}/layer_norm2/LayerNormalization"]
        fc1 = node[f"/model/encoder/layers.{l}/mlp/fc1/MatMul"]
        assert next_ops(ln2, "MatMul") == [fc1]
        fold(ln2.name, fc1, [(node[f"/model/encoder/layers.{l}/mlp/fc1/Add"], (0, 3072))])
    pl = node["/model/post_layernorm/LayerNormalization"]
    tr = node["/model/head/attention/Transpose_1"]
    perm = [a_.ints for a_ in tr.attribute if a_.name == "perm"][0]
    assert list(perm)[-1] == 2, perm                              # 마지막 축(채널)은 그대로
    assert [g.node[i].name for i in cons[pl.output[0]]] == ["/model/head/attention/Transpose_1"], [g.node[i].name for i in cons[pl.output[0]]]
    hmm, hadd = node["/model/head/attention/MatMul_1"], node["/model/head/attention/Add_1"]
    assert arr[w_of(hmm)].shape == (768, 1536)
    fold(pl.name, hmm, [(hadd, (0, 1536))])
    # 바꾼 수: conv 2 + 위치 1 + 블록마다 (out_proj·fc2 W·b 4 + LN1 γβ·qkv W·b3 6 + LN2 γβ·fc1 W·b 4) × 12 + post LN γβ·kv W·b 4
    assert len(used) == 3 + 14 * 12 + 4, len(used)
    # ---- 저장 (원래 dtype)
    for t in g.initializer:
        if t.name in used:
            t.CopyFrom(numpy_helper.from_array(arr[t.name].astype(numpy_helper.to_array(t).dtype), t.name))
    onnx.save(m, a.out)
    o, r = onnx.load(a.zoo, load_external_data=False), onnx.load(a.out, load_external_data=False)
    same_graph = ([(n.op_type, n.name, list(n.input), list(n.output)) for n in o.graph.node] ==
                  [(n.op_type, n.name, list(n.input), list(n.output)) for n in r.graph.node])
    same_inits = [t.name for t in o.graph.initializer] == [t.name for t in r.graph.initializer]
    ops = collections.Counter(n.op_type for n in r.graph.node)
    row = {"date": datetime.now().isoformat(timespec="seconds"), "what": "vrot", "zoo": a.zoo, "zoo_sha256": sha(a.zoo), "q_sha256": sha(a.q),
           "out": a.out, "out_sha256": sha(a.out), "n_nodes": len(r.graph.node), "ops": dict(sorted(ops.items())),
           "same_nodes_order_io": same_graph, "same_initializer_names": same_inits, "n_modified_initializers": len(used)}
    if a.check_npy:
        import onnxruntime as ort
        x = np.load(a.check_npy).astype(np.float32)
        def run(p):
            so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; so.intra_op_num_threads = 6
            s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]); n = s.get_inputs()[0].name
            return np.concatenate([s.run(["1393"], {n: x[i:i + 1]})[0] for i in range(len(x))])
        e0, e1 = run(a.zoo), run(a.out)
        c = (e0 / np.linalg.norm(e0, axis=1, keepdims=True) * (e1 / np.linalg.norm(e1, axis=1, keepdims=True))).sum(1)
        row.update(check_n=len(x), cos_min=float(c.min()), cos_mean=float(c.mean()), maxabs=float(np.abs(e0 - e1).max()))
    print(json.dumps(row, ensure_ascii=False))
    import s2common as S
    out = S.require_results_dir(); out.mkdir(parents=True, exist_ok=True)
    env = __import__("scripts.env_info", fromlist=["collect"]).collect()
    row.update(commit=env["git"]["commit"], dirty=env["git"]["dirty"])
    with open(out / "vrot.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    assert same_graph and same_inits, "그래프가 원본과 다름"
    if a.check_npy:
        assert row["cos_min"] >= 0.9999, f"FP32 동치 실패 cos_min {row['cos_min']}"


if __name__ == "__main__":
    main()
