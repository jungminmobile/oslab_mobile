"""a_compare.py — B32 본선 HEF 바탕 ONNX(배치 1)와 제출 ONNX 바탕(배치 가변)이 같은 모델인지 확인 (README A절, CPU만)
================================================================================================================
[2] initializer + Constant 노드 값을 대응시켜 비트 비교 (이름이 다르면 소비 노드 이름·입력 위치로 대응)
[3] 두 export 기록의 rotq_sha256와 회전 행렬 파일 sha256
[4] sink 마스크 상수(zm_*, m_*)의 값과 그 상수를 쓰는 노드(블록)를 그래프에서 읽기
[5] dev 64장(pets_dev 앞 64, Zoo 전처리 256 캐시) ORT FP32 배치 1 출력 cos·최대 절대 차이
[사용] python a_compare.py --b1 <b1.onnx> --dyn <dyn.onnx> --rotq <rotq_rothad.npy> --x <pets_dev_x.npy> --out a_result.json
"""
import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper as nh


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tensors(g):
    """{키: (배열, 이름, 소비자 목록)}. 키는 이름이 안정적이면 이름, onnx::* 같은 자동 이름이면 첫 소비 노드 이름·입력 위치."""
    cons = defaultdict(list)
    for n in g.node:
        for i, x in enumerate(n.input):
            cons[x].append((n.name, n.op_type, i))
    out = {}
    for t in g.initializer:
        k = t.name if not t.name.startswith("onnx::") else "@" + "|".join(f"{a}:{i}" for a, _, i in sorted(cons[t.name]))
        out[("init", k)] = (nh.to_array(t), t.name, cons[t.name])
    for n in g.node:
        if n.op_type == "Constant":
            a = [x for x in n.attribute if x.name == "value"]
            if not a:
                continue
            out[("const", n.name)] = (nh.to_array(a[0].t), n.output[0], cons[n.output[0]])
    return out


def category(kind, name, arr, consumers):
    base = name.split("/")[-1]
    if kind == "init":
        if re.match(r"trunk\.(kc|vc)_", name):
            return "M2_sink_KV"
        if re.match(r"trunk\.uc_", name):
            return "M2_uc"
        if re.match(r"trunk\.(zm|m)_b", name):
            return "M2_mask"
        if "norm" in name:
            return "LN"
        if any(op in ("MatMul", "Conv") for _, op, _ in consumers) and arr.ndim >= 2:
            return "rotated_weight(MatMul/Conv)"
        return "other_param(bias·pos_embed·latent)"
    # Constant 노드: 모양 계산용(정수, Shape/Reshape/Concat/Unsqueeze/Gather/Slice/Expand/ConstantOfShape 쪽)인지
    if arr.dtype.kind in "iub":
        return "const_int(shape·index)"
    return "const_float"


def cmp_tensors(g1, g2):
    t1, t2 = tensors(g1), tensors(g2)
    rows, summ = [], defaultdict(lambda: {"same": 0, "diff": 0})
    for k in sorted(set(t1) | set(t2)):
        if k not in t1 or k not in t2:
            only = "b1" if k in t1 else "dyn"
            a, nm, cons = (t1 if k in t1 else t2)[k]
            cat = category(k[0], nm, a, cons)
            rows.append({"key": k, "only_in": only, "cat": cat, "shape": list(a.shape), "dtype": str(a.dtype),
                         "value": a.tolist() if a.size <= 8 else None, "consumers": [c[0] for c in cons][:3]})
            summ[f"{cat}|only_{only}"]["same"] += 1
            continue
        a, nm1, cons1 = t1[k]
        b, nm2, cons2 = t2[k]
        cat = category(k[0], nm1, a, cons1)
        same = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        summ[cat]["same" if same else "diff"] += 1
        if not same:
            rows.append({"key": k, "cat": cat, "b1_name": nm1, "dyn_name": nm2, "b1": a.tolist() if a.size <= 16 else list(a.shape),
                         "dyn": b.tolist() if b.size <= 16 else list(b.shape), "consumers": [c[0] for c in cons1][:3]})
    return dict(summ), rows


def sink_rules(g):
    """마스크 상수 값과 쓰는 노드. zm_bX: u_c 대체 시작 블록 X의 토큰 행 마스크, m_bX_i: K/V 대체 블록 X의 i번째 sink 마스크."""
    cons = defaultdict(list)
    for n in g.node:
        for x in n.input:
            cons[x].append(n.name)
    out = {}
    for t in g.initializer:
        if re.match(r"trunk\.(zm|m)_b", t.name):
            a = nh.to_array(t).reshape(-1)
            out[t.name] = {"n": int(a.size), "unique": sorted(set(a.tolist())), "zero_rows": [int(i) for i in np.where(a == 0)[0]],
                           "one_rows": [int(i) for i in np.where(a == 1)[0]], "used_by": cons[t.name]}
        elif re.match(r"trunk\.(kc|vc|uc)_", t.name):
            out.setdefault("_const_users", {})[t.name] = cons[t.name]
    return out


def run(p, x):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads, so.inter_op_num_threads = 4, 1
    s = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(None, {"image": x[i:i + 1]})[0] for i in range(len(x))]).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    for k in ("b1", "dyn", "rotq", "x", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    m1, m2 = onnx.load(a.b1), onnx.load(a.dyn)
    summ, rows = cmp_tensors(m1.graph, m2.graph)
    s1, s2 = sink_rules(m1.graph), sink_rules(m2.graph)
    x = np.ascontiguousarray(np.load(a.x, mmap_mode="r")[:a.n]).astype(np.float32)
    e1, e2 = run(a.b1, x), run(a.dyn, x)
    n1, n2 = e1 / np.linalg.norm(e1, axis=1, keepdims=True), e2 / np.linalg.norm(e2, axis=1, keepdims=True)
    cos = (n1 * n2).sum(1)
    import onnxruntime as ort
    res = {"b1": {"path": a.b1, "sha256": sha(a.b1), "nodes": len(m1.graph.node), "n_init": len(m1.graph.initializer)},
           "dyn": {"path": a.dyn, "sha256": sha(a.dyn), "nodes": len(m2.graph.node), "n_init": len(m2.graph.initializer)},
           "rotq": {"path": a.rotq, "sha256": sha(a.rotq)},
           "tensor_summary": summ, "tensor_rows_diff_or_only": [{**r, "key": list(r["key"])} for r in rows],
           "sink_b1": s1, "sink_dyn": s2, "sink_equal": json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True),
           "output": {"n": int(len(x)), "data": f"{a.x}[0:{a.n}]", "batch": 1, "cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
                      "max_abs_raw": float(np.abs(e1 - e2).max()), "max_abs_l2n": float(np.abs(n1 - n2).max()),
                      "bit_equal_raw": bool(np.array_equal(e1, e2))},
           "ort": ort.__version__, "onnx": onnx.__version__,
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("tensor_summary", "sink_equal", "output", "commit", "dirty")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
