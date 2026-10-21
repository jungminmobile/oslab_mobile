"""
s2_ortfuse.py — (O1, O-B) 양자화 전 FP32 그래프 정리: ORT CPU 속도용, 수학적으로 같은 변환만
==============================================================================================
- 원인 (O1 README 2절): 잔차 Add의 출력은 LayerNorm과 다음 잔차 Add 두 곳에서 쓰인다. ORT C++ SkipLayerNormFusion은
  Add 출력의 소비자가 하나일 때만 합치므로 25개 중 마지막 `/trunk/norm` 하나만 합쳐진다.
- (1) --skip-ln: ORT python 융합기(onnxruntime.transformers)의 FusionSkipLayerNormalization으로 Add+LayerNorm →
  com.microsoft SkipLayerNormalization (소비자가 여럿이면 4번째 출력 input_skip_bias_sum으로 합을 내보낸다),
  이어서 FusionBiasSkipLayerNormalization으로 앞 MatMul의 bias Add(1차원 상수)를 SkipLayerNorm의 bias 입력으로 흡수.
- (2) --fold-shape: Reshape의 모양 입력이 Shape→Gather→Unsqueeze→Concat으로 계산되는 경우(배치 가변 때문),
  기호 모양 추론으로 출력 모양이 [batch, 고정값...]이고 입력 0번 축도 batch일 때만 상수 [0, 고정값...]으로 바꾼다
  (Reshape의 0 = 입력 축 크기 그대로, allowzero=0). 쓰이지 않게 된 모양 계산 노드는 지운다.
- 정수/가중치 값은 건드리지 않는다. 결과 이름: <src>_o1b.onnx (--skip-ln·--fold-shape 둘 다일 때). 하나만 켜면 _o1b-sln / _o1b-fs.
- 동치 확인 (--check N): 같은 입력 N장(배치 1과 배치 4)으로 원본과 결과의 출력 cos·최대 절대 차이를 낸다.
[사용] (siglip2/)  python s2_ortfuse.py --src siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx --skip-ln --fold-shape --check 64
"""
import argparse
import collections
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2common as S                     # noqa: E402
from scripts import env_info             # noqa: E402

RC = S.RC
WORK = Path(os.environ.get("G6_WORK", Path.home() / "siglip2_work" / "g6"))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ops(model):
    return dict(collections.Counter(f"{n.domain + '.' if n.domain else ''}{n.op_type}" for n in model.graph.node).most_common())


def skip_ln(model):
    import onnxruntime.transformers  # noqa: F401  (python 융합기 폴더를 sys.path에 넣는다)
    from fusion_skiplayernorm import FusionBiasSkipLayerNormalization, FusionSkipLayerNormalization
    from onnx_model import OnnxModel
    m = OnnxModel(model)
    n_ln = sum(n.op_type == "LayerNormalization" for n in m.model.graph.node)
    FusionSkipLayerNormalization(m).apply()
    n_sln = sum(n.op_type == "SkipLayerNormalization" for n in m.model.graph.node)
    FusionBiasSkipLayerNormalization(m).apply()
    n_bias = sum(n.op_type == "SkipLayerNormalization" and len(n.input) >= 5 and bool(n.input[4]) for n in m.model.graph.node)
    m.prune_graph()
    m.topological_sort()
    if not any(o.domain == "com.microsoft" for o in m.model.opset_import):
        m.model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    return m.model, {"layernorm_before": n_ln, "skip_layernorm": n_sln, "with_bias": int(n_bias)}


def fold_shape(model):
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    inf = SymbolicShapeInference.infer_shapes(model, auto_merge=True)
    vi = {v.name: v for v in (*inf.graph.value_info, *inf.graph.input, *inf.graph.output)}

    def dims(t):
        if t not in vi:
            return None
        return [d.dim_param if d.dim_param else (d.dim_value if d.HasField("dim_value") else None)
                for d in vi[t].type.tensor_type.shape.dim]
    batch = model.graph.input[0].type.tensor_type.shape.dim[0].dim_param
    assert batch, "입력 0번 축이 기호(배치 가변)가 아님"
    inits = {i.name for i in model.graph.initializer}
    prod = {o: n for n in model.graph.node for o in n.output}
    changed, kept = [], []
    for n in model.graph.node:
        if n.op_type != "Reshape" or n.input[1] in inits:
            continue
        p = prod.get(n.input[1])
        if p is None or p.op_type == "Constant":
            continue
        di, do = dims(n.input[0]), dims(n.output[0])
        if not (di and do and di[0] == batch and do[0] == batch and all(isinstance(d, int) and d > 0 for d in do[1:])):
            kept.append(n.name)
            continue
        if any(a.name == "allowzero" and a.i for a in n.attribute):
            kept.append(n.name)
            continue
        c = f"{n.name}_o1b_shape"
        model.graph.initializer.append(numpy_helper.from_array(np.array([0, *do[1:]], np.int64), c))
        n.input[1] = c
        changed.append({"node": n.name, "shape": [0, *do[1:]]})
    # 쓰이지 않게 된 노드 제거 (그래프 출력에서 거꾸로)
    need = {o.name for o in model.graph.output}
    keep = []
    for n in reversed(model.graph.node):
        if any(o in need for o in n.output):
            keep.append(n)
            need.update(i for i in n.input if i)
    removed = collections.Counter(n.op_type for n in model.graph.node if n not in keep)
    del model.graph.node[:]
    model.graph.node.extend(reversed(keep))
    used = {i for n in model.graph.node for i in n.input}
    for i in [i for i in model.graph.initializer if i.name not in used]:
        model.graph.initializer.remove(i)
    return model, {"reshape_folded": len(changed), "reshape_kept": kept, "removed_nodes": dict(removed), "folded": changed}


def check(src, dst, n):
    import onnxruntime as ort
    size = onnx.load(str(src), load_external_data=False).graph.input[0].type.tensor_type.shape.dim[2].dim_value
    pix = np.load(WORK / f"dev_zoo{size}" / "pets_dev_pix.npy", mmap_mode="r")
    x = (np.asarray(pix[:n], np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2).copy()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = 4
    so.log_severity_level = 3
    out = {}
    for k, p in (("src", src), ("dst", dst)):
        s = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
        b1 = np.concatenate([s.run(None, {"image": x[i:i + 1]})[0] for i in range(n)])
        b4 = np.concatenate([s.run(None, {"image": x[i:i + 4]})[0] for i in range(0, n, 4)])
        out[k] = (b1, b4)
    l2n = lambda v: v / np.linalg.norm(v, axis=-1, keepdims=True)
    r = {"n": n, "size": size, "data": f"pets_dev[0:{n}] (Zoo 전처리 캐시)"}
    for j, nm in ((0, "b1"), (1, "b4")):
        a, b = out["src"][j], out["dst"][j]
        cs = (l2n(a) * l2n(b)).sum(1)
        r[nm] = {"cos_min": float(cs.min()), "cos_mean": float(cs.mean()), "max_abs": float(np.abs(a - b).max()),
                 "max_abs_rel": float(np.abs(a - b).max() / np.abs(a).max())}
    a1, a4 = out["dst"]
    r["dst_b1_vs_b4_max_abs"] = float(np.abs(a1 - a4).max())
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="FP32 ONNX 이름 ($RN50_ONNX 아래, siglip2_image_fp32로 시작)")
    ap.add_argument("--skip-ln", action="store_true")
    ap.add_argument("--fold-shape", action="store_true")
    ap.add_argument("--check", type=int, default=64, help="동치 확인 장수 (4의 배수), 0이면 안 함")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    assert a.skip_ln or a.fold_shape, "--skip-ln·--fold-shape 중 하나 이상"
    assert a.src.startswith("siglip2_image_fp32") and a.src.endswith(".onnx")
    src = RC.ONNX_DIR / a.src
    tag = "_o1b" if (a.skip_ln and a.fold_shape) else ("_o1b-sln" if a.skip_ln else "_o1b-fs")
    dst = src.with_name(src.stem + tag + ".onnx")
    if dst.exists() and not a.overwrite:
        raise SystemExit(f"{dst.name} 이미 있음 (--overwrite)")
    model = onnx.load(str(src))
    row = {"date": datetime.now().isoformat(timespec="seconds"), "src": src.name, "src_sha256": sha256(src), "ops_before": ops(model)}
    if a.skip_ln:
        model, row["skip_ln"] = skip_ln(model)
    if a.fold_shape:
        model, row["fold_shape"] = fold_shape(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    row.update({"dst": dst.name, "dst_sha256": sha256(dst), "ops_after": ops(model), "options": vars(a)})
    if a.check:
        assert a.check % 4 == 0
        row["equiv"] = check(src, dst, a.check)
    env = env_info.collect()
    row.update({"commit": env["git"]["commit"], "dirty": env["git"]["dirty"], **env_info.flat(env)})
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ortfuse_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: row[k] for k in ("dst", "skip_ln", "fold_shape", "equiv") if k in row}, ensure_ascii=False, indent=1)[:3000])


if __name__ == "__main__":
    main()
