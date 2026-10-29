"""onnx_counts.py — 제출 ONNX의 노드·파라미터 개수 (보고서 2장·5.2의 92.10M, 0.16M, Q/DQ 개수 출처, 2026-10-24)
최적화 전 파일 그대로 센다(ORT가 실행 때 융합한 뒤의 개수와 다름).
python siglip2/scripts/onnx_counts.py <제출 O-AB onnx(7a01044c)> <원래 FP32 onnx(be19a8ad)> <바탕 M2rot FP32 onnx(23236123, Release)>
레지스터 캐시 상수는 바탕 FP32에서 센다(제출 파일에서는 O-B·양자화 때 이름이 바뀜)
"""
import collections, hashlib, json, sys
from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper

out = {}
for tag, p in zip(("oab", "fp32", "m2rot_fp32"), sys.argv[1:4]):
    m = onnx.load(p)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    cnt = collections.Counter(n.op_type for n in g.node)
    sha = hashlib.sha256(Path(p).read_bytes()).hexdigest()
    r = {"file": Path(p).name, "sha256": sha, "op_counts": dict(sorted(cnt.items())),
         "n_initializer": len(inits), "initializer_elems_total": int(sum(np.prod(i.dims) for i in inits.values()))}
    by_dtype = collections.Counter()
    for i in inits.values():
        by_dtype[onnx.TensorProto.DataType.Name(i.data_type)] += int(np.prod(i.dims))
    r["initializer_elems_by_dtype"] = dict(by_dtype)
    if tag == "oab":
        # MatMul 가중치(INT8로 저장): DequantizeLinear 입력 0이 INT8 initializer인 것
        w8 = {n.input[0] for n in g.node if n.op_type == "DequantizeLinear" and n.input[0] in inits and inits[n.input[0]].data_type == onnx.TensorProto.INT8}
        r["int8_weight_tensors"] = len(w8)
        r["int8_weight_elems"] = int(sum(np.prod(inits[k].dims) for k in w8))
    # 레지스터 캐시 상수 (M2): kc_/vc_/uc_/zm_/m_b 이름
    rc = [k for k in inits if k.split("/")[-1].split(".")[-1].startswith(("kc_", "vc_", "uc_", "zm_", "m_b"))]
    r["register_cache_tensors"] = len(rc)
    r["register_cache_elems"] = int(sum(np.prod(inits[k].dims) for k in rc))
    r["register_cache_names_sample"] = sorted(rc)[:8]
    out[tag] = r
print(json.dumps(out, ensure_ascii=False, indent=1))
