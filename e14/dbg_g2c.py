"""dbg_g2c.py — G2(c) 불통과 원인 확인 (K 자리 7곳). 진단 사본에서 out·X를 꺼내 shape와 관계를 본다."""
import sys, numpy as np, onnx, onnxruntime as ort
from pathlib import Path
from onnx import numpy_helper as nh
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "e12"))
import rewrite_mask as RW, session_opts as SO
O = r"C:\s2art\onnx\e12_3\siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx"
m0 = onnx.load(O); init = {i.name: nh.to_array(i) for i in m0.graph.initializer}
sites = {r["sub"]: r for r in RW.find_sites(m0)}
pix = np.load(r"C:\s2art_b32\dev256\pets_dev_pix.npy", mmap_mode="r")[:1]
x = np.ascontiguousarray((np.asarray(pix, np.float32) / 127.5 - 1).transpose(0, 3, 1, 2))
for lvl in ("ALL", "DISABLE"):
    so = SO.make_options("C3", 4)
    if lvl == "DISABLE": so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(r"C:\s2art_b32\e14\diag_siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx", so, providers=["CPUExecutionProvider"])
    on = [o.name for o in s.get_outputs()]
    r = dict(zip(on, s.run(None, {s.get_inputs()[0].name: x})))
    for k in ("/trunk/Sub_1", "/trunk/Sub_2", "/trunk/Sub_13", "/trunk/Sub_15"):
        t = sites[k]; o, X, Y = r[t["out"]], r[t["X"]], r[t["Y"]]
        ref = Y + init[t["m"]] * (init[t["c"]] - X)
        print(lvl, k, "out", o.shape, "X", X.shape, "Y", Y.shape, "|out-ref|", float(np.abs(o - ref).max()),
              "|out-Y|", float(np.abs(o - Y).max()), "|out-Y^T(0,3,2,1)|", float(np.abs(o - Y.transpose(0, 3, 2, 1)).max()) if o.ndim == 4 and o.shape == Y.transpose(0,3,2,1).shape else "-")
