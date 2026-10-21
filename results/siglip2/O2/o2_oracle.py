"""O2 FP32 동치 확인 (기록): B16 통일 M2rot (s2_g6 export와 같은 PyTorch 모델) vs 독립 HF 구현.
B16 timm은 정확한 GELU(erf), HF SigLIP2 설정은 tanh 근사 GELU라 원래 모델끼리도 cos 최소 0.99999를 못 넘는다 (G6_pre m2xchk 0.99998).
그래서 HF MLP 활성함수를 timm과 같은 정확한 GELU로 바꾼 비교(oracle)를 판정에 쓰고, tanh(기존 방식) 값도 같이 적는다.
+ 통일 FP32 ONNX가 Release FP32·기존 M2rot FP32와 얼마나 다른지 (dev64, ORT).
사용 (siglip2/):  RN50_RESULTS=../results/siglip2/O2/export python ../results/siglip2/O2/o2_oracle.py
"""
import os, sys, json
os.environ["S2_MODEL_ID"] = "hf-hub:timm/ViT-B-16-SigLIP2"
sys.argv = [sys.argv[0], "--model", "b16"]
sys.path.insert(0, os.getcwd())
import numpy as np, torch
import s2_g6 as G
import s2_g6pre as P
from transformers.activations import ACT2FN

x = G.dev64(224)
xcal = np.load(G.PRE / "cache_zoo224" / "calib256_s0_x.npy")
sinks = [(p, k, u) for (p, k), u in zip(*G.UNIFIED["b16"])]
row = {"what": "o2_oracle", "n": len(x), "data": "dev64 (pets 32 + caltech 32, Zoo 224)", "unified": G.UNIFIED_TAG["b16"]}

plain, _ = G.load_visual(False)
model, qsha = G.load_visual(True)
rc, _, _ = G.m2_trunk(model, xcal, "b16")
model.visual.trunk = rc
e_t = G.feat(model.visual, x)
o_t = G.feat(plain.visual, x)
row["rotq_sha256"] = qsha
for act in ("gelu_pytorch_tanh", "gelu"):
    vis = P.vis_of(P.load_hf("ref"))
    assert vis.config.hidden_act == "gelu_pytorch_tanh", vis.config.hidden_act
    if act == "gelu":
        f = ACT2FN["gelu"]
        for lyr in vis.encoder.layers:
            lyr.mlp.activation_fn = f
        vis.head.mlp.activation_fn = f
    kvc, ucs, n = P.build_m2_consts(vis, xcal, sinks)
    assert n == 256
    e_h = P.batched(lambda t: P.hf_forward(vis, t, sinks, kvc, ucs), x)
    o_h = P.batched(lambda t: P.hf_forward(vis, t), x)
    c, c0 = P.cosr(e_t, e_h), P.cosr(o_t, o_h)
    row[f"hf_{act}"] = {"m2rot_timm_vs_hf_m2_cos_min": float(c.min()), "m2rot_timm_vs_hf_m2_cos_mean": float(c.mean()),
                        "orig_timm_vs_hf_orig_cos_min": float(c0.min()), "orig_timm_vs_hf_orig_cos_mean": float(c0.mean())}
row["oracle_pass_cos_min_ge_0.99999"] = row["hf_gelu"]["m2rot_timm_vs_hf_m2_cos_min"] >= 0.99999

import onnxruntime as ort
def onnx_emb(name):
    so = ort.SessionOptions(); so.intra_op_num_threads = 6; so.log_severity_level = 3
    s = ort.InferenceSession(str(G.RC.ONNX_DIR / name), so, providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(None, {"image": x[i:i + 16]})[0] for i in range(0, len(x), 16)])
u = onnx_emb("siglip2_image_fp32_o2_b16_m2rotu.onnx")
for k, nm in (("vs_release_fp32", "siglip2_image_fp32.onnx"), ("vs_old_m2rot_fp32", "siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx")):
    c = P.cosr(u, onnx_emb(nm))
    row[f"unified_onnx_{k}"] = {"file": nm, "cos_min": float(c.min()), "cos_mean": float(c.mean())}
row["unified_onnx_vs_torch_cos_min"] = float(P.cosr(u, e_t).min())
G.append("oracle.jsonl", row)
