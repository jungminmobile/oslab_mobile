"""
s2_dfc_native_check.py — 파싱/최적화된 HAR을 float 에뮬레이션(SDK_NATIVE·SDK_FP_OPTIMIZED)으로 돌려 FP32 ONNX 임베딩과 cos 비교 (G3R G3b 진단, 10/18)
[사용] (WSL Ubuntu-22.04, hailo10_venv)  python s2_dfc_native_check.py --har X.har --ctx native|fp_optimized --data pets_dev --k 16
  양자화 전 그래프가 맞는지(파서 변환이 계산을 바꾸지 않았는지) 확인한다. 경로는 S2_DFC_CACHE (s2_dfc_screen.py와 같음).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("S2_DFC_CACHE", "/mnt/d/mokerton/artifacts/siglip2/cache"))
ap = argparse.ArgumentParser()
ap.add_argument("--har", required=True)
ap.add_argument("--ctx", default="native", choices=["native", "fp_optimized", "quantized"])
ap.add_argument("--data", default="pets_dev")
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--bs", type=int, default=4)
a = ap.parse_args()
from hailo_sdk_client import ClientRunner, InferenceContext  # noqa: E402

ctx = {"native": InferenceContext.SDK_NATIVE, "fp_optimized": InferenceContext.SDK_FP_OPTIMIZED,
       "quantized": InferenceContext.SDK_QUANTIZED}[a.ctx]
x = np.load(CACHE / f"{a.data}_x.npy", mmap_mode="r")[:a.k].astype(np.float32)
ref = np.load(CACHE / f"{a.data}_ref_fp32.npy")[:a.k]
runner = ClientRunner(har=a.har)
with runner.infer_context(ctx) as c:
    e = np.asarray(runner.infer(c, np.ascontiguousarray(x.transpose(0, 2, 3, 1)), batch_size=a.bs)).reshape(len(x), -1)
e = e / np.linalg.norm(e, axis=1, keepdims=True)
cos = (e * ref).sum(1)
print(json.dumps({"har": Path(a.har).name, "ctx": a.ctx, "data": a.data, "k": a.k,
                  "cos_mean": round(float(cos.mean()), 6), "cos_min": round(float(cos.min()), 6)}))
