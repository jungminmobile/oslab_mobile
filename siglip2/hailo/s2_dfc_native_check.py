"""
s2_dfc_native_check.py — HAR의 SDK_NATIVE(float) 에뮬레이션이 FP32 ONNX와 동치인지 (dev 64장 cos 최소) (빌드 재현성 기록용, 10/19)
[사용] (WSL, hailo10_venv, CUDA_VISIBLE_DEVICES=-1)  python s2_dfc_native_check.py --har <parsed.har> --out <jsonl>
  사진: dev 캐시 Pets·Caltech 앞 32장씩 (G3Q·G3R와 같은 구성), 기준: <set>_ref_fp32.npy (원래 FP32 ONNX 임베딩)
"""
import argparse, hashlib, json, os, time
from datetime import datetime
from pathlib import Path
import numpy as np

CACHE = Path(os.environ.get("S2_DFC_CACHE", os.path.expanduser("~/siglip2_work/artifacts/cache")))


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


ap = argparse.ArgumentParser()
ap.add_argument("--har", required=True)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--out", required=True)
a = ap.parse_args()
from hailo_sdk_client import ClientRunner, InferenceContext
x = np.concatenate([np.load(CACHE / f"{d}_x.npy", mmap_mode="r")[:a.n // 2] for d in ("pets_dev", "caltech_dev")]).astype(np.float32)
ref = l2n(np.concatenate([np.load(CACHE / f"{d}_ref_fp32.npy")[:a.n // 2] for d in ("pets_dev", "caltech_dev")]))
r = ClientRunner(har=a.har)
t0 = time.time()
with r.infer_context(InferenceContext.SDK_NATIVE) as ctx:
    e = l2n(np.asarray(r.infer(ctx, np.ascontiguousarray(x.transpose(0, 2, 3, 1)), batch_size=8)).reshape(len(x), -1).astype(np.float32))
c = (e * ref).sum(1)
row = {"date": datetime.now().isoformat(timespec="seconds"), "har": Path(a.har).name,
       "har_sha256": hashlib.sha256(open(a.har, "rb").read()).hexdigest(), "exec": "dfc_sdk_native", "n": int(len(x)),
       "cos_min": float(c.min()), "cos_mean": float(c.mean()), "sec": round(time.time() - t0)}
print(json.dumps(row))
with open(a.out, "a") as f:
    f.write(json.dumps(row) + "\n")
