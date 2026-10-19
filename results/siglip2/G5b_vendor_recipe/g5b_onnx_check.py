"""g5b_onnx_check.py — Hailo export ONNX(벤더)와 우리 FP32 ONNX가 같은 모델인지 ORT FP32로 확인 (G5b 1단계)
둘 다 입력은 NCHW float32, SigLIP 정규화(평균·표준편차 0.5) 후 값이다 (벤더 ONNX 첫 노드가 입력에 바로 Conv).
dev 캐시 6종 × 앞 K장(기본 11 → 66장)의 임베딩 cos(L2 정규화 후)를 잰다. 최소 cos < 0.999이면 rc=2.
사용: python -I g5b_onnx_check.py --vendor <sim.onnx> --ours <fp32.onnx> --cache <cache 폴더> --out <jsonl>
"""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

SETS = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


ap = argparse.ArgumentParser()
ap.add_argument("--vendor", required=True); ap.add_argument("--ours", required=True)
ap.add_argument("--cache", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=11)
a = ap.parse_args()
repo = Path(__file__).resolve().parents[3]
git = lambda *x: subprocess.run(["git", "-C", str(repo), *x], capture_output=True, text=True).stdout.strip()
so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sv = ort.InferenceSession(a.vendor, so, providers=["CPUExecutionProvider"])
su = ort.InferenceSession(a.ours, so, providers=["CPUExecutionProvider"])
vin, uin = sv.get_inputs()[0].name, su.get_inputs()[0].name
vout = [o.name for o in sv.get_outputs()]
print("vendor in", vin, sv.get_inputs()[0].shape, "outs", [(o.name, o.shape) for o in sv.get_outputs()])
print("ours in", uin, su.get_inputs()[0].shape, "outs", [(o.name, o.shape) for o in su.get_outputs()])
cos, rows = [], []
for d in SETS:
    x = np.load(Path(a.cache) / f"{d}_x.npy", mmap_mode="r")[: a.k].astype(np.float32)
    ref = np.load(Path(a.cache) / f"{d}_ref_fp32.npy")[: a.k]
    ev = np.concatenate([sv.run(["1393"], {vin: x[i:i + 1]})[0] for i in range(len(x))])
    eu = su.run(None, {uin: x})[0]
    c = (l2n(ev) * l2n(eu)).sum(-1); cr = (l2n(eu) * l2n(ref)).sum(-1)
    cos.append(c)
    rows.append({"set": d, "n": len(x), "cos_min": float(c.min()), "cos_mean": float(c.mean()),
                 "ours_vs_cache_ref_min": float(cr.min()), "maxabs_raw": float(np.abs(ev - eu).max()),
                 "norm_vendor_mean": float(np.linalg.norm(ev, axis=-1).mean()), "norm_ours_mean": float(np.linalg.norm(eu, axis=-1).mean())})
    print(json.dumps(rows[-1]))
c = np.concatenate(cos)
res = {"what": "g5b_onnx_check", "commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
       "ort": ort.__version__, "vendor_sha256": sha(a.vendor), "ours_sha256": sha(a.ours), "n": int(len(c)),
       "cos_min": float(c.min()), "cos_mean": float(c.mean()), "pass": bool(c.min() >= 0.999), "per_set": rows}
with open(a.out, "a") as f:
    f.write(json.dumps(res) + "\n")
print("RESULT", json.dumps({k: res[k] for k in ["n", "cos_min", "cos_mean", "pass"]}))
sys.exit(0 if res["pass"] else 2)
