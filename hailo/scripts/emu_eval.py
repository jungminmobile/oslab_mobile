"""E1 4단계-③: DFC 에뮬레이터로 dev 정확도 측정 (Hailo-10H 장비 없이 노트북 CPU에서).
사용: python emu_eval.py --har <optimized.har> --ctx quantized|fp --data pets_dev --n 200 --tag rn50_c4
- 입력: dev 캐시(uint8, NCHW, 224 crop) → NHWC float 0~255 (정규화는 .alls의 normalization 층이 NPU 안에서 처리)
- 출력: top1/top5, FP32(ORT) 기준 임베딩과의 cos. 결과는 ~/hailo10_work/results/emu_eval.jsonl
"""
import argparse, glob, json, os, time
from datetime import datetime
import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext

ap = argparse.ArgumentParser()
ap.add_argument("--har", required=True); ap.add_argument("--ctx", choices=["quantized", "fp"], default="quantized")
ap.add_argument("--data", default="pets_dev"); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--tag", default="")
a = ap.parse_args()
cache = glob.glob("/mnt/c/study/3/2/2026*/oslab_mobile/rn50/cache")[0]
L = lambda k: np.load(f"{cache}/{a.data}_{k}.npy", mmap_mode="r")
x8, y, W, ref = L("x"), np.asarray(L("y")), np.asarray(L("clf")), np.asarray(L("ref_fp32"))
n = min(a.n, len(y))
x = np.ascontiguousarray(np.asarray(x8[:n]).transpose(0, 2, 3, 1)).astype(np.float32)
ctx = {"quantized": InferenceContext.SDK_QUANTIZED, "fp": InferenceContext.SDK_FP_OPTIMIZED}[a.ctx]
r = ClientRunner(har=a.har)
t0 = time.time()
with r.infer_context(ctx) as c:
    out = r.infer(c, x, batch_size=a.bs)
sec = time.time() - t0
if isinstance(out, (list, tuple)): out = out[0]
if isinstance(out, dict): out = list(out.values())[0]
feat = np.asarray(out, np.float32).reshape(n, -1)
feat /= np.linalg.norm(feat, axis=1, keepdims=True)
top5 = np.argsort(-(feat @ W), 1)[:, :5]
cos = (feat * ref[:n]).sum(1)
ref_top1 = (np.argmax(ref[:n] @ W, 1) == y[:n]).mean() * 100
row = {"date": datetime.now().isoformat(timespec="seconds"), "tag": a.tag, "har": os.path.basename(a.har), "ctx": a.ctx,
       "data": a.data, "n": n, "top1": round(float((top5[:, 0] == y[:n]).mean() * 100), 2),
       "top5": round(float((top5 == y[:n, None]).any(1).mean() * 100), 2),
       "fp32_ort_top1_same_imgs": round(float(ref_top1), 2), "cos_mean": round(float(cos.mean()), 4),
       "cos_p1": round(float(np.percentile(cos, 1)), 4), "sec": round(sec, 1), "sec_per_img": round(sec / n, 3)}
print(json.dumps(row, ensure_ascii=False))
os.makedirs(os.path.expanduser("~/hailo10_work/results"), exist_ok=True)
with open(os.path.expanduser("~/hailo10_work/results/emu_eval.jsonl"), "a") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")
