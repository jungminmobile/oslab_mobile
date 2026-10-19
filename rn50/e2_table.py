"""E2 결과표: 보정 방법 × 구성. results/<E2>/evaluate.jsonl + pred/ → 마크다운 표 (정수 커널 값만)."""
import json, os, sys, subprocess
import numpy as np
import common as C
from bootstrap import correct, paired_bootstrap
HR = ["pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
rows = [json.loads(l) for l in open(C.RESULTS_DIR / "evaluate.jsonl")]
by = {}
for r in rows:
    if r.get("exec") != "int": continue
    by.setdefault(r["model"], {})[r["data"]] = r
fp = {d: C.topk_metrics(*(np.load(C.CACHE_DIR / f"{d}_{k}.npy") for k in ("ref_fp32", "clf", "y")))["top1"] for d in ["cifar_A"] + HR}
print(f"| 방법 | 구성 | 크기 MiB | CIFAR A | 고화질 평균 | 차이(FP32 대비) | 95% 구간 | cos(고화질 평균) |")
print("|---|---|---|---|---|---|---|---|")
print(f"| FP32 | — | 146.1 | {fp['cifar_A']} | {np.mean([fp[d] for d in HR]):.2f} | 0 | — | 1.000 |")
order = {"c3": 0, "c4": 1, "c5": 2, "Fon": 3}
def key(m):
    meth = m.split("_q1_")[1].split("_")[0]; cfg = m.removesuffix(".onnx").split("_")[-1]
    return (order.get(cfg, 9), meth)
for m in sorted(by, key=key):
    d = by[m]
    if not all(x in d for x in ["cifar_A"] + HR): continue
    meth = m.split("_q1_")[1].split("_")[0]; cfg = m.removesuffix(".onnx").split("_")[-1]
    pt, lo, hi = paired_bootstrap([(correct("fp32", x), correct(m, x)) for x in HR])
    print(f"| {meth} | {cfg} | {d['pets_dev']['MiB']} | {d['cifar_A']['top1']} | {np.mean([d[x]['top1'] for x in HR]):.2f} | "
          f"{pt:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {np.mean([d[x]['cos_mean'] for x in HR]):.4f} |")
