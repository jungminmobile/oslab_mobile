"""E1 7단계 근거: 원본 FP32 모델에서 잔차 덧셈(Add) 출력의 값 범위를 층별로 잰다.
목적: "layer4 덧셈에만 유독 큰 값(outlier)이 있어 8비트 눈금이 뭉개진다"는 가설 확인.
사용: rn50/.venv 의 python 으로  python hailo/scripts/act_range.py [--n 64]
출력: results/E1_dfc_gate/step7_add16/act_range.csv
"""
import argparse, csv, os, re, sys
import numpy as np, onnx, onnxruntime as ort
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "rn50"))
from common import MEAN, STD
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=64); a = ap.parse_args()
m = onnx.shape_inference.infer_shapes(onnx.load(os.path.join(ROOT, "rn50/onnx/rn50_image_fp32_q1.onnx")))
vi = {v.name: v for v in m.graph.value_info}
adds = [(n.name, n.output[0]) for n in m.graph.node if n.op_type == "Add" and re.search(r"layer\d", n.name)]
for _, o in adds: m.graph.output.append(vi[o])
s = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
x8 = np.load(os.path.join(ROOT, "rn50/cache/pets_dev_x.npy"), mmap_mode="r")[: a.n]
x = ((np.asarray(x8, np.float32) / 255.0 - np.array(MEAN, np.float32)[:, None, None]) / np.array(STD, np.float32)[:, None, None]).astype(np.float32)
inp = s.get_inputs()[0].name
outs = s.run([o for _, o in adds], {inp: x})
rows = []
for (name, _), v in zip(adds, outs):
    av = np.abs(v).ravel(); mx = float(av.max()); p999 = float(np.percentile(av, 99.9)); p50 = float(np.percentile(av, 50))
    step8 = mx / 127.0                       # 8비트(부호 있음, 대칭) 한 칸의 크기
    crushed = float((av < step8 / 2).mean())  # 8비트로 바꾸면 0이 되는 값의 비율
    ch_max = np.abs(v).max(axis=(0, 2, 3)); top_ch = int(ch_max.argmax())
    rows.append({"add": name, "layer": re.search(r"layer(\d)", name).group(1), "max_abs": round(mx, 2), "p99.9_abs": round(p999, 3),
                 "median_abs": round(p50, 4), "max/p99.9": round(mx / p999, 1), "max/median": round(mx / max(p50, 1e-9), 0),
                 "8bit_step": round(step8, 4), "crushed_to_0_pct": round(100 * crushed, 1),
                 "top_channel": top_ch, "top_ch_max/2nd": round(float(np.sort(ch_max)[-1] / np.sort(ch_max)[-2]), 1)})
out = os.path.join(ROOT, "results/E1_dfc_gate/step7_add16/act_range.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(r)
print("saved", out, "| images", a.n)
