"""
s2_har_layer_snr.py — optimize가 HAR에 저장한 층별 잡음 분석(layer_noise_analysis: 누적 snr, 층 자체 partial_snr)을 csv로 (G3R 5번째 진단, 10/18)
  `hailo analyze-noise`는 이 모델에서 DFC 5.4.0 내부 오류(차원 768 vs 780)로 원래 D256L1에서도 실패해, optimize 단계 결과를 쓴다.
[사용] (hailo10_venv, 저장소 밖 폴더에서)  python s2_har_layer_snr.py <opt.har> <out.csv>
"""
import csv
import sys

import numpy as np
from hailo_sdk_client import ClientRunner

r = ClientRunner(har=sys.argv[1])
st = r.get_params_statistics()
L = r.get_hn_dict()["layers"]
rows = []
for name, l in L.items():
    k = f"{name}/layer_noise_analysis/snr"
    if k not in st:
        continue
    g = lambda s: float(np.asarray(st[f"{name}/layer_noise_analysis/{s}"]).ravel()[0]) if f"{name}/layer_noise_analysis/{s}" in st else float("nan")
    rows.append((name.split("/", 1)[1], l["type"], l.get("quantization_params", {}).get("precision_mode"),
                 round(g("snr"), 3), round(g("partial_snr"), 3), ";".join(l.get("original_names", []))[:70]))
with open(sys.argv[2], "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["layer", "type", "precision_mode", "snr_cumulative_db", "partial_snr_db", "onnx"])
    w.writerows(rows)
print(len(rows), "layers →", sys.argv[2])
