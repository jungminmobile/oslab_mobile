"""
s2_har_limvals.py — 최적화된 HAR의 층별 보정 범위(limvals_out)를 csv로 (G3R G3b 진단, 10/18)
[사용] (hailo10_venv, 저장소 밖 폴더에서 실행 — hailo_sdk.client.log가 cwd에 생김)  python s2_har_limvals.py <opt.har> <out.csv>
"""
import csv
import sys

import numpy as np
from hailo_sdk_client import ClientRunner

p = ClientRunner(har=sys.argv[1]).get_params_translated()
rows = []
for k in p.keys():
    if k.endswith("/limvals_out:0"):
        v = np.asarray(p[k], dtype=float).ravel()
        rows.append((k.split("/")[1], float(v.min()), float(v.max()), float(np.abs(v).max())))
rows.sort(key=lambda r: -r[3])
with open(sys.argv[2], "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["layer", "lim_min", "lim_max", "abs_max"])
    w.writerows([(n, round(a, 6), round(b, 6), round(c, 6)) for n, a, b, c in rows])
print(len(rows), "layers →", sys.argv[2])
