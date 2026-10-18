"""
s2_har_layers.py — HAR의 층 이름 ↔ 원래 ONNX 노드 이름 표 (layer_map_siglip2.csv와 같은 형식) (G3R G3b, 10/18)
[사용] (WSL Ubuntu-22.04, hailo10_venv)  python s2_har_layers.py <har> <out.csv>
  층 이름이 그래프에 따라 바뀌므로(.alls의 conv 번호), 새 그래프로 .alls를 만들기 전에 이 표로 확인한다.
"""
import csv
import sys

from hailo_sdk_client import ClientRunner

har, out = sys.argv[1], sys.argv[2]
hn = ClientRunner(har=har).get_hn_dict()
rows = []
for name, l in hn["layers"].items():
    short = name.split("/", 1)[-1]
    rows.append((short, l.get("type"), ";".join(l.get("original_names", []))))
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["dfc_layer", "dfc_type", "onnx_nodes"])
    w.writerows(rows)
print(len(rows), "layers →", out)
