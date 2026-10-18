"""
s2_har_residual_path.py — 최적화된 HAR에서 잔차 흐름 위의 층과 정밀도(precision_mode) 표 (G3R 5번째 시도 진단, 10/18)
[사용] (hailo10_venv, 저장소 밖 폴더에서)  python s2_har_residual_path.py <opt.har> <잔차 ew_add 목록(쉼표)> <out.csv>
  잔차 ew_add(16비트로 지정한 잔차 덧셈)에서 출발해, 다음 잔차 ew_add·LayerNorm·행렬곱·conv에 닿기 전까지 지나는 원소별 층(ew_add/ew_sub/ew_mult 등)을
  "잔차 경로 위 층"으로 본다. 각 층의 precision_mode와 원래 ONNX 노드 이름을 적는다.
"""
import csv
import sys

from hailo_sdk_client import ClientRunner

har, res_list, out = sys.argv[1], [x.strip() for x in sys.argv[2].split(",")], sys.argv[3]
hn = ClientRunner(har=har).get_hn_dict()
L = hn["layers"]
net = next(iter(L)).split("/")[0]
full = lambda s: f"{net}/{s}"
short = lambda s: s.split("/", 1)[-1]
res_all = {full(x) for x in res_list}
missing = sorted(short(x) for x in res_all if x not in L)
res = {x for x in res_all if x in L}
STOP = ("layer_normalization", "normalization", "conv", "matmul", "output_layer", "softmax", "dense")
rows, seen = [], set()
for r in sorted(res, key=lambda x: int("".join(c for c in short(x) if c.isdigit()) or 0)):
    stack = list(L[r].get("output", []))
    while stack:
        n = stack.pop()
        if n in seen or n in res or n not in L:
            continue
        t = L[n]["type"]
        if any(t.startswith(s) for s in STOP):
            continue
        seen.add(n)
        q = L[n].get("quantization_params", {})
        rows.append((short(r), short(n), t, q.get("precision_mode"), ";".join(L[n].get("original_names", []))[:80]))
        stack.extend(L[n].get("output", []))
prec = {short(x): L[x].get("quantization_params", {}).get("precision_mode") for x in res}
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["from_residual_ew_add", "layer", "type", "precision_mode", "onnx"])
    w.writerows(rows)
print("최적화 HAR에 없는(합쳐진) 잔차 층:", missing)
print("잔차 ew_add 정밀도:", sorted(set(prec.values())))
print("잔차 경로 위 다른 층:", len(rows), "개, 정밀도", sorted({r[3] for r in rows}))
for r in rows[:40]:
    print("  ", r)
