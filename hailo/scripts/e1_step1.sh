#!/bin/bash
# E1 1단계: ONNX 파싱 (hailo10h) + DFC 층 목록/매핑표
set -u
W=~/hailo10_work; mkdir -p $W/parse; cd $W/parse
source ~/hailo10_venv/bin/activate
for M in rn50 rn50x4; do
  echo "===== PARSE $M  $(date +%T)"
  /usr/bin/time -v hailo parser onnx $W/onnx/${M}_image_fp32_q1.onnx --hw-arch hailo10h -y \
     --har-path $W/parse/${M}.har --parsing-report-path $W/parse/${M}_parsing_report.json > $W/parse/${M}_parse.log 2>&1
  echo "exit=$?"
  grep -E "Start nodes|End nodes|Translation completed|Saved HAR|rror|nsupported|recommend|Unsupported|WARNING|warning" $W/parse/${M}_parse.log | cut -c1-300 | head -30
  grep -E "Elapsed|Maximum resident" $W/parse/${M}_parse.log
done
echo "===== LAYER MAP  $(date +%T)"
python - <<'PY'
import json, csv, os
from hailo_sdk_client import ClientRunner
W=os.path.expanduser("~/hailo10_work/parse")
for m in ["rn50","rn50x4"]:
    p=f"{W}/{m}.har"
    if not os.path.exists(p): print(m,"no har"); continue
    r=ClientRunner(har=p)
    hn=r.get_hn_dict() if hasattr(r,"get_hn_dict") else json.loads(r.get_hn())
    layers=hn["layers"]
    rows=[]
    from collections import Counter
    c=Counter()
    for name,L in layers.items():
        t=L.get("type"); c[t]+=1
        rows.append([name,t,";".join(L.get("original_names",[])),str(L.get("input_shapes",""))[:60],str(L.get("output_shapes",""))[:60]])
    with open(f"{W}/layer_map_{m}.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["dfc_layer","type","onnx_nodes","input_shapes","output_shapes"]); w.writerows(rows)
    print(m,"layers",len(rows),dict(c))
    tail=[r_ for r_ in rows if any(k in r_[2] for k in ("attnpool",))]
    print(m,"attnpool layers",len(tail))
    for r_ in tail[:40]: print("  ",r_[0],r_[1],r_[2][:120])
PY
echo "===== STEP1 DONE $(date +%T)"
