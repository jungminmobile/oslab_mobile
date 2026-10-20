#!/usr/bin/env bash
# g6b_build.sh — README 2절 순서로 DFC 빌드: M_rb → M_z16 → M_both (G6 g6_dfc.sh 그대로, ONNX·보정은 G6 M과 같음)
# 하나가 끝날 때마다 sha256·크기·컨텍스트 수를 build.jsonl에 바로 적는다. 실패하면 멈춘다.
# 사용: nohup bash g6b_build.sh > ~/siglip2_work/g6/g6b_build.log 2>&1 < /dev/null &
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); G6=$H/../G6; W=$HOME/siglip2_work/g6
B=$H/build.jsonl
build() {   # tag alls
  local tag=$1
  TAG=$tag ONNX=$W/onnx/g6_b32_m2rot_b1.onnx NET=g6_$tag SIZE=256 ALLS=$2 CALIB=$W/calib256_s0_zoo256_nhwc.npy NCAL=256 bash $G6/g6_dfc.sh > $W/dfc_$tag.out 2>&1
  local rc=$? D=$W/dfc/$tag
  mkdir -p $H/logs/$tag; cp $D/run_info.txt $D/optimize.log $D/model.alls $H/logs/$tag/ 2>/dev/null; tail -c 150000 $D/compile.log > $H/logs/$tag/compile_tail.log 2>/dev/null
  local hef=$(ls $D/comp/*.hef 2>/dev/null | head -1)
  python3 - "$tag" "$rc" "$hef" "$D" >> $B <<'PY'
import sys, json, hashlib, re, os, datetime
tag, rc, hef, d = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
info = open(f"{d}/run_info.txt").read()
row = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "what": "build", "tag": tag, "rc": rc,
       "commit_dirty": re.findall(r"commit=(\S+) dirty=(\S+)", info)[:1],
       "onnx_sha256": (re.findall(r"onnx_sha256=(\w+)", info) or [None])[0], "alls_sha256": (re.findall(r"alls_sha256=(\w+)", info) or [None])[0],
       "steps": {m[0]: {"rc": int(m[1]), "sec": int(m[2]), "maxrss_kb": int(m[3] or 0)} for m in re.findall(r"(parse|optimize|compile) rc=(\d+) sec=(\d+) maxrss_kb=(\d*)", info)},
       "calib_log_counts": re.findall(r"Using dataset with (\d+) entries", open(f"{d}/optimize.log").read()) if os.path.exists(f"{d}/optimize.log") else None}
if hef:
    row.update(hef=hef.replace(os.path.expanduser("~"), "~"), bytes=os.path.getsize(hef), sha256=hashlib.sha256(open(hef, "rb").read()).hexdigest(),
               contexts=len(set(re.findall(r"_context_(\d+)", open(f"{d}/compile.log").read()))))
print(json.dumps(row, ensure_ascii=False))
PY
  tail -1 $B
  [ $rc -eq 0 ] || { echo "BUILD FAIL $tag — 멈춤"; exit 1; }
}
build b32_M_rb $H/alls/g6b_b32_M_rb.alls
build b32_M_z16 $H/alls/g6b_b32_M_z16.alls
build b32_M_both $H/alls/g6b_b32_M_both.alls
echo "G6B BUILD ALL DONE $(date +%T)"
