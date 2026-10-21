#!/usr/bin/env bash
# g6f_build.sh — README 1절 순서: GF(G HAR에서 compile만) → ALL8(parse부터) → ALL8F(ALL8 HAR에서 compile만). B32 compile 상한 2시간.
# full()은 G6d g6d_build.sh와 같다 (G6 g6_dfc.sh, build.jsonl에 sha256·크기·컨텍스트·시간). ALL8이 실패하면 ALL8F는 건너뛴다.
# 사용: nohup bash g6f_build.sh > ~/siglip2_work/g6f/g6f_build.log 2>&1 < /dev/null &
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); G6=$H/../G6; W=$HOME/siglip2_work/g6; B=$H/build.jsonl
G_HAR=$W/dfc/b32_G/g6_b32_G_optimized.har
A8_HAR=$W/dfc/b32_ALL8/g6_b32_ALL8_optimized.har
full() {   # tag onnx net alls calib ncal
  local tag=$1
  TAG=$tag ONNX=$2 NET=$3 SIZE=256 ALLS=$4 CALIB=$5 NCAL=$6 COMPILE_TIMEOUT=7200 bash $G6/g6_dfc.sh > $W/dfc_$tag.out 2>&1
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
}
bash $H/g6f_compile_only.sh b32_GF $G_HAR $H/alls/g6f_b32_GF.alls 7200
full b32_ALL8 $W/onnx/g6_b32_m2rot_b1.onnx g6_b32_ALL8 $H/alls/g6f_b32_ALL8.alls $W/calib256_s0_zoo256_nhwc.npy 256
if [ -f $A8_HAR ]; then bash $H/g6f_compile_only.sh b32_ALL8F $A8_HAR $H/alls/g6f_b32_ALL8F.alls 7200; else echo "ALL8 HAR 없음 — ALL8F 건너뜀"; fi
echo "G6F BUILD ALL DONE $(date +%T)"
