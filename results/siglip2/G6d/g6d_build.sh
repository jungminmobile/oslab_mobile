#!/usr/bin/env bash
# g6d_build.sh — README 2절 순서: Dmax(compile만) → S1 → S2 → F(compile만) → Vrot → G. B32 compile 상한 2시간. 실패해도 다음으로 (서로 독립).
# 하나 끝날 때마다 build.jsonl에 sha256·크기·컨텍스트·시간을 바로 적는다.
# 사용: nohup bash g6d_build.sh > ~/siglip2_work/g6/g6d_build.log 2>&1 < /dev/null &
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); G6=$H/../G6; W=$HOME/siglip2_work/g6; Z=$HOME/siglip2_work/g6pre; B=$H/build.jsonl
MB_HAR=$W/dfc/b32_M_both/g6_b32_M_both_optimized.har
full() {   # tag onnx net alls calib ncal [start end]
  local tag=$1
  TAG=$tag ONNX=$2 NET=$3 SIZE=256 ALLS=$4 CALIB=$5 NCAL=$6 START=${7:-} END=${8:-} COMPILE_TIMEOUT=7200 bash $G6/g6_dfc.sh > $W/dfc_$tag.out 2>&1
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
bash $H/g6d_compile_only.sh b32_Dmax $MB_HAR $H/alls/g6d_b32_Dmax.alls 7200
full b32_S1 $W/onnx/g6_b32_rot_b1.onnx g6_b32_S1 $H/alls/g6d_b32_S1.alls $W/calib64_s0_zoo256_nhwc.npy 64
full b32_S2 $W/onnx/g6_b32_m2rot_b1.onnx g6_b32_S2 $H/alls/g6d_b32_S2.alls $W/calib64_s0_zoo256_nhwc.npy 64
bash $H/g6d_compile_only.sh b32_F $MB_HAR $H/alls/g6d_b32_F.alls 7200
full b32_Vrot $HOME/siglip2_work/g6d/vrot/siglip2-base-patch32-256_vision_encoder_vrot.onnx siglip2_b_32_256_image_encoder $H/alls/g6d_b32_Vrot.alls $W/calib64_s0_zoo256_nhwc.npy 64 onnx::Cast_0 1393
full b32_G $W/onnx/g6_b32_m2rot_b1.onnx g6_b32_G $H/alls/g6d_b32_G.alls $W/calib256_s0_zoo256_nhwc.npy 256
echo "G6D BUILD ALL DONE $(date +%T)"
