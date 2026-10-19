#!/usr/bin/env bash
# g6_build_all.sh — README 2절 순서로 DFC 빌드: b32_V → b32_D → b32_R → b32_M → s1b16_V → s1b16_R → (s1b16_D, ALSO_S1B16_D=1일 때)
# 하나가 끝날 때마다 sha256·크기·컨텍스트 수를 build.jsonl에 바로 적고 로그를 logs/<tag>/에 복사한다. 실패하면 멈춘다.
# 사용: [ONLY=<tag>] [ALSO_S1B16_D=1] nohup bash g6_build_all.sh > ~/siglip2_work/g6/build_all.log 2>&1 < /dev/null &
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); W=$HOME/siglip2_work/g6; Z=$HOME/siglip2_work/g6pre; O=$W/onnx
B=$H/build.jsonl
build() {   # tag onnx net size alls calib ncal [start end]
  local tag=$1
  [ -n "${ONLY:-}" ] && [ "$tag" != "$ONLY" ] && return 0      # ONLY=<tag>: 그 구성만 (S1B16 D 선택 빌드)
  TAG=$tag ONNX=$2 NET=$3 SIZE=$4 ALLS=$5 CALIB=$6 NCAL=$7 START=${8:-} END=${9:-} bash $H/g6_dfc.sh > $W/dfc_$tag.out 2>&1
  local rc=$? D=$W/dfc/$tag
  mkdir -p $H/logs/$tag; cp $D/run_info.txt $D/optimize.log $D/model.alls $H/logs/$tag/ 2>/dev/null; tail -c 150000 $D/compile.log > $H/logs/$tag/compile_tail.log 2>/dev/null
  local hef=$(ls $D/comp/*.hef 2>/dev/null | head -1)
  python3 - "$tag" "$rc" "$hef" "$D" >> $B <<'PY'
import sys, json, hashlib, re, os, datetime
tag, rc, hef, d = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
info = open(f"{d}/run_info.txt").read()
row = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "what": "build", "tag": tag, "rc": rc,
       "commit_dirty": re.findall(r"commit=(\S+) dirty=(\S+)", info)[:1],
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
build b32_V $Z/dl_b32/siglip2-base-patch32-256_vision_encoder.sim.onnx siglip2_b_32_256_image_encoder 256 $H/alls/zoo_v540_b32.alls $W/calib64_s0_zoo256_nhwc.npy 64 onnx::Cast_0 1393
build b32_D $O/g6_b32_plain_b1.onnx g6_b32_D 256 $H/alls/g6_b32_D.alls $W/calib256_s0_zoo256_nhwc.npy 256
build b32_R $O/g6_b32_rot_b1.onnx g6_b32_R 256 $H/alls/g6_b32_R.alls $W/calib256_s0_zoo256_nhwc.npy 256
build b32_M $O/g6_b32_m2rot_b1.onnx g6_b32_M 256 $H/alls/g6_b32_M.alls $W/calib256_s0_zoo256_nhwc.npy 256
build s1b16_V $Z/dl_b16/siglip_base_patch16_224_vision_encoder.onnx siglip_b_16_image_encoder 224 $H/alls/zoo_v540_s1b16.alls $W/calib64_s0_zoo224_nhwc.npy 64 input.1 1725
build s1b16_R $O/g6_s1b16_rot_b1.onnx g6_s1b16_R 224 $H/alls/g6_s1b16_R.alls $W/calib256_s0_zoo224_nhwc.npy 256
[ "${ALSO_S1B16_D:-0}" = 1 ] && build s1b16_D $O/g6_s1b16_plain_b1.onnx g6_s1b16_D 224 $H/alls/g6_s1b16_D.alls $W/calib256_s0_zoo224_nhwc.npy 256
echo "G6 BUILD ALL DONE $(date +%T)"
