#!/usr/bin/env bash
# g6c_build.sh — B16_rb → B16_both (G6 g6_dfc.sh, HAR·보정은 B16 M2rot2 빌드와 같은 파일, G3RQ repro)
# 하나가 끝날 때마다 sha256·크기·컨텍스트 수를 build.jsonl에 바로 적는다. 실패하면 멈춘다.
# 사용: nohup bash g6c_build.sh > ~/siglip2_work/g6/g6c_build.log 2>&1 < /dev/null &
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); G6=$H/../G6; W=$HOME/siglip2_work/g6
B=$H/build.jsonl
build() {   # tag alls
  local tag=$1
  TAG=$tag ONNX=$HOME/siglip2_work/artifacts/onnx/siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad_b1.onnx HAR=$HOME/siglip2_work/hailo_repro/siglip2_m2rot.har NET=siglip2_m2rot SIZE=224 ALLS=$2 CALIB=$HOME/siglip2_work/artifacts/dfc/calib_highres256_s0_nhwc.npy CALIB_DOMAIN=normalized NCAL=256 bash $G6/g6_dfc.sh > $W/dfc_$tag.out 2>&1
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
# 1차 b16_rb(resources_param 1.0)는 compile 실패 (README 1절 끝) → b16_both 먼저, 그다음 max_compute 0.9로 다시
build b16_both $H/alls/g6c_b16_both.alls
build b16_rb09 $H/alls/g6c_b16_rb09.alls
echo "G6C BUILD ALL DONE $(date +%T)"
