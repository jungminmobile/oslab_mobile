#!/bin/bash
# E1 4단계-②: c4 설정으로 양자화. 사용법: bash ~/e1_step4_opt.sh rn50
set -u; M=${1:-rn50}; CFG=${2:-c4}
W=~/hailo10_work; mkdir -p $W/opt_$CFG $W/logs; cd $W/opt_$CFG
source ~/hailo10_venv/bin/activate
echo "[$(date +%T)] ===== $M $CFG 양자화 시작"
/usr/bin/time -v hailo optimize $W/parse/$M.har --hw-arch hailo10h \
   --calib-set-path $W/calib/calib_${M}_highres64_s0_nhwc_u8.npy \
   --model-script ~/${M}_${CFG}.alls 2>&1 | tee $W/logs/${M}_${CFG}_optimize.log
ls $W/opt_$CFG/${M}*optimized*.har >/dev/null 2>&1 && echo "[$(date +%T)] ===== $M $CFG 양자화 완료 ✅" || echo "[$(date +%T)] ===== $M $CFG 양자화 실패 ❌"
