#!/bin/bash
# E1 2단계: 양자화(optimize). 사용법: bash ~/e1_step2.sh [rn50|rn50x4|all]  (기본 all = rn50 다음 rn50x4)
set -u
W=~/hailo10_work; mkdir -p $W/opt $W/logs; cd $W/opt
source ~/hailo10_venv/bin/activate
WHICH=${1:-all}
for M in rn50 rn50x4; do
  [ "$WHICH" != all ] && [ "$WHICH" != "$M" ] && continue
  [ -f $W/opt/${M}_optimized.har ] && { echo "[$(date +%T)] $M 이미 완료 → 건너뜀"; continue; }
  echo "[$(date +%T)] ===== $M 양자화 시작 (예상 RN50 20~40분 / RN50x4 30~60분)"
  /usr/bin/time -v hailo optimize $W/parse/$M.har --hw-arch hailo10h \
     --calib-set-path $W/calib/calib_${M}_highres64_s0_nhwc_u8.npy \
     --model-script ~/e1_base.alls 2>&1 | tee $W/logs/${M}_optimize.log
  if ls $W/opt/${M}*optimized*.har >/dev/null 2>&1; then
    echo "[$(date +%T)] ===== $M 양자화 완료 ✅  $(ls $W/opt/${M}*optimized*.har)"
  else
    echo "[$(date +%T)] ===== $M 양자화 실패 ❌  로그: $W/logs/${M}_optimize.log"
  fi
done
echo "[$(date +%T)] ===== 2단계 끝. 이제 노트북을 덮어도 됩니다."
