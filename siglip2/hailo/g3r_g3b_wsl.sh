#!/usr/bin/env bash
# g3r_g3b_wsl.sh — G3R G3b: 제자리 마스크(bcast) HAR → optimize(구성별 .alls) → SDK_QUANTIZED 에뮬레이션 스크리닝 (dev 6종 × 앞 200장)
# WSL Ubuntu-22.04 (연구실 데스크탑), ~/hailo10_venv, CUDA_VISIBLE_DEVICES=-1. 시작 전 free 여유 ≥ MIN_FREE_GB를 기다린다 (회전 실험과 메모리 공유).
# 사용: bash g3r_g3b_wsl.sh A256L1_all8 B256L1_nosink16      (구성 이름 X → g3r_X.alls)
set -uo pipefail
source ~/hailo10_venv/bin/activate
export CUDA_VISIBLE_DEVICES=-1 S2_DFC_CACHE=$HOME/siglip2_work/artifacts/cache S2_DFC_REPO=$HOME/oslab_mobile_yong
MIN_FREE_GB=${MIN_FREE_GB:-15}
H=$(cd "$(dirname "$0")" && pwd)
# SRC_HAR: 파싱된 HAR (기본 = 제자리 마스크 bcast), OUT: 결과 폴더 이름 (기본 g3b). LOCK: DFC 공용 잠금 (회전 세션과 같은 파일, 10/18)
SRC_HAR=${SRC_HAR:-$HOME/siglip2_work/hailo_g3r_maskbc/siglip2_rcmaskbc55at5_13at9.har}
LOCK=${LOCK:-$HOME/siglip2_work/dfc.lock}
HAR=$(basename $SRC_HAR)
W=$HOME/siglip2_work/g3b_$(date +%Y%m%d_%H%M%S); mkdir -p $W; cd $W
cp $SRC_HAR $HOME/siglip2_work/artifacts/dfc/calib_highres256_s0_nhwc.npy .
O=$S2_DFC_REPO/results/siglip2/G3R_regcache/${OUT:-g3b}; mkdir -p $O
echo "commit=$(git -C $S2_DFC_REPO rev-parse --short HEAD) dirty=$(git -C $S2_DFC_REPO status --porcelain --untracked-files=no | wc -l) cfgs=$* har=$HAR har_sha256=$(sha256sum $HAR | cut -c1-64) calib_sha256=$(sha256sum calib_highres256_s0_nhwc.npy | cut -c1-64) lock=$LOCK" | tee -a $O/run_info.txt
waitmem() { while true; do f=$(free -g | awk '/Mem:/{print $7}'); [ "$f" -ge "$MIN_FREE_GB" ] && { echo "$(date +%T) $1 free=${f}G 시작"; return; }; echo "$(date +%T) $1 free=${f}G < ${MIN_FREE_GB}G 대기"; sleep 60; done; }
for cfg in "$@"; do
  cp $H/g3r_$cfg.alls .
  echo "$(date +%T) optimize $cfg 잠금 대기" | tee -a $O/run_info.txt
  (
    flock 9; echo "$(date +%T) optimize $cfg 잠금 잡음" | tee -a $O/run_info.txt
    waitmem "optimize $cfg" | tee -a $O/run_info.txt
    t0=$(date +%s)
    hailo optimize $HAR --hw-arch hailo10h --model-script g3r_$cfg.alls --calib-set-path calib_highres256_s0_nhwc.npy \
        --output-har-path opt_$cfg.har > optimize_$cfg.log 2>&1
    rc=$?; echo "$(date +%T) $cfg optimize rc=$rc sec=$(( $(date +%s) - t0 ))" | tee -a $O/run_info.txt; exit $rc
  ) 9>$LOCK
  rc=$?
  cp optimize_$cfg.log g3r_$cfg.alls $O/
  [ $rc -eq 0 ] || continue
  sha256sum opt_$cfg.har >> $O/har_sha256.txt
  echo "$(date +%T) screen $cfg 잠금 대기" | tee -a $O/run_info.txt
  (
    flock 9; echo "$(date +%T) screen $cfg 잠금 잡음" | tee -a $O/run_info.txt
    waitmem "screen $cfg" | tee -a $O/run_info.txt
    t0=$(date +%s)
    python $H/s2_dfc_screen.py --har opt_$cfg.har --tag $cfg --k 200 --bs 8 --out $O > screen_$cfg.log 2>&1
    echo "$(date +%T) $cfg screen rc=$? sec=$(( $(date +%s) - t0 )) $(tail -1 screen_$cfg.log)" | tee -a $O/run_info.txt
  ) 9>$LOCK
  cp screen_$cfg.log $O/
done
echo "G3B DONE $W" | tee -a $O/run_info.txt
