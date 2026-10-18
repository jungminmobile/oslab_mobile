#!/usr/bin/env bash
# G3Q-QFT 6절: CPU 미세조정 시험 1회 (rothad_D256L1 + finetune dataset_size=32 epochs=1 batch 2). 목적: 에러 없이 끝나는지·시간. 스크리닝 안 함.
# flock ~/siglip2_work/dfc.lock 안, free ≥ 15 GB 대기, CUDA_VISIBLE_DEVICES=-1. 작업 폴더 ~/siglip2_work/hailo_g3q (저장소 밖)
set -uo pipefail
source ~/hailo10_venv/bin/activate
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1
H=$(cd "$(dirname "$0")" && pwd); O=$HOME/oslab_rot/results/siglip2/G3Q_rotate/hailo_qft; mkdir -p $O
cd $HOME/siglip2_work/hailo_g3q
cp $H/g3q_qft_cpu_trial.alls .
CAL=$HOME/siglip2_work/artifacts/dfc/calib_highres256_s0_nhwc.npy
echo "$(date +%T) commit=$(git -C $HOME/oslab_rot rev-parse --short HEAD) dirty=$(git -C $HOME/oslab_rot status --porcelain --untracked-files=no | wc -l) 잠금 대기" | tee -a $O/cpu_trial_info.txt
flock $HOME/siglip2_work/dfc.lock bash -c '
  while [ $(free -g | awk "/Mem:/{print \$7}") -lt 15 ]; do sleep 60; done
  echo "$(date +%T) 시작 free=$(free -g | awk "/Mem:/{print \$7}")G load=$(cut -d" " -f1-3 /proc/loadavg)"
  t0=$(date +%s)
  /usr/bin/time -v hailo optimize siglip2_rothad_b16_224.har --hw-arch hailo10h --model-script g3q_qft_cpu_trial.alls \
     --calib-set-path '"$CAL"' --output-har-path opt_rothad_D256L1_qft_cpu_trial.har > optimize_qft_cpu_trial.log 2> optimize_qft_cpu_trial.time
  echo "$(date +%T) rc=$? sec=$(( $(date +%s) - t0 )) maxrss_kb=$(grep "Maximum resident" optimize_qft_cpu_trial.time | awk "{print \$NF}")"
' 2>&1 | tee -a $O/cpu_trial_info.txt
cp optimize_qft_cpu_trial.log optimize_qft_cpu_trial.time $O/ 2>/dev/null
echo "QFT CPU TRIAL DONE $(date +%T)" | tee -a $O/cpu_trial_info.txt
