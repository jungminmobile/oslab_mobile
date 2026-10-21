#!/usr/bin/env bash
# O2: 정확도(dev 전체, O1과 같은 s2_g6b_ort.py) → 지연(oab·uoab 번갈아, 스레드 1·2·4 × 3세션) + peak 메모리 (O1 o1_ort_speed.sh와 같은 게이트)
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
cd "$D/../../../siglip2"
source ~/siglip2_env.sh; source $D/o2_models.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
R=../results/siglip2/O2
while true; do
  n=$(timeout 30 ps -eo cmd | grep -E "s2_bench_speed|s2_g6b_bench|s2_model_stats|s2_profile_ort|hailortcli" | grep -v grep | wc -l)
  [ "$n" -eq 0 ] && break
  echo "$(date '+%F %T') 대기 (다른 세션 지연 측정 프로세스 $n개)" | tee -a $R/eval_wait.log; sleep 300
done
echo "$(date '+%F %T') 정확도 시작 (측정 프로세스 0) commit=$(git rev-parse --short HEAD)" | tee -a $R/eval_wait.log
M=""; for k in $ORDER; do M="$M o2$k=$SIGLIP2_ARTIFACTS/onnx/${F[$k]}:224:b16"; done
P="o2uoab-o2oab o2ubase-o2base"; for k in ofp ufp base oab ubase uoab; do P="$P o2$k-o2fp"; done
RN50_RESULTS=$R/accuracy nice -n 10 $SIGLIP2_PYTHON -u s2_g6b_ort.py --threads 4 --models $M --pairs $P 2>&1 | grep -vE 'HF_TOKEN|Warning'
echo "O2_EVAL_DONE $(date +%T)"
SM="oab=${F[oab]}:224 uoab=${F[uoab]}:224"
sed -e 's#R=../results/siglip2/O1#R=../results/siglip2/O2#' -e 's#O1_SPEED_DONE#O2_SPEED_DONE#' ../results/siglip2/O1/o1_ort_speed.sh > $D/o2_ort_speed.sh
M="$SM" SZ="oab=224 uoab=224" bash $D/o2_ort_speed.sh
