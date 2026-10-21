#!/usr/bin/env bash
# O1 정확도: dev 전체(G6 Zoo 전처리 캐시, 고화질 5종 + CIFAR A), G6b와 같은 eval 코드 s2_g6b_ort.py. nice -n 10, ORT 4스레드.
# 시작 전: 다른 세션이 지연 측정 중이면(s2_bench_speed·s2_g6b_bench·s2_model_stats·s2_profile_ort·hailortcli) 5분 간격으로 기다리고 기록.
# 사용 (아무 곳):  bash results/siglip2/O1/o1_eval.sh
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
cd "$D/../../../siglip2"
source ~/siglip2_env.sh
source $D/o1_models.sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
R=../results/siglip2/O1
WLOG=$R/eval_wait.log
while true; do
  n=$(timeout 30 ps -eo cmd | grep -E "s2_bench_speed|s2_g6b_bench|s2_model_stats|s2_profile_ort|hailortcli" | grep -v grep | wc -l)
  [ "$n" -eq 0 ] && break
  echo "$(date '+%F %T') 대기 (다른 세션 지연 측정 프로세스 $n개)" | tee -a $WLOG; sleep 300
done
echo "$(date '+%F %T') 시작 (측정 프로세스 0) commit=$(git rev-parse --short HEAD)" | tee -a $WLOG
M=""; for k in $ORDER; do
  ref=${k:0:3}; M="$M o1$k=$SIGLIP2_ARTIFACTS/onnx/${F[$k]}:$(size_of $k):$ref"
done
P=""; for b in b16 b32; do
  for c in base oa ob oab; do P="$P o1$b$c-o1${b}fp"; done
  for c in oa ob oab; do P="$P o1$b$c-o1${b}base"; done
done
RN50_RESULTS=$R/accuracy nice -n 10 $SIGLIP2_PYTHON -u s2_g6b_ort.py --threads 4 --models $M --pairs $P 2>&1 | grep -vE 'HF_TOKEN|Warning'
echo "O1_EVAL_DONE $(date +%T)"
