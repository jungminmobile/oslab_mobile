#!/usr/bin/env bash
# O1 4스레드 프로파일 (워밍업 10 + 30장): 모델별 기준과 지정 후보. 지연 측정과 같은 게이트.
# 사용 (아무 곳):  bash results/siglip2/O1/o1_profile.sh b16oab b32oab   (각 모델의 base는 자동 포함)
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
cd "$D/../../../siglip2"
source ~/siglip2_env.sh; source $D/o1_models.sh
export OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
R=../results/siglip2/O1; LOG=$R/speed/gate.log
others() { timeout 30 ps -eo pid,cmd | grep -E "venvs/siglip2/bin/python|hailo10_venv|hailortcli|hailo_tools|hailo (parser|optimize|compiler)" | grep -v grep | wc -l; }
for c in "$@"; do
  b=${c:0:3}
  while true; do n=$(others); l=$(cut -d' ' -f1 /proc/loadavg)
    [ "$n" -eq 0 ] && awk "BEGIN{exit !($l < 2)}" && break
    echo "$(date '+%F %T') profile $b 대기 others=$n load1=$l" >> $LOG; sleep 300; done
  echo "$(date '+%F %T') profile $b pre others=$n load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOG
  RN50_RESULTS=$R/profile timeout 3600 $SIGLIP2_PYTHON -u s2_profile_ort.py --threads 4 --zoo-size $(size_of $c) \
    --models ${b}base=$SIGLIP2_ARTIFACTS/onnx/${F[${b}base]} $c=$SIGLIP2_ARTIFACTS/onnx/${F[$c]} 2>&1 | grep -vE 'HF_TOKEN|Warning'
done
echo "O1_PROFILE_DONE $(date +%T)"
