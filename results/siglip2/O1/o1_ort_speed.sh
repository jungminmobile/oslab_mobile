#!/usr/bin/env bash
# O1 ORT CPU 지연·메모리 (G6b g6b_ort_speed.sh 복사, 결과 폴더만 바꿈) (비공식, 이 PC): G8 g8_speed.sh·G3RQ-ORT g3rq_ort_speed.sh와 같은 게이트·명령, 측정 스크립트만 s2_g6b_bench.py (크기 다른 모델).
# 게이트 (세션·스레드마다 측정 직전): WSL 안 다른 python·hailo 프로세스 0개 + DFC 없음 + /proc/loadavg 1분 평균 < 2. 안 맞으면 5분마다 다시 (상한 없음).
# Windows 쪽 부하는 확인 못 함 (interop 꺼짐). 측정 뒤에도 같은 값을 기록해 도중 끼어든 작업을 표시한다.
# 사용 (siglip2/):  M="b16_fp32=…:224 …" bash ../results/siglip2/O1/o1_ort_speed.sh
set -uo pipefail
cd "$(dirname "$0")/../../../siglip2"
source ~/siglip2_env.sh
export OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
R=../results/siglip2/O1
LOG=$R/speed/gate.log; mkdir -p $R/speed
M=${M:?}; SZ=${SZ:?}
others() { timeout 30 ps -eo pid,cmd | grep -E "venvs/siglip2/bin/python|hailo10_venv|hailortcli|hailo_tools|hailo (parser|optimize|compiler)" | grep -v grep | wc -l; }
load1() { cut -d' ' -f1 /proc/loadavg; }
waits=0
echo "$(date '+%F %T') start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l) gate=WSL(others=0, load1<2, 5분 간격) windows=확인못함(interop 꺼짐)" | tee -a $LOG
gate() {
  while true; do
    n=$(others); l=$(load1)
    if [ "$n" -eq 0 ] && awk "BEGIN{exit !($l < 2)}"; then break; fi
    waits=$((waits+1)); echo "$(date '+%F %T') $1 대기#$waits others=$n load1=$l" >> $LOG; sleep 300
  done
}
for s in 1 2 3; do
  for t in 1 2 4; do
    gate "s=$s t=$t"
    echo "$(date '+%F %T') s=$s t=$t pre others=$n load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOG
    RN50_RESULTS=$R/speed timeout 3600 "$PY" -u s2_g6b_bench.py --warmup 20 --n 200 --repeats 3 --threads $t --models $M 2>&1 | grep -vE 'HF_TOKEN|Warning'
    echo "$(date '+%F %T') s=$s t=$t done post others=$(others) load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOG
  done
done
gate "mem"
echo "$(date '+%F %T') peak memory pre others=$n load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOG
RN50_RESULTS=$R/model_stats timeout 3600 "$PY" -u s2_model_stats.py --models $(echo $M | sed -E 's/:[0-9]+//g') --sizes $SZ 2>&1 | grep -vE 'HF_TOKEN|Warning'
echo "$(date '+%F %T') O1_SPEED_DONE waits=$waits" | tee -a $LOG
