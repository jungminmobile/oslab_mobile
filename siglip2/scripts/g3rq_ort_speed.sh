#!/usr/bin/env bash
# G3RQ-ORT d단계 지연 측정: 3세션 × 스레드 1·2·4 (비공식) + peak 메모리. g3q_speed_e3.sh와 측정 명령은 같고, 부하 게이트만 WSL 안에서 확인한다.
# 게이트 (세션·스레드마다 측정 직전): 다른 python·hailo 프로세스 0개 + /proc/loadavg 1분 평균 < 2. 맞을 때까지 1분마다 다시 확인 (상한 없음).
# WSL interop이 꺼져 powershell.exe를 못 써서 Windows 쪽 부하는 확인하지 못한다 (기록). 측정 뒤에도 같은 값을 기록해 도중 끼어든 작업을 표시한다.
# 사용 (siglip2/):  bash scripts/g3rq_ort_speed.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
R=../results/siglip2/G3RQ_m2rot/ort
LOG=$R/speed/gate.log; mkdir -p $R/speed
M="fp32=siglip2_image_fp32.onnx R0_5eee=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx M2rot_ort=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx"
others() { ps -eo pid,cmd | grep -E "venvs/siglip2/bin/python|hailo10_venv|hailortcli" | grep -v grep | wc -l; }
load1() { cut -d' ' -f1 /proc/loadavg; }
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l) gate=WSL(others=0, load1<2) windows=확인못함(interop 꺼짐)" | tee -a $LOG
for s in 1 2 3; do
  for t in 1 2 4; do
    while true; do
      n=$(others); l=$(load1)
      if [ "$n" -eq 0 ] && awk "BEGIN{exit !($l < 2)}"; then break; fi
      echo "$(date +%T) s=$s t=$t 대기 others=$n load1=$l" >> $LOG; sleep 60
    done
    echo "$(date +%T) s=$s t=$t pre others=$n load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOG
    RN50_RESULTS=$R/speed "$PY" -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t --models $M 2>&1 | grep -vE 'HF_TOKEN|Warning'
    echo "$(date +%T) s=$s t=$t done post others=$(others) load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOG
  done
done
echo "$(date +%T) peak memory" | tee -a $LOG
RN50_RESULTS=$R/model_stats "$PY" -u s2_model_stats.py --models $M 2>&1 | grep -vE 'HF_TOKEN|Warning'
echo "G3RQ_ORT_SPEED_DONE $(date +%T)" | tee -a $LOG
