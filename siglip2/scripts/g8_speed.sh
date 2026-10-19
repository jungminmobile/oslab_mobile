#!/usr/bin/env bash
# G8 지연·메모리·시간 분해 (비공식, 이 PC): g7_speed.sh와 같고 모델이 4개(FP32·S-all·S-const·D-m2rot), 끝에 s2_profile_ort.py 4스레드(S-all·S-const·D-m2rot).
# 게이트 (세션·스레드마다 측정 직전): WSL 안 다른 python·hailo 프로세스 0개 + /proc/loadavg 1분 평균 < 2. 안 맞으면 5분마다 다시 확인 (상한 없음).
# 대기 횟수·시각은 gate.log에 남긴다. Windows 쪽 부하는 확인하지 못한다 (interop 꺼짐). 측정 뒤에도 같은 값을 기록해 도중 끼어든 작업을 표시한다.
# 사용 (siglip2/):  bash scripts/g8_speed.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
R=../results/siglip2/G8_constb
LOG=$R/speed/gate.log; mkdir -p $R/speed
M="fp32=siglip2_image_fp32.onnx S_all=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx S_const=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_cbo_all.onnx D_m2rot=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_dynamic_ws8_pc_ops-MatMul_rr.onnx"
P="S_all=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx S_const=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_cbo_all.onnx D_m2rot=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_dynamic_ws8_pc_ops-MatMul_rr.onnx"
others() { timeout 30 ps -eo pid,cmd | grep -E "venvs/siglip2/bin/python|hailo10_venv|hailortcli|hailo_tools" | grep -v grep | wc -l; }
load1() { cut -d' ' -f1 /proc/loadavg; }
waits=0
echo "$(date '+%F %T') start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l) gate=WSL(others=0, load1<2, 5분 간격) windows=확인못함(interop 꺼짐)" | tee -a $LOG
for s in 1 2 3; do
  for t in 1 2 4; do
    while true; do
      n=$(others); l=$(load1)
      if [ "$n" -eq 0 ] && awk "BEGIN{exit !($l < 2)}"; then break; fi
      waits=$((waits+1)); echo "$(date '+%F %T') s=$s t=$t 대기#$waits others=$n load1=$l" >> $LOG; sleep 300
    done
    echo "$(date '+%F %T') s=$s t=$t pre others=$n load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOG
    RN50_RESULTS=$R/speed timeout 3600 "$PY" -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t --models $M 2>&1 | grep -vE 'HF_TOKEN|Warning'
    echo "$(date '+%F %T') s=$s t=$t done post others=$(others) load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOG
  done
done
while true; do
  n=$(others); l=$(load1)
  if [ "$n" -eq 0 ] && awk "BEGIN{exit !($l < 2)}"; then break; fi
  waits=$((waits+1)); echo "$(date '+%F %T') mem 대기#$waits others=$n load1=$l" >> $LOG; sleep 300
done
echo "$(date '+%F %T') peak memory pre others=$n load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOG
RN50_RESULTS=$R/model_stats timeout 3600 "$PY" -u s2_model_stats.py --models $M 2>&1 | grep -vE 'HF_TOKEN|Warning'
while true; do
  n=$(others); l=$(load1)
  if [ "$n" -eq 0 ] && awk "BEGIN{exit !($l < 2)}"; then break; fi
  waits=$((waits+1)); echo "$(date '+%F %T') profile 대기#$waits others=$n load1=$l" >> $LOG; sleep 300
done
echo "$(date '+%F %T') profile pre others=$n load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOG
RN50_RESULTS=$R/profile timeout 3600 "$PY" -u s2_profile_ort.py --threads 4 --models $P 2>&1 | grep -vE 'HF_TOKEN|Warning'
echo "$(date '+%F %T') profile done post others=$(others) load=$(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOG
echo "$(date '+%F %T') G8_SPEED_DONE waits=$waits" | tee -a $LOG
