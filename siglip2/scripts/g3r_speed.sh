#!/usr/bin/env bash
# G3R G2 지연 3세션 (스레드 1·2·4, 비공식) + peak 메모리. WSL Ubuntu-22.04용.
# 측정 직전 Windows 전체 CPU(10초 평균)·다른 WSL 배포판·WSL loadavg를 기록하고, THRESH% 이상이면 2분 뒤 다시 (최대 15회, 넘으면 busy로 표시하고 측정).
# 사용 (siglip2/ 폴더):  bash scripts/g3r_speed.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"; THRESH=${THRESH:-25}
R=../results/siglip2/G3R_regcache
LOG=$R/speed/gate.log; mkdir -p $R/speed
M="fp32=siglip2_image_fp32.onnx R0_fc2L2=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_idcheck_fc2L2.onnx R1_int8=siglip2_image_int8_rc55at5-13at9_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx"
wincpu() { timeout 60 powershell.exe -NoProfile -Command '$s=0;1..5|%{$s+=(Get-CimInstance Win32_Processor|Measure-Object LoadPercentage -Average).Average;Start-Sleep 2};[math]::Round($s/5)' 2>/dev/null | tr -d '\r'; }
wintop() { timeout 60 powershell.exe -NoProfile -Command 'Get-Counter "\Process(*)\% Processor Time" -SampleInterval 2 -MaxSamples 1 | % CounterSamples | ? {$_.InstanceName -notin "_total","idle"} | sort CookedValue -Desc | select -First 5 | % {"{0}={1:N0}" -f $_.InstanceName,($_.CookedValue/'$(nproc)')}' 2>/dev/null | tr -d '\r' | paste -sd' '; }
distros() { timeout 20 wsl.exe -l --running -q 2>/dev/null | tr -d '\r\0' | paste -sd','; }
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l) thresh=$THRESH" | tee -a $LOG
for s in 1 2 3; do
  for t in 1 2 4; do
    tries=0
    while true; do
      c=$(wincpu); echo "$(date +%T) s=$s t=$t pre win_cpu10s=${c}% wsl_load=$(cut -d' ' -f1-3 /proc/loadavg) running=[$(distros)] top=[$(wintop)]" | tee -a $LOG
      if [ "${c:-100}" -lt "$THRESH" ]; then busy=0; break; fi
      tries=$((tries+1)); if [ $tries -ge 15 ]; then busy=1; echo "$(date +%T) s=$s t=$t BUSY 측정 강행 (기록)" | tee -a $LOG; break; fi
      sleep 120
    done
    RN50_RESULTS=$R/speed "$PY" -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t --models $M 2>&1 | grep -vE 'HF_TOKEN|Warning'
    echo "$(date +%T) s=$s t=$t done busy=$busy post win_cpu10s=$(wincpu)%" | tee -a $LOG
  done
done
echo "$(date +%T) peak memory" | tee -a $LOG
RN50_RESULTS=$R/model_stats "$PY" -u s2_model_stats.py --models $M 2>&1 | grep -vE 'HF_TOKEN|Warning'
echo "G3R_SPEED_DONE $(date +%T)" | tee -a $LOG
