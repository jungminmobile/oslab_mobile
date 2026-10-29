#!/usr/bin/env bash
# G3Q Q3 지연 3세션 (스레드 1·2·4, 비공식) + peak 메모리. WSL Ubuntu-22.04용. scripts/g3r_speed.sh(siglip2/g3r-regcache)와 같은 방법, 모델·결과 폴더만 다름.
# 측정 직전 다른 프로세스 CPU 합(Windows, 10초, WSL 자신 vmmemwsl 제외)이 PROC_THRESH% 이상이면 2분 뒤 다시 (최대 15회, 넘으면 busy로 표시하고 측정).
# Windows 전체 CPU(WMI LoadPercentage)·다른 WSL 배포판·WSL loadavg·상위 프로세스는 기록만 한다
# (10/18: 285K에서 LoadPercentage가 프로세스 합 약 3.5%일 때도 28~36%로 나와 게이트로 쓰지 않게 바꿈).
# 사용 (siglip2/ 폴더):  bash scripts/g3q_speed.sh     이어서 재기: START_S=2 START_T=4 bash scripts/g3q_speed.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"; PROC_THRESH=${PROC_THRESH:-10}; START_S=${START_S:-1}; START_T=${START_T:-1}
R=../results/siglip2/G3Q_rotate
LOG=$R/speed/gate.log; mkdir -p $R/speed
M="fp32=siglip2_image_fp32.onnx R0_fc2L2=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_idcheck_fc2L2.onnx rothad_fc2L2=siglip2_image_int8_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx"
wincpu() { timeout 60 powershell.exe -NoProfile -Command '$s=0;1..5|%{$s+=(Get-CimInstance Win32_Processor|Measure-Object LoadPercentage -Average).Average;Start-Sleep 2};[math]::Round($s/5)' 2>/dev/null | tr -d '\r'; }
wintop() { timeout 60 powershell.exe -NoProfile -Command 'Get-Counter "\Process(*)\% Processor Time" -SampleInterval 2 -MaxSamples 1 | % CounterSamples | ? {$_.InstanceName -notin "_total","idle"} | sort CookedValue -Desc | select -First 5 | % {"{0}={1:N0}" -f $_.InstanceName,($_.CookedValue/'$(nproc)')}' 2>/dev/null | tr -d '\r' | paste -sd' '; }
procsum() { timeout 60 powershell.exe -NoProfile -Command '$a=Get-Process|select Id,CPU; Start-Sleep 10; $s=0; Get-Process|?{$_.Name -notin "vmmemwsl","vmmem","Idle"}|%{$p=$_; $o=$a|? Id -eq $p.Id; if($o -and $p.CPU -and $o.CPU){$s+=$p.CPU-$o.CPU}}; [math]::Round($s/10/[Environment]::ProcessorCount*100)' 2>/dev/null | tr -d '\r'; }
distros() { timeout 20 wsl.exe -l --running -q 2>/dev/null | tr -d '\r\0' | paste -sd','; }
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l) proc_thresh=$PROC_THRESH start=s$START_S/t$START_T" | tee -a $LOG
for s in 1 2 3; do
  for t in 1 2 4; do
    if [ $s -lt $START_S ] || { [ $s -eq $START_S ] && [ $t -lt $START_T ]; }; then continue; fi
    tries=0
    while true; do
      c=$(wincpu); pc=$(procsum); echo "$(date +%T) s=$s t=$t pre proc_sum10s=${pc}% win_cpu10s=${c}% wsl_load=$(cut -d' ' -f1-3 /proc/loadavg) running=[$(distros)] top=[$(wintop)]" | tee -a $LOG
      if [ "${pc:-100}" -lt "$PROC_THRESH" ]; then busy=0; break; fi
      tries=$((tries+1)); if [ $tries -ge 15 ]; then busy=1; echo "$(date +%T) s=$s t=$t BUSY 측정 강행 (기록)" | tee -a $LOG; break; fi
      sleep 120
    done
    RN50_RESULTS=$R/speed "$PY" -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t --models $M 2>&1 | grep -vE 'HF_TOKEN|Warning'
    echo "$(date +%T) s=$s t=$t done busy=$busy post proc_sum10s=$(procsum)% win_cpu10s=$(wincpu)%" | tee -a $LOG
  done
done
echo "$(date +%T) peak memory" | tee -a $LOG
RN50_RESULTS=$R/model_stats "$PY" -u s2_model_stats.py --models $M 2>&1 | grep -vE 'HF_TOKEN|Warning'
echo "G3Q_SPEED_DONE $(date +%T)" | tee -a $LOG
