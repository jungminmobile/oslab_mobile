#!/usr/bin/env bash
# speed_when_idle.sh — 지연 재측정(스레드 1·2·4)을 PC가 정말 한가할 때만 (10/14 14:07 재측정이 Windows Update로 오염되어 중단)
# 규칙: ① 시스템 전체 CPU 1분 평균이 THRESH% 미만인 상태가 3분 연속일 때만 잠금 대기표를 받는다
#       ② 잠금을 잡은 뒤에도 30초 동안 다시 재서 THRESH% 이상이면(다른 SigLIP2 작업 제외 부하) 바로 놓고 처음으로 돌아간다
#       ③ 스레드 설정마다 따로 잠금 (1 → 2 → 4). 측정 직후 그 사이 CPU를 기록
# 기준치: 이 PC 배경 부하(Claude 앱·Defender·Intel 서비스)는 10~15%. THRESH=20.
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
THRESH=${THRESH:-20}
R=../results/siglip2/G3_cumulative
P=siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16
LOG=$SIGLIP2_ARTIFACTS/logs/speed_when_idle_v2.log
cpu() { "$PY" -c "import psutil; print(round(psutil.cpu_percent(interval=$1)))"; }
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) thresh=$THRESH"
for t in 1 2 4; do
  while true; do
    ok=0
    while [ $ok -lt 3 ]; do
      c=$(cpu 60); if [ "$c" -lt "$THRESH" ]; then ok=$((ok+1)); else ok=0; fi
      echo "$(date +%T) t=$t idle-check cpu1min=$c% streak=$ok" >> $LOG
    done
    # 잠금 안에서: 30초 다시 재고, 조용하면 측정
    RN50_RESULTS=$R/speed "$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "SigLIP2 latency threads=$t (idle-gated)" -- \
      "C:/Program Files/Git/bin/bash.exe" -c "c=\$('$PY' -c 'import psutil; print(round(psutil.cpu_percent(interval=30)))'); echo \"in-lock cpu30s=\$c%\"; \
        if [ \"\$c\" -ge $THRESH ]; then echo BUSY_RETRY; exit 75; fi; \
        '$PY' -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t \
          --models fp32=siglip2_image_fp32.onnx int8_all=siglip2_image_int8_minmax_highres256_s0_s8_all.onnx \
                   fc2L7=${P}_mm_nofc2L.onnx fc2L4=${P}_fc2L4.onnx fc2L2=${P}_fc2L2.onnx \
                   fc2L2_u8=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_fc2L2.onnx"
    rc=$?
    echo "$(date +%T) t=$t rc=$rc after-cpu60s=$(cpu 5)%"
    [ $rc -eq 75 ] && { sleep 120; continue; }
    break
  done
done
echo "SPEED_IDLE_DONE $(date +%T)"
