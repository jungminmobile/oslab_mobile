#!/usr/bin/env bash
# speed_w4_ab.sh — (10/16 07:50) D-038 (나) 판정 보충: 4비트 A·B는 아직 지연을 안 쟀다. C·D가 U8+rr보다 약 2배 느려서,
#  B(MLP만 4비트)가 기준을 만족할 수 있는지 4스레드 1세션으로 확인한다 (needs idle PC, 잠금 안 CPU 확인).
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
[ -f /d/mokerton/locks/siglip2.STOP ] && { echo "STOP 파일 — 멈춤"; exit 3; }
c=$("$PY" -c 'import psutil; print(round(psutil.cpu_percent(interval=30)))')
echo "in-lock cpu30s=$c%"
[ "$c" -ge 25 ] && { echo BUSY; exit 75; }
U=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16
RN50_RESULTS=../results/siglip2/G3_w4/speed_AB "$PY" -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads 4 \
  --models u8rr=${U}_rr_fc2L2.onnx w4A=${U}_rr_w4b32a4_w4A_fc2L2.onnx w4B=${U}_rr_w4b32a4_w4B_fc2L2.onnx
