#!/usr/bin/env bash
# speed_w4.sh — (10/16, D-038 (나) 판정용) 4비트 C·D와 U8+rr 최종의 지연을 같은 측정 안에서 3회 (세션 a·b·c, 스레드 1·2·4)
#  판정 기준(결과 전에 고정): 4스레드 지연이 U8+rr보다 느리지 않음 (3회 중앙값). 잠금 안 CPU 30초 25% 미만일 때만, STOP 파일이면 멈춤.
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"; STOP=/d/mokerton/locks/siglip2.STOP
R=../results/siglip2/G3_w4
U=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=no -- . | tr '\n' ' ')]"
timed() {
  local job=$1; shift
  while true; do
    [ -f $STOP ] && { echo "$(date +%T) STOP 파일 — 멈춤"; exit 3; }
    "$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "$job" -- \
      "C:/Program Files/Git/bin/bash.exe" -c "c=\$('$PY' -c 'import psutil; print(round(psutil.cpu_percent(interval=30)))'); echo \"in-lock cpu30s=\$c%\"; \
        if [ \"\$c\" -ge 25 ]; then echo BUSY_RETRY; exit 75; fi; $*"
    local rc=$?; echo "$(date +%T) $job rc=$rc"
    [ $rc -eq 75 ] && { sleep 180; continue; }
    return $rc
  done
}
M="fp32=siglip2_image_fp32.onnx u8rr=${U}_rr_fc2L2.onnx w4C=${U}_rr_w4b32a4_w4C_fc2L2.onnx w4D=${U}_rr_w4b32a4_w4D_fc2L2.onnx"
for s in a b c; do for t in 1 2 4; do
  timed "SigLIP2 latency w4 $s threads=$t (needs idle PC)" "RN50_RESULTS=$R/speed_$s '$PY' -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t --models $M"
done; done
echo "SPEED_W4 DONE $(date +%T)"
