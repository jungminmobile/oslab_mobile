#!/usr/bin/env bash
# speed_1016.sh — (10/16 새벽, 쭈리: 고성능 모드 유지 확인) U8 + reduce_range 지연을 2회 더 재서 PRD 5절 "3회 측정"을 채운다 (D-030 확정 근거)
#  비교 대상 FP32·S8·U8도 같은 세션에서 다시 잰다 (비교는 같은 측정 안의 비율로). 측정 2세션(a, b), 각 세션은 스레드 1·2·4.
#  순서: C256L1 dev 전체(WSL) → portable_eval 검증 → WSL Ubuntu-22.04 종료(메모리 반환) → 지연. 잠금 안에서 CPU 30초 25% 미만일 때만.
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
ART="$SIGLIP2_ARTIFACTS"
THRESH=25
R=../results/siglip2/G3_cumulative
P=siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16
U=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16
until grep -q "G4_MORE7 DONE" $ART/logs/g4_more7_wsl.log 2>/dev/null; do sleep 60; done
until [ -f ../results/siglip2/G3_xplat/x86_i7-1165G7/result.json ] || grep -q "Traceback" $ART/logs/portable_eval_x86.log 2>/dev/null; do sleep 60; done
sleep 60
wsl.exe -t Ubuntu-22.04; echo "$(date +%T) WSL Ubuntu-22.04 terminated rc=$?"
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=no -- . | tr '\n' ' ')]"
timed() {  # $1 job name, 나머지: 명령 — 잠금 안에서 CPU 30초 확인 후 실행, 바쁘면 3분 뒤 다시
  local job=$1; shift
  while true; do
    "$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "$job" -- \
      "C:/Program Files/Git/bin/bash.exe" -c "c=\$('$PY' -c 'import psutil; print(round(psutil.cpu_percent(interval=30)))'); echo \"in-lock cpu30s=\$c%\"; \
        if [ \"\$c\" -ge $THRESH ]; then echo BUSY_RETRY; exit 75; fi; $*"
    local rc=$?; echo "$(date +%T) $job rc=$rc"
    [ $rc -eq 75 ] && { sleep 180; continue; }
    return $rc
  done
}
M="fp32=siglip2_image_fp32.onnx fc2L2=${P}_fc2L2.onnx fc2L2_u8=${U}_fc2L2.onnx fc2L2_u8rr=${U}_rr_fc2L2.onnx"
for s in a b; do
  for t in 1 2 4; do
    timed "SigLIP2 latency 1016$s threads=$t (needs idle PC)" "RN50_RESULTS=$R/speed_1016$s '$PY' -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t --models $M"
  done
done
echo "SPEED_1016 DONE $(date +%T)"
