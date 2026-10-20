#!/usr/bin/env bash
# overnight_1015.sh — 10/15 새벽 자동 진행 (쭈리 수면 중)
#  1) qkv_precision.sh가 끝나면 세 번째 지연 측정 (PC가 조용한 밤): FP32·INT8 전부·fc2L7·L4·L2·U8·reduce_range·U8U8,
#     스레드 1·2·4, PRD 5절 조건. 잠금 안에서 전체 CPU 30초 25% 미만일 때만 (아니면 3분 뒤 다시). job 이름에 "needs idle PC".
#  2) 그다음 WSL: C 컴파일 결과 정리(프로파일러)와 DFC 최적화 수준 1 (hailo/g4_more3_wsl.sh).
#     개인연구 세션 요청: SigLIP2 Windows 무거운 작업과 WSL을 동시에 띄우지 않는다 → Windows 작업이 모두 끝난 뒤 띄운다.
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
ART="$SIGLIP2_ARTIFACTS"
THRESH=25
R=../results/siglip2/G3_cumulative
P=siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16
U=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16
until grep -q "QKV_PRECISION DONE" $ART/logs/qkv_precision.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=all -- . | tr '\n' ' ')]"

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
M="fp32=siglip2_image_fp32.onnx int8_all=siglip2_image_int8_minmax_highres256_s0_s8_all.onnx fc2L7=${P}_mm_nofc2L.onnx fc2L4=${P}_fc2L4.onnx fc2L2=${P}_fc2L2.onnx fc2L2_u8=${U}_fc2L2.onnx fc2L2_u8rr=${U}_rr_fc2L2.onnx fc2L2_u8u8=${U}_wu8_fc2L2.onnx"
for t in 1 2 4; do
  timed "SigLIP2 latency night threads=$t (needs idle PC)" "RN50_RESULTS=$R/speed_night '$PY' -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t --models $M"
done
echo "$(date +%T) latency night done"

wsl.exe -d Ubuntu-22.04 -e bash -lc "tr -d '\r' < /mnt/d/mokerton/oslab_mobile/siglip2/hailo/g4_more3_wsl.sh > ~/g4_more3_wsl.sh && cd ~ && bash ~/g4_more3_wsl.sh > /mnt/d/mokerton/artifacts/siglip2/logs/g4_more3_wsl.log 2>&1"
rc=$?; echo "$(date +%T) WSL chain rc=$rc $(tail -1 $ART/logs/g4_more3_wsl.log)"
echo "OVERNIGHT_1015 DONE $(date +%T)"
