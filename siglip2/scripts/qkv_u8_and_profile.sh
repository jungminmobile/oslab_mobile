#!/usr/bin/env bash
# qkv_u8_and_profile.sh — (10/14 쭈리 요청) S8 4스레드 지연 원인 분석 + QKV 투영만 U8 옵션 실험 (attention만 U8은 효과 없음: 합쳐지지 않던 것은 QKV였음)
#  1) fc2L2 + --u8-match /attn/qkv/,/attn_pool/kv/ 양자화 (fc2L2와 같은 보정·인자, --dump-ops로 정수 MatMul 수 확인)          [잠금]
#  2) 스크리닝: dev 6종 × 200장, 정수 커널 (fc2L2 S8·U8과 같은 형식)                                 [잠금]
#  3) dev 전체 + 짝 부트스트랩: fc2L2 S8 · U8 · attnU8 (사진별 예측, 쌍별 차이)                          [잠금]
#  4) 시간 측정 (잠금 + PC 조용할 때만, speed_when_idle.sh와 같은 규칙):
#     - ORT 연산별 프로파일 (FP32, fc2L2 S8, U8, attnU8; 스레드 1·4)
#     - 지연 (같은 4개 모델, 스레드 1·2·4, PRD 5절 조건)
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
ART="$SIGLIP2_ARTIFACTS"
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
THRESH=${THRESH:-25}
R=../results/siglip2/G3_cumulative
P=siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16
S8=${P}_fc2L2.onnx
U8=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_fc2L2.onnx
A8=siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16_u8m-attn.qkv-attn_pool.kv_fc2L2.onnx
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=all -- . | tr '\n' ' ')]"

RN50_RESULTS=$R "${LOCK[@]}" "SigLIP2 quantize fc2L2 qkvU8" -- "$PY" -u s2_quantize.py --calib-source highres --calib-n 256 \
    --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --overwrite --dump-ops --tag fc2L2 --u8-match /attn/qkv/,/attn_pool/kv/ \
    --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/" > $ART/logs/qkvu8_quantize.log 2>&1
rc=$?; echo "$(date +%T) quantize qkvU8 rc=$rc $(grep -a -o '"int_kernel_ops": {[^}]*}' $ART/logs/qkvu8_quantize.log | tail -1)"
[ -f "$ART/onnx/$A8" ] || { echo "QUANT FAILED"; exit 1; }

RN50_RESULTS=$R "${LOCK[@]}" "SigLIP2 screen qkvU8" -- "$PY" -u s2_screen.py --k 200 --tag cumulative --models $A8 \
    > $ART/logs/qkvu8_screen.log 2>&1
rc=$?; echo "$(date +%T) screen rc=$rc $(grep -a '"hr_cos"' $R/screen.jsonl | tail -1 | grep -o '"hr_cos": [0-9.]*, "hr_dtop1": [-0-9.]*')"

RN50_RESULTS=$R/full_qkvu8 "${LOCK[@]}" "SigLIP2 paired_ci S8/U8/qkvU8" -- "$PY" -u s2_paired_ci.py --models $S8 $U8 $A8 \
    > $ART/logs/qkvu8_paired_ci.log 2>&1
rc=$?; echo "$(date +%T) paired_ci rc=$rc"

timed() {  # $1 job name, 나머지: 명령 — 잠금 안에서 CPU 30초 확인 후 실행, 바쁘면 3분 뒤 다시
  local job=$1; shift
  while true; do
    "$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "$job" -- \
      "C:/Program Files/Git/bin/bash.exe" -c "c=\$('$PY' -c 'import psutil; print(round(psutil.cpu_percent(interval=30)))'); echo \"in-lock cpu30s=\$c%\"; \
        if [ \"\$c\" -ge $THRESH ]; then echo BUSY_RETRY; exit 75; fi; $*"
    rc=$?; echo "$(date +%T) $job rc=$rc"
    [ $rc -eq 75 ] && { sleep 180; continue; }
    return $rc
  done
}
timed "SigLIP2 ORT profile (needs idle PC)" "RN50_RESULTS=$R/profile '$PY' -u s2_profile_ort.py --threads 1 4 --models fp32=siglip2_image_fp32.onnx fc2L2=$S8 fc2L2_u8=$U8 fc2L2_qkvu8=$A8"
for t in 1 2 4; do
  timed "SigLIP2 latency qkvU8 threads=$t (needs idle PC)" "RN50_RESULTS=$R/speed_qkvu8 '$PY' -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t --models fp32=siglip2_image_fp32.onnx fc2L2=$S8 fc2L2_u8=$U8 fc2L2_qkvu8=$A8"
done
echo "QKVU8 DONE $(date +%T)"
