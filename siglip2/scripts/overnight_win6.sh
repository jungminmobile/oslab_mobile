#!/usr/bin/env bash
# overnight_win6.sh — 독립 검토 지적 반영 후 원인 분석 재실행 (64장, 공용 잠금)
#  1) s2_why_massive.py: high-norm 토큰 위치 전부 기록, attention 집중을 이전 블록 토큰 기준으로도 (why_massive.jsonl 3번째 줄)
#  2) s2_why_writers.py: 분위수 범위를 정확히(ORT Percentile 정의), ORT 실제 정수 모델 범위로도 개입, 위치별 값 (why_writers.jsonl 2번째 줄)
# 첫 실행의 pq/clip 행은 1/7 간격 표본 버그로 무효 (196 = 28×7 → 모든 사진에서 같은 위치만 뽑힘).
set -uo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1
PY="$SIGLIP2_PYTHON"
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
ART="$SIGLIP2_ARTIFACTS"
R=../results/siglip2
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=all -- . | tr '\n' ' ')]"
RN50_RESULTS=$R/G3_why "${LOCK[@]}" "SigLIP2 why-massive v2 (positions, prev-block attention)" -- \
    "$PY" -u s2_why_massive.py --n 32 > $ART/logs/overnight_why_v2.log 2>&1
echo "$(date +%T) why v2 rc=$?"
RN50_RESULTS=$R/G3_why "${LOCK[@]}" "SigLIP2 why-writers v2 (exact pct, ORT ranges)" -- \
    "$PY" -u s2_why_writers.py --n 32 --ort-model siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16_mm.onnx \
    > $ART/logs/overnight_writers_v2.log 2>&1
echo "$(date +%T) writers v2 rc=$?"
echo "OVERNIGHT_WIN6 DONE $(date +%T)"
