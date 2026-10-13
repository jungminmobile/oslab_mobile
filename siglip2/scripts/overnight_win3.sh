#!/usr/bin/env bash
# overnight_win3.sh — 원인 분석 2단계(s2_why_writers.py: 거대 값을 쓰는 층 + 개입 실험) 본 실행 (사진 64장)
# overnight_win2.sh가 끝난 뒤 시작하고 공용 잠금을 잡는다 (메모리 약 4 GB → 무거운 작업).
set -uo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1
PY="$SIGLIP2_PYTHON"
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
ART="$SIGLIP2_ARTIFACTS"
R=../results/siglip2
until grep -q "OVERNIGHT_WIN2 DONE" $ART/logs/overnight_win2.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=all -- . | tr '\n' ' ')]"
RN50_RESULTS=$R/G3_why "${LOCK[@]}" "SigLIP2 why-writers (interventions)" -- "$PY" -u s2_why_writers.py --n 32 > $ART/logs/overnight_writers.log 2>&1
echo "$(date +%T) writers rc=$?"
echo "OVERNIGHT_WIN3 DONE $(date +%T)"
