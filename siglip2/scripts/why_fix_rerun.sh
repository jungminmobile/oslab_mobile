#!/usr/bin/env bash
# why_fix_rerun.sh — 원인 분석 3단계 깨끗한 재실행 (14:08 첫 실행 시작 때 커밋 전 스크립트 speed_when_idle.sh가 siglip2/에 있어 dirty)
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
until grep -a -q "WHY_FIX DONE" $SIGLIP2_ARTIFACTS/logs/why_fix_queue.log 2>/dev/null; do sleep 30; done
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=all -- . | tr '\n' ' ')]"
RN50_RESULTS=../results/siglip2/G3_why "$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "SigLIP2 why-fix clean rerun" -- \
    "$PY" -u s2_why_fix.py --n 32 > $SIGLIP2_ARTIFACTS/logs/why_fix_rerun.log 2>&1
echo "$(date +%T) why_fix rerun rc=$?"
echo "WHY_FIX_RERUN DONE $(date +%T)"
