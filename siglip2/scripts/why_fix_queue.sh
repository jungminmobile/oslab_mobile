#!/usr/bin/env bash
# why_fix_queue.sh — 원인 분석 3단계(s2_why_fix.py, 64장) 본 실행. 지연 재측정이 끝난 뒤 공용 잠금 순서로.
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
until grep -a -q "SPEED_RERUN4_DONE" $SIGLIP2_ARTIFACTS/logs/speed_rerun_t4.log 2>/dev/null; do sleep 30; done
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=all -- . | tr '\n' ' ')]"
RN50_RESULTS=../results/siglip2/G3_why "$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "SigLIP2 why-fix (scale granularity)" -- \
    "$PY" -u s2_why_fix.py --n 32 > $SIGLIP2_ARTIFACTS/logs/why_fix.log 2>&1
echo "$(date +%T) why_fix rc=$?"
echo "WHY_FIX DONE $(date +%T)"
