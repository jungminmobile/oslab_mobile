#!/usr/bin/env bash
# overnight_win2.sh — overnight_win.sh의 원인 분석·히스토그램 확인을 깨끗한 작업 트리에서 다시 실행
# 이유: 첫 실행 때 WSL hailo 명령이 siglip2/ 폴더에 로그 파일(acceleras.log 등)을 남겨 dirty=true로 기록됐다
#       (코드는 커밋 706e473과 같아 값은 유효하지만, 보고서에는 dirty=false 기록만 쓴다 — CLAUDE.md 6절)
# overnight_win.sh가 끝난 뒤 시작하고, 단계마다 공용 잠금을 잡는다.
set -uo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1
PY="$SIGLIP2_PYTHON"
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
ART="$SIGLIP2_ARTIFACTS"
R=../results/siglip2
until grep -q "OVERNIGHT_WIN DONE" $ART/logs/overnight_win.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=all -- . | tr '\n' ' ')]"
RN50_RESULTS=$R/G3_why "${LOCK[@]}" "SigLIP2 why-massive (clean rerun)" -- "$PY" -u s2_why_massive.py --n 32 > $ART/logs/overnight_why2.log 2>&1
rc=$?; echo "$(date +%T) why rerun rc=$rc"
RN50_RESULTS=$R/G3_calib_method/hist_chunk_check16 "${LOCK[@]}" "SigLIP2 hist-chunk check16 (clean rerun)" -- \
    "$PY" -u s2_hist_chunk_check.py --n 32 --chunk 16 > $ART/logs/overnight_histcheck16_2.log 2>&1
rc=$?; echo "$(date +%T) histcheck16 rerun rc=$rc"
echo "OVERNIGHT_WIN2 DONE $(date +%T)"
