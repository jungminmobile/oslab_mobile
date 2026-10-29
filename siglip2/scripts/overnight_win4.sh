#!/usr/bin/env bash
# overnight_win4.sh — 원인 분석 일반성 확인: 거대 채널·고정 위치 토큰이 데이터셋에 상관없이 나타나는가
# (처음 분석은 Pets·Caltech 64장뿐) → dev 6종(CIFAR A 포함) 각각 16장, 데이터셋마다 한 행
# overnight_win3.sh가 끝난 뒤 시작하고 공용 잠금을 잡는다.
set -uo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1
PY="$SIGLIP2_PYTHON"
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
ART="$SIGLIP2_ARTIFACTS"
R=../results/siglip2
until grep -q "OVERNIGHT_WIN3 DONE" $ART/logs/overnight_win3.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=all -- . | tr '\n' ' ')]"
for d in pets_dev caltech_dev flowers_dev cars_dev food_dev cifar_A; do
  RN50_RESULTS=$R/G3_why/per_set "${LOCK[@]}" "SigLIP2 why-massive $d" -- "$PY" -u s2_why_massive.py --n 16 --sets $d \
      > $ART/logs/overnight_why_$d.log 2>&1
  rc=$?; echo "$(date +%T) why $d rc=$rc"
done
echo "OVERNIGHT_WIN4 DONE $(date +%T)"
