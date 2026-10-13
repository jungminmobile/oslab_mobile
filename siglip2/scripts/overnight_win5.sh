#!/usr/bin/env bash
# overnight_win5.sh — 캐시 결정성(Flowers·Cars) 깨끗한 재실행
# 이유: overnight_win.sh의 Flowers·Cars 단계가 도는 동안 siglip2/에 새 분석 스크립트(커밋 전)가 있어 dirty=true로 기록됐다.
#       sha256 비교 자체는 파일 내용 비교라 영향이 없지만, 기록 규칙(dirty 결과는 보고서에 쓰지 않음)에 맞춘다.
# overnight_win4.sh가 끝난 뒤 시작하고 단계마다 공용 잠금을 잡는다. 결과는 sha256_<d>_rerun.txt.
set -uo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1
PY="$SIGLIP2_PYTHON"
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
ART="$SIGLIP2_ARTIFACTS"
R=../results/siglip2
until grep -q "OVERNIGHT_WIN4 DONE" $ART/logs/overnight_win4.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=all -- . | tr '\n' ' ')]"
for d in flowers cars; do
  RN50_CACHE="$ART/cache_check_$d" RN50_RESULTS=$R/G2_fp32_dev/cache_check "${LOCK[@]}" "SigLIP2 cache determinism $d (clean rerun)" -- \
      "$PY" -u s2_build_dev.py $d --threads 4 > $ART/logs/overnight_cache_${d}_rerun.log 2>&1
  rc=$?
  ( cd "$ART" && for k in x y clf ref_fp32; do sha256sum cache/${d}_dev_$k.npy cache_check_$d/${d}_dev_$k.npy; done
    sha256sum cache/${d}_dev_sha1.txt cache_check_$d/${d}_dev_sha1.txt cache/calib_${d}_f32.npy cache_check_$d/calib_${d}_f32.npy ) \
    > $R/G2_fp32_dev/cache_check/sha256_${d}_rerun.txt
  echo "$(date +%T) cache $d rerun rc=$rc same=$(awk '{print $1}' $R/G2_fp32_dev/cache_check/sha256_${d}_rerun.txt | uniq | wc -l)/6 distinct hashes"
  rm -rf "$ART/cache_check_$d"
done
echo "OVERNIGHT_WIN5 DONE $(date +%T)"
