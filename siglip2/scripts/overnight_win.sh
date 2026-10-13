#!/usr/bin/env bash
# overnight_win.sh — 밤샘 Windows 작업. 단계마다 공용 잠금(D:\mokerton\locks\heavy_lock.py)을 잡고 놓는다.
#  1) 원인 분석: 활성값 이상치의 구조(채널·토큰·attention·8비트 0 반올림) → results/siglip2/G3_why
#  2) 히스토그램 나눠 누적 동등성 재확인: 실제 쓴 16장 단위로 (독립 검토 지적) → results/siglip2/G3_calib_method/hist_chunk_check16
#  3) 캐시 결정성: Flowers·Cars·Caltech·Food dev 캐시를 다른 폴더에 다시 만들어 sha256 비교 (Pets는 G2에서 확인)
# 사용 (Git Bash, siglip2 폴더): bash scripts/overnight_win.sh
set -uo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1
PY="$SIGLIP2_PYTHON"
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
ART="$SIGLIP2_ARTIFACTS"
R=../results/siglip2
echo "$(date +%T) start commit=$(git rev-parse --short HEAD)"

RN50_RESULTS=$R/G3_why "${LOCK[@]}" "SigLIP2 why-massive analysis" -- "$PY" -u s2_why_massive.py --n 32 \
    > $ART/logs/overnight_why.log 2>&1
echo "$(date +%T) why rc=$?"

RN50_RESULTS=$R/G3_calib_method/hist_chunk_check16 "${LOCK[@]}" "SigLIP2 hist-chunk check (chunk 16)" -- \
    "$PY" -u s2_hist_chunk_check.py --n 32 --chunk 16 > $ART/logs/overnight_histcheck16.log 2>&1
echo "$(date +%T) histcheck16 rc=$?"

mkdir -p $R/G2_fp32_dev/cache_check
for d in flowers cars caltech food; do
  RN50_CACHE="$ART/cache_check_$d" RN50_RESULTS=$R/G2_fp32_dev/cache_check "${LOCK[@]}" "SigLIP2 cache determinism $d" -- \
      "$PY" -u s2_build_dev.py $d --threads 4 > $ART/logs/overnight_cache_$d.log 2>&1
  rc=$?
  ( cd "$ART" && for k in x y clf ref_fp32; do sha256sum cache/${d}_dev_$k.npy cache_check_$d/${d}_dev_$k.npy; done
    sha256sum cache/${d}_dev_sha1.txt cache_check_$d/${d}_dev_sha1.txt cache/calib_${d}_f32.npy cache_check_$d/calib_${d}_f32.npy ) \
    > $R/G2_fp32_dev/cache_check/sha256_$d.txt
  echo "$(date +%T) cache $d rc=$rc same=$(awk '{print $1}' $R/G2_fp32_dev/cache_check/sha256_$d.txt | uniq | wc -l)/6 distinct hashes"
  rm -rf "$ART/cache_check_$d"
done
echo "OVERNIGHT_WIN DONE $(date +%T)"
