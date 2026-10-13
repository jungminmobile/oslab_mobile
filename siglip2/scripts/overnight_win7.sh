#!/usr/bin/env bash
# overnight_win7.sh — overnight_win5/6을 오프라인 모드로 다시 실행
# 이유: 06:17 Hugging Face 서버가 요청 제한(HTTP 429)을 걸어 모델·토크나이저 불러오기가 실패했다
#       (캐시에 파일은 모두 있는데 transformers·huggingface_hub가 파일 목록을 서버에 묻는다).
#       HF_HUB_OFFLINE=1이면 캐시만 쓴다 (06:25 확인: load_model 정상).
# 단계: 원인 분석 v2(why_massive, why_writers) → 캐시 결정성 Flowers·Cars 깨끗한 재실행. 단계마다 공용 잠금(먼저 온 순서).
set -uo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
ART="$SIGLIP2_ARTIFACTS"
R=../results/siglip2
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=all -- . | tr '\n' ' ')]"
RN50_RESULTS=$R/G3_why "${LOCK[@]}" "SigLIP2 why-massive v2 (offline)" -- \
    "$PY" -u s2_why_massive.py --n 32 > $ART/logs/overnight_why_v2b.log 2>&1
echo "$(date +%T) why v2 rc=$?"
RN50_RESULTS=$R/G3_why "${LOCK[@]}" "SigLIP2 why-writers v2 (offline)" -- \
    "$PY" -u s2_why_writers.py --n 32 --ort-model siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16_mm.onnx \
    > $ART/logs/overnight_writers_v2b.log 2>&1
echo "$(date +%T) writers v2 rc=$?"
for d in flowers cars; do
  rm -rf "$ART/cache_check_$d"
  RN50_CACHE="$ART/cache_check_$d" RN50_RESULTS=$R/G2_fp32_dev/cache_check "${LOCK[@]}" "SigLIP2 cache determinism $d (clean rerun, offline)" -- \
      "$PY" -u s2_build_dev.py $d --threads 4 > $ART/logs/overnight_cache_${d}_rerun.log 2>&1
  rc=$?
  ( cd "$ART" && for k in x y clf ref_fp32; do sha256sum cache/${d}_dev_$k.npy cache_check_$d/${d}_dev_$k.npy; done
    sha256sum cache/${d}_dev_sha1.txt cache_check_$d/${d}_dev_sha1.txt cache/calib_${d}_f32.npy cache_check_$d/calib_${d}_f32.npy ) \
    > $R/G2_fp32_dev/cache_check/sha256_${d}_rerun.txt 2>&1
  n=$(grep -c "^[0-9a-f]\{64\} " $R/G2_fp32_dev/cache_check/sha256_${d}_rerun.txt)
  pairs_same=$(awk '/^[0-9a-f]{64} /{h[NR]=$1} END{s=0; for(i=1;i<=NR;i+=2) if(h[i]!="" && h[i]==h[i+1]) s++; print s}' $R/G2_fp32_dev/cache_check/sha256_${d}_rerun.txt)
  echo "$(date +%T) cache $d rerun rc=$rc hashes=$n/12 identical_pairs=$pairs_same/6"
  rm -rf "$ART/cache_check_$d"
done
echo "OVERNIGHT_WIN7 DONE $(date +%T)"
