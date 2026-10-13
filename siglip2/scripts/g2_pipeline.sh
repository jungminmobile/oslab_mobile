#!/usr/bin/env bash
# g2_pipeline.sh — G2 FP32 기준치(S-A행) 재현: 프롬프트 비교, dev FP32 정확도, 캐시 결정성, 모델 지표, 지연
# 사용 (siglip2 폴더에서): bash scripts/g2_pipeline.sh
# 전제: s2_export_onnx.py로 ONNX, s2_build_dev.py로 dev 캐시 7종(cifar_A·B + 고화질 5종)이 있음
# 속도([5/5])는 다른 무거운 작업이 없을 때 돌린다. 이 PC 값은 비공식(D-010)
set -euo pipefail
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1
PY="$SIGLIP2_PYTHON"
ART="$SIGLIP2_ARTIFACTS"
R=../results/siglip2
Q=siglip2_image_int8_minmax_highres256_s0_s8_all.onnx

echo "[1/5] 프롬프트 템플릿 비교 (CLIP 공식 vs SigLIP 'This is a photo of {}.', dev만)"
RN50_RESULTS=$R/G2_prompt_compare "$PY" -u s2_prompt_compare.py

echo "[2/5] dev FP32 정확도 (캐시된 FP32 ONNX 임베딩, ORT 기본 설정)"
RN50_RESULTS=$R/G2_fp32_dev "$PY" -u s2_evaluate.py --data cifar_A cifar_B pets_dev flowers_dev cars_dev food_dev caltech_dev \
    --models fp32 --exec int --threads 4

echo "[3/5] 캐시 결정성: pets dev 캐시를 다른 폴더에 다시 만들어 sha256 비교"
RN50_CACHE="$ART/cache_check" RN50_RESULTS=$R/G2_fp32_dev/cache_check "$PY" -u s2_build_dev.py pets --threads 4
( cd "$ART" && for k in x y clf ref_fp32; do sha256sum cache/pets_dev_$k.npy cache_check/pets_dev_$k.npy; done
  sha256sum cache/pets_dev_sha1.txt cache_check/pets_dev_sha1.txt cache/calib_pets_f32.npy cache_check/calib_pets_f32.npy ) \
  | tee $R/G2_fp32_dev/cache_check/sha256.txt
rm -rf "$ART/cache_check"

echo "[4/5] 파라미터 수·GMACs·ONNX 크기·peak 메모리 (모델마다 새 프로세스)"
RN50_RESULTS=$R/G2_fp32_baseline "$PY" -u s2_model_stats.py --threads 4 \
    --models fp32=siglip2_image_fp32.onnx int8_all=$Q

echo "[5/5] 지연 (rn50/bench_speed, 워밍업 20, 200장 × 3회 번갈아, 배치 1, 스레드 1·2·4) — 비공식"
for t in 1 2 4; do
  RN50_RESULTS=$R/G2_fp32_baseline/speed "$PY" -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t \
      --models fp32=siglip2_image_fp32.onnx int8_all=$Q
done
echo "G2PIPELINE DONE"
