#!/usr/bin/env bash
# g1_pipeline.sh — G1 기록 재현: ONNX 동일성·연산 목록, export·양자화 결정성(sha256), INT8 전부(MinMax) dev 정수 커널 평가
# 사용 (siglip2 폴더에서): bash scripts/g1_pipeline.sh
# 전제: s2_export_onnx.py로 ONNX가 있고, s2_build_dev.py로 dev 캐시 6종(cifar_A + 고화질 5종)과 보정 후보가 있음
set -euo pipefail
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1
PY="$SIGLIP2_PYTHON"
ART="$SIGLIP2_ARTIFACTS"
R1=../results/siglip2/G1_onnx_export
R2=../results/siglip2/G1_int8_minmax_all
Q=siglip2_image_int8_minmax_highres256_s0_s8_all.onnx
mkdir -p "$R1" "$R2"

echo "[1/5] PyTorch vs ONNX 동일성 + PyTorch fused vs unfused + 연산 목록"
"$PY" -u s2_export_onnx.py --verify 64 --ops --threads 4 --out "$R1"

echo "[2/5] fused attention 그대로 내보낸 변형의 연산 목록 (D-019 근거 기록, 임시 폴더)"
RN50_ONNX="$ART/onnx_tmp_fused" "$PY" -u s2_export_onnx.py --fused-attn --ops --overwrite --threads 4 --out "$R1/fused_variant"
rm -rf "$ART/onnx_tmp_fused"

echo "[3/5] 같은 코드로 다시 export → sha256 비교 (결정성)"
RN50_ONNX="$ART/onnx_tmp_reexport" "$PY" -u s2_export_onnx.py --overwrite --threads 4 --out "$ART/logs/reexport_check"
( cd "$ART" && sha256sum onnx/siglip2_image_fp32.onnx onnx_tmp_reexport/siglip2_image_fp32.onnx \
    onnx/siglip2_image_fp32_b1.onnx onnx_tmp_reexport/siglip2_image_fp32_b1.onnx ) | tee "$R1/reexport_sha256.txt"
rm -rf "$ART/onnx_tmp_reexport"

echo "[4/5] 같은 조건으로 다시 양자화 → sha256 비교 (결정성) + 정수 커널 연산 목록"
before=$(sha256sum "$ART/onnx/$Q" | cut -d' ' -f1)
RN50_RESULTS="$R2" "$PY" -u s2_quantize.py --calib-source highres --calib-n 256 --tag all --dump-ops --overwrite --threads 4
after=$(sha256sum "$ART/onnx/$Q" | cut -d' ' -f1)
printf 'before %s\nafter  %s\nsame   %s\n' "$before" "$after" "$([ "$before" = "$after" ] && echo yes || echo NO)" | tee "$R2/requant_sha256.txt"

echo "[5/5] dev 6종 정수 커널 평가 (FP32 = 캐시된 FP32 ONNX 임베딩)"
RN50_RESULTS="$R2" "$PY" -u s2_evaluate.py --data cifar_A pets_dev flowers_dev cars_dev food_dev caltech_dev \
    --models fp32 "$Q" --exec int --threads 4
echo "G1PIPELINE DONE"
