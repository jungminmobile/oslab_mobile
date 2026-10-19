#!/usr/bin/env bash
# G7 정확도: dev 스크리닝 6×200 → dev 전체 짝 비교 (FP32, M2rot_ort, D-default, D-pc, D-m2rot). nice -n 10, ORT 4스레드.
# 사용 (siglip2/):  bash scripts/g7_eval.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
PY="nice -n 10 $SIGLIP2_PYTHON -u"
R=../results/siglip2/G7_dynamic
M="siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx siglip2_image_int8_dynamic_ws8_pt_ops-MatMul.onnx siglip2_image_int8_dynamic_ws8_pc_ops-MatMul_rr.onnx siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_dynamic_ws8_pc_ops-MatMul_rr.onnx"
echo "$(date +%T) screen start commit=$(git rev-parse --short HEAD)"
RN50_RESULTS=$R/screen $PY s2_screen.py --k 200 --threads 4 --tag g7 --models siglip2_image_fp32.onnx $M 2>&1 | grep -vE 'HF_TOKEN|Warning'
echo "$(date +%T) full start"
RN50_RESULTS=$R/full $PY s2_paired_ci.py --threads 4 --n-boot 1000 --models siglip2_image_fp32.onnx $M 2>&1 | grep -vE 'HF_TOKEN|Warning'
echo "G7_EVAL_DONE $(date +%T)"
