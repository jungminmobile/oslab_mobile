#!/usr/bin/env bash
# G3Q 10절 진단: R0·rothad 위에 attention 행렬곱 FP32 예외 D1(확률×V)·D2(Q×Kᵀ)·D3(둘 다) → dev 6×200 스크리닝. 지연 없음.
# 단계마다 free 여유 ≥ 10 GB 대기 (G3R 세션 DFC와 메모리 공유). 사용 (siglip2/): bash scripts/g3q_diag_attn.sh
set -uo pipefail
source ~/siglip2_env.sh; export OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")/.."
R=../results/siglip2/G3Q_rotate/diag
waitmem() { while true; do f=$(free -g | awk '/Mem:/{print $7}'); [ "$f" -ge 10 ] && { echo "$(date +%T) $1 free=${f}G"; return; }; echo "$(date +%T) $1 free=${f}G 대기"; sleep 60; done; }
BASE="/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/"
KEEP="/trunk/blocks/blocks.0/attn/MatMul_1,/trunk/blocks/blocks.1/attn/MatMul_1,/trunk/blocks/blocks.2/attn/MatMul_1,/trunk/blocks/blocks.3/attn/MatMul_1,/trunk/blocks/blocks.4/attn/MatMul_1,/trunk/blocks/blocks.5/attn/MatMul_1,/trunk/blocks/blocks.6/attn/MatMul_1,/trunk/blocks/blocks.7/attn/MatMul_1,/trunk/blocks/blocks.8/attn/MatMul_1,/trunk/blocks/blocks.9/attn/MatMul_1,/trunk/blocks/blocks.10/attn/MatMul_1,/trunk/blocks/blocks.11/attn/MatMul_1"
COMMON="--calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --dump-ops"
P=percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr
for src in "" rothad; do
  SRC=""; [ -n "$src" ] && SRC="--src siglip2_image_fp32_$src.onnx"
  waitmem "quant ${src:-R0} D1"; RN50_RESULTS=$R "$SIGLIP2_PYTHON" s2_quantize.py $SRC $COMMON --exclude "$BASE,/attn/MatMul_1" --tag fc2L2_D1 || exit 1
  waitmem "quant ${src:-R0} D2"; RN50_RESULTS=$R "$SIGLIP2_PYTHON" s2_quantize.py $SRC $COMMON --exclude "$BASE,/attn/MatMul" --keep "$KEEP" --tag fc2L2_D2 || exit 1
  waitmem "quant ${src:-R0} D3"; RN50_RESULTS=$R "$SIGLIP2_PYTHON" s2_quantize.py $SRC $COMMON --exclude "$BASE,/attn/MatMul" --tag fc2L2_D3 || exit 1
done
waitmem screen
RN50_RESULTS=$R/screen "$SIGLIP2_PYTHON" s2_screen.py --k 200 --tag diag_attn --models \
  siglip2_image_int8_${P}_idcheck_fc2L2.onnx siglip2_image_int8_${P}_fc2L2_D1.onnx siglip2_image_int8_${P}_fc2L2_D2.onnx siglip2_image_int8_${P}_fc2L2_D3.onnx \
  siglip2_image_int8_rothad_${P}_fc2L2.onnx siglip2_image_int8_rothad_${P}_fc2L2_D1.onnx siglip2_image_int8_rothad_${P}_fc2L2_D2.onnx siglip2_image_int8_rothad_${P}_fc2L2_D3.onnx
echo "DIAG DONE $(date +%T)"
