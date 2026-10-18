#!/usr/bin/env bash
# G3Q-R2: r2·rothad_r2 INT8 (R0와 같은 명령 + --src) → dev 6×200 스크리닝 (R0·rothad·E3 같은 실행). free ≥ 10 GB 대기.
set -uo pipefail
source ~/siglip2_env.sh; export OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")/.."
R=../results/siglip2/G3Q_rotate/r2
waitmem() { while [ $(free -g | awk '/Mem:/{print $7}') -lt 10 ]; do echo "$(date +%T) $1 대기"; sleep 60; done; echo "$(date +%T) $1 free=$(free -g | awk '/Mem:/{print $7}')G"; }
COMMON="--calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --dump-ops"
P=percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr
for s in r2had64 rothad_r2had64; do
  waitmem "quant $s"
  RN50_RESULTS=$R/int8 "$SIGLIP2_PYTHON" s2_quantize.py --src siglip2_image_fp32_$s.onnx $COMMON --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/" --tag fc2L2 || exit 1
done
waitmem screen
RN50_RESULTS=$R/screen "$SIGLIP2_PYTHON" s2_screen.py --k 200 --tag r2 --models \
  siglip2_image_int8_${P}_idcheck_fc2L2.onnx siglip2_image_int8_rothad_${P}_fc2L2.onnx siglip2_image_int8_${P}_fc2L2_E3.onnx \
  siglip2_image_int8_r2had64_${P}_fc2L2.onnx siglip2_image_int8_rothad_r2had64_${P}_fc2L2.onnx
echo "R2 INT8 DONE $(date +%T)"
