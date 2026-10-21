#!/usr/bin/env bash
# O1 1단계: 회귀 확인(옵션 끔 → 기준 sha256과 같아야 함) + O-A(--no-matmul-out-q) 양자화, B16·B32. nice -n 10, 4스레드.
# 사용 (아무 곳):  bash results/siglip2/O1/o1_quant_a.sh
set -euo pipefail
cd "$(dirname "$0")/../../../siglip2"
source ~/siglip2_env.sh
export OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
R=../results/siglip2/O1
A=$SIGLIP2_ARTIFACTS
PY="nice -n 10 $SIGLIP2_PYTHON -u"
COM="--calib-source highres --calib-n 256 --seed 0 --method percentile --percentile 99.999 --act-type u8 --reduce-range --op-types MatMul --hist-chunk 16 --tag all --dump-ops --threads 4"
B16=siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx
B32=siglip2_image_fp32_g6b_b32_m2rot.onnx
NPY32="--calib-npy $HOME/siglip2_work/g6pre/cache_zoo256/calib256_s0_x.npy"
echo "commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
# 회귀: 별도 폴더(원본 FP32는 심볼릭 링크)에서 옵션을 끄고 다시 만든다
Q=$A/o1_qregress; mkdir -p $Q
ln -sf $A/onnx/$B16 $Q/$B16; ln -sf $A/onnx/$B32 $Q/$B32
RN50_ONNX=$Q RN50_RESULTS=$R/regress $PY s2_quantize.py --src $B16 $COM --overwrite
RN50_ONNX=$Q RN50_RESULTS=$R/regress $PY s2_quantize.py --src $B32 $NPY32 $COM --overwrite
sha256sum $Q/siglip2_image_int8_*_all.onnx $A/onnx/siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
  $A/onnx/siglip2_image_int8_g6b_b32_m2rot_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx | tee $R/regress/quantize_sha256.txt
# O-A
RN50_RESULTS=$R/quant $PY s2_quantize.py --src $B16 $COM --no-matmul-out-q
RN50_RESULTS=$R/quant $PY s2_quantize.py --src $B32 $NPY32 $COM --no-matmul-out-q
sha256sum $A/onnx/siglip2_image_int8_*_moq0_all.onnx | tee $R/quant/sha256_oa.txt
echo O1_QUANT_A_DONE
