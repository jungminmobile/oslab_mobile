# O1 모델 목록 (source해서 쓴다). 파일은 $SIGLIP2_ARTIFACTS/onnx 아래.
B16S=rcmaskbcmlp55at5-13at9_rothad; B32S=g6b_b32_m2rot
Q=percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr
declare -A F=(
  [b16fp]=siglip2_image_fp32.onnx
  [b16base]=siglip2_image_int8_${B16S}_${Q}_all.onnx
  [b16oa]=siglip2_image_int8_${B16S}_${Q}_moq0_all.onnx
  [b16ob]=siglip2_image_int8_${B16S}_o1b_${Q}_all.onnx
  [b16oab]=siglip2_image_int8_${B16S}_o1b_${Q}_moq0_all.onnx
  [b32fp]=siglip2_image_fp32_g6b_b32.onnx
  [b32base]=siglip2_image_int8_${B32S}_${Q}_all.onnx
  [b32oa]=siglip2_image_int8_${B32S}_${Q}_moq0_all.onnx
  [b32ob]=siglip2_image_int8_${B32S}_o1b_${Q}_all.onnx
  [b32oab]=siglip2_image_int8_${B32S}_o1b_${Q}_moq0_all.onnx
)
ORDER="b16fp b16base b16oa b16ob b16oab b32fp b32base b32oa b32ob b32oab"
size_of() { case $1 in b16*) echo 224;; *) echo 256;; esac; }
