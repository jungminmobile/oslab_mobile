# O2 모델 목록 (source). 파일은 $SIGLIP2_ARTIFACTS/onnx 아래, 모두 B16 224.
Q=percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr
declare -A F=(
  [fp]=siglip2_image_fp32.onnx
  [ofp]=siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx
  [ufp]=siglip2_image_fp32_o2_b16_m2rotu.onnx
  [base]=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_${Q}_all.onnx
  [oab]=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_o1b_${Q}_moq0_all.onnx
  [ubase]=siglip2_image_int8_o2_b16_m2rotu_${Q}_all.onnx
  [uoab]=siglip2_image_int8_o2_b16_m2rotu_o1b_${Q}_moq0_all.onnx
)
ORDER="fp ofp ufp base oab ubase uoab"
