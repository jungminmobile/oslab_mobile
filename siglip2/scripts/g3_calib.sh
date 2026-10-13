#!/usr/bin/env bash
# g3_calib.sh — G3 보정 방식 비교: MinMax / Percentile(99.99, 99.999, 비대칭) / Entropy (ROADMAP G3, RN50 E2와 같은 3종)
# 구성 2개 (G1 진단에서 붕괴하지 않은 형태):
#   mm        : MatMul만 INT8 (잔차 Add·LayerNorm·Softmax·GELU는 FP32)
#   mm_nofc2L : MatMul만 INT8, 블록 5~11 fc2는 FP32 (이상치가 큰 fc2 입력 보호)
# 보정: highres 256장 시드 0 (PRD E3 전 기본). 스크리닝: s2_screen.py (dev 각 200장, 정수 커널, D-024)
# Percentile·Entropy는 메모리 때문에 16장씩 나눠 히스토그램 누적(--hist-chunk 16, 쭈리 승인 10/13).
#   Percentile은 ORT 표준과 동등 확인, Entropy는 "근사"(32장에서 고화질 cos 차이 0.003) — results/siglip2/G3_calib_method/hist_chunk_check
# 추가: D-023 확인(S8 vs U8, mm_nofc2L·MinMax), 모든 양자화에서 정수로 실행된 노드 목록 기록(--dump-ops, VERIFICATION D-G3)
# 사용 (siglip2 폴더에서): bash scripts/g3_calib.sh
set -euo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1
PY="$SIGLIP2_PYTHON"
R=../results/siglip2/G3_calib_method
LATE_FC2=$(for i in $(seq 5 11); do printf '/blocks.%s/mlp/fc2/,' "$i"; done); LATE_FC2=${LATE_FC2%,}
q() {  # q <파일> <s2_quantize 인자...>
  local f=$1; shift
  RN50_RESULTS=$R "$PY" -u s2_quantize.py --calib-source highres --calib-n 256 --op-types MatMul --overwrite --dump-ops "$@" >&2   # 항상 다시 만듦
  [ -f "$SIGLIP2_ARTIFACTS/onnx/$f" ] || { echo "QUANT FAILED $f" >&2; kill -TERM $$; exit 1; }
  echo "$f"
}
M=()
for cfg in mm mm_nofc2L; do
  EX=(); [ $cfg = mm_nofc2L ] && EX=(--exclude "$LATE_FC2")
  M+=("$(q siglip2_image_int8_minmax_highres256_s0_s8_ops-MatMul_$cfg.onnx --method minmax --tag $cfg "${EX[@]}")")
  M+=("$(q siglip2_image_int8_percentile9999_highres256_s0_s8_ops-MatMul_hc16_$cfg.onnx --method percentile --percentile 99.99 --hist-chunk 16 --tag $cfg "${EX[@]}")")
  M+=("$(q siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16_$cfg.onnx --method percentile --percentile 99.999 --hist-chunk 16 --tag $cfg "${EX[@]}")")
  M+=("$(q siglip2_image_int8_entropy_highres256_s0_s8_ops-MatMul_hc16_$cfg.onnx --method entropy --hist-chunk 16 --tag $cfg "${EX[@]}")")
done
M+=("$(q siglip2_image_int8_minmax_highres256_s0_u8_ops-MatMul_mm_nofc2L.onnx --method minmax --act-type u8 --tag mm_nofc2L --exclude "$LATE_FC2")")
RN50_RESULTS=$R "$PY" -u s2_screen.py --k 200 --tag calib --models "${M[@]}"
echo "G3CALIB DONE"
