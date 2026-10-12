#!/usr/bin/env bash
# g3_sensitivity.sh — G3 민감도 지도: 블록 단위(14) + 연산 유형(6), 하나만 INT8, 나머지 FP32 (ROADMAP G3)
# 사용 (siglip2 폴더에서): bash scripts/g3_sensitivity.sh [block|optype|all]
# 양자화: s2_quantize.py 기본(MinMax, highres 256 s0, S8S8). 스크리닝: s2_screen.py (dev 6종 각 앞 200장, 정수 커널, D-024)
set -euo pipefail
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1
PY="$SIGLIP2_PYTHON"
R=../results/siglip2/G3_sensitivity
WHAT=${1:-all}
q() {  # q <tag> <s2_quantize 노드 선택 인자...>
  local tag=$1; shift
  local f="siglip2_image_int8_minmax_highres256_s0_s8_$tag.onnx"
  if [ ! -f "$SIGLIP2_ARTIFACTS/onnx/$f" ]; then
    RN50_RESULTS=$R "$PY" -u s2_quantize.py --calib-source highres --calib-n 256 --tag "$tag" "$@" >&2
  fi
  echo "$f"
}
exact() {  # exact <블록 안 노드 접미사> → 블록 0~11의 정확한 노드 이름 쉼표 목록
  local s=""; for i in $(seq 0 11); do s="$s/trunk/blocks/blocks.$i/$1,"; done; echo "${s%,}"
}

if [ "$WHAT" = block ] || [ "$WHAT" = all ]; then
  echo "[block] 하나만 INT8: patch embedding, 블록 0~11, pooling head(+마지막 norm)"
  M=()
  M+=("$(q only_patch --only /trunk/patch_embed/)")
  for i in $(seq 0 11); do M+=("$(q only_b$i --only /trunk/blocks/blocks.$i/)"); done
  M+=("$(q only_head --only /trunk/attn_pool/,/trunk/norm/)")
  RN50_RESULTS=$R "$PY" -u s2_screen.py --k 200 --tag block --models "${M[@]}"
fi

if [ "$WHAT" = optype ] || [ "$WHAT" = all ]; then
  echo "[optype] 블록 0~11에서 한 연산 유형만 INT8 (MatMul 노드만, bias Add·LayerNorm·잔차 Add는 FP32)"
  M=()
  M+=("$(q only_qkv --only-exact "$(exact attn/qkv/MatMul)")")
  M+=("$(q only_attn_qk --only-exact "$(exact attn/MatMul)")")
  M+=("$(q only_attn_av --only-exact "$(exact attn/MatMul_1)")")
  M+=("$(q only_proj --only-exact "$(exact attn/proj/MatMul)")")
  M+=("$(q only_fc1 --only-exact "$(exact mlp/fc1/MatMul)")")
  M+=("$(q only_fc2 --only-exact "$(exact mlp/fc2/MatMul)")")
  RN50_RESULTS=$R "$PY" -u s2_screen.py --k 200 --tag optype --models "${M[@]}"
fi
echo "G3SENS DONE"
