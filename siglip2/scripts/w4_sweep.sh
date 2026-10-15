#!/usr/bin/env bash
# w4_sweep.sh — (10/15 밤, D-036 위임: 4비트 시작) ORT 최종(fc2L2 U8 + reduce_range, D-030) 위에 가중치 4비트(MatMulNBits, 블록 32, accuracy_level 4)를 넓혀 가며
#  크기-정확도 곡선을 만든다. 대회 안내 4.1 "INT8/INT4", 성능 개선(크기) 25 + 기법 25.
#  A: 블록 0~11 MLP fc1   B: + MLP fc2 (블록 5·9 fc2는 FP32 그대로)   C: + attention qkv·proj   D: + pooling head 가중치
#  나머지 MatMul은 최종과 같은 U8+rr INT8 QDQ. 단계마다 공용 잠금. 결과: results/siglip2/G3_w4/
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
ART="$SIGLIP2_ARTIFACTS"
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
# W4_ONE_LOCK=1: 바깥에서 잠금 하나로 전체(약 40분)를 감싸 실행할 때 (10/15 밤: 개인연구 E20 뒤 WSL이 내려가 있는 동안 끝내려고).
#  단계마다 잠금을 놓으면 그 사이 WSL 조각이 끼어들어 WSL VM 메모리와 Windows 양자화가 겹친다
nolock() { shift 2; "$@"; }                     # 사용: nolock "<job>" -- <명령>
[ "${W4_ONE_LOCK:-0}" = 1 ] && LOCK=(nolock)
R=../results/siglip2/G3_w4; mkdir -p $R
Q=(--calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --overwrite --dump-ops
   --act-type u8 --reduce-range --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/")
B=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_w4b32a4
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=no -- . | tr '\n' ' ')]"
declare -A RX=(
  [A]='^/blocks\.\d+/mlp/fc1/MatMul$'
  [B]='^/blocks\.\d+/mlp/fc[12]/MatMul$'
  [C]='^/blocks\.\d+/(mlp/fc[12]|attn/(qkv|proj))/MatMul$'
  [D]='^/(blocks\.\d+/(mlp/fc[12]|attn/(qkv|proj))|attn_pool/(q|kv|proj|mlp/fc[12]))/MatMul$'
)
# 0) 옵션을 끈 기본 경로가 이전과 같은지: 4비트 코드 추가 뒤 U8+rr 최종 모델을 다시 만들어 sha256을 기록값(5eee7839…)과 비교 (CLAUDE.md 협업 규칙 2)
REF=5eee783955591d1de818b532ea73b2bb2ea8a008e327cc92d323992c0f9083ac
RN50_RESULTS=$R "${LOCK[@]}" "SigLIP2 quantize identity check" -- "$PY" -u s2_quantize.py "${Q[@]}" --tag idcheck_fc2L2 > $ART/logs/w4_idcheck.log 2>&1
H=$(sha256sum "$ART/onnx/siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_idcheck_fc2L2.onnx" | cut -c1-64)
echo "$(date +%T) identity check sha256=$H $([ "$H" = "$REF" ] && echo SAME || echo DIFFERENT)" | tee -a $R/identity_check.txt
[ "$H" = "$REF" ] || { echo "IDENTITY CHECK FAILED — 기본 경로가 바뀜, 4비트 중단"; exit 1; }
M=()
for k in A B C D; do
  RN50_RESULTS=$R "${LOCK[@]}" "SigLIP2 quantize w4$k" -- "$PY" -u s2_quantize.py "${Q[@]}" --w4-regex "${RX[$k]}" --tag "w4${k}_fc2L2" > $ART/logs/w4_quantize_$k.log 2>&1
  rc=$?; echo "$(date +%T) quantize w4$k rc=$rc $(grep -a saved $ART/logs/w4_quantize_$k.log | tail -1)"
  [ -f "$ART/onnx/${B}_w4${k}_fc2L2.onnx" ] && M+=("${B}_w4${k}_fc2L2.onnx")
done
[ ${#M[@]} -gt 0 ] || { echo "W4 QUANT FAILED"; exit 1; }
RN50_RESULTS=$R "${LOCK[@]}" "SigLIP2 screen w4" -- "$PY" -u s2_screen.py --k 200 --tag w4 --models "${M[@]}" > $ART/logs/w4_screen.log 2>&1
rc=$?; echo "$(date +%T) screen rc=$rc"; grep -a hr_dtop1 $ART/logs/w4_screen.log
echo "W4_SWEEP DONE $(date +%T)"
