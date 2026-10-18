#!/usr/bin/env bash
# G3Q 10-7 진단 2: R0 위에 묶음 하나씩 FP32 (E1 풀링 헤드, E2 qkv, E3 attn proj, E4 블록 fc1, E5 나머지 fc2) → dev 6×200 스크리닝. 지연 없음.
# 단계마다 free 여유 ≥ 10 GB 대기. 사용 (siglip2/): bash scripts/g3q_diag_groups.sh
set -uo pipefail
source ~/siglip2_env.sh; export OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")/.."
R=../results/siglip2/G3Q_rotate/diag2
waitmem() { while true; do f=$(free -g | awk '/Mem:/{print $7}'); [ "$f" -ge 10 ] && { echo "$(date +%T) $1 free=${f}G"; return; }; echo "$(date +%T) $1 free=${f}G 대기"; sleep 60; done; }
BASE="/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/"
FC1=$(for i in $(seq 0 11); do printf "/blocks.$i/mlp/fc1/,"; done | sed 's/,$//')
FC2=$(for i in 0 1 2 3 4 6 7 8 10 11; do printf "/blocks.$i/mlp/fc2/,"; done | sed 's/,$//')
declare -A EX=([E1]="/attn_pool/" [E2]="/attn/qkv/" [E3]="/attn/proj/" [E4]="$FC1" [E5]="$FC2")
COMMON="--calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --dump-ops"
P=percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr
M="siglip2_image_int8_${P}_idcheck_fc2L2.onnx"
for e in E1 E2 E3 E4 E5; do
  waitmem "quant $e"
  RN50_RESULTS=$R "$SIGLIP2_PYTHON" s2_quantize.py $COMMON --exclude "$BASE,${EX[$e]}" --tag fc2L2_$e || exit 1
  M="$M siglip2_image_int8_${P}_fc2L2_$e.onnx"
done
waitmem screen
RN50_RESULTS=$R/screen "$SIGLIP2_PYTHON" s2_screen.py --k 200 --tag diag_groups --models $M
echo "DIAG2 DONE $(date +%T)"
