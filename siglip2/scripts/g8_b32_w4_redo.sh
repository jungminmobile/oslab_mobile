#!/usr/bin/env bash
# g8_b32_w4_redo.sh — B/32 4비트 B·C·D 다시 (10/20 20시). g8_b32_full.sh에서 `env`(MSYS 프로그램)를 거치면 정규식의 '[12]'·'(…|…)' 때문에
#  Cygwin 명령줄 glob 처리가 역슬래시를 지워 '/blocks.d+/'가 됐다 (B·C는 0개로 실패, D는 풀링 헤드 5개만 → 무효, onnx_bad로 옮김). A(12개)는 정상.
#  여기서는 env 없이 RN50_RESULTS를 export해서 잠금(python)이 바로 python을 부르게 한다 (B/16 w4_sweep.sh와 같은 방식)
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2_b32
PY="$SIGLIP2_PYTHON"; ART=$SIGLIP2_ARTIFACTS; L=$ART/logs
R=../results/siglip2/G8_candidates/b32; export RN50_RESULTS=$R/G3_w4
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
declare -A RX=([B]='/blocks\.\d+/mlp/fc[12]/MatMul$' [C]='/blocks\.\d+/(mlp/fc[12]|attn/(qkv|proj))/MatMul$' [D]='(/blocks\.\d+/(mlp/fc[12]|attn/(qkv|proj))|/attn_pool/(q|kv|proj|mlp/fc[12]))/MatMul$')
W4Q=(--calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/" --threads 4 --overwrite)
for k in B C D; do
  "${LOCK[@]}" "SigLIP2 G8 B/32 w4_$k redo" -- "$PY" -u s2_quantize.py "${W4Q[@]}" --w4-regex "${RX[$k]}" --tag w4${k}_fc2L2 > $L/full_w4_${k}_redo.log 2>&1
  echo "$(date +%T) w4_$k rc=$? $(grep -a 'w4-regex' $L/full_w4_${k}_redo.log | head -1)"
done
R0=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
M=$R0; for k in A B C D; do f=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_w4b32a4_w4${k}_fc2L2.onnx; [ -f $ART/onnx/$f ] && M="$M $f"; done
"${LOCK[@]}" "SigLIP2 G8 B/32 w4 screen redo" -- "$PY" -u s2_screen.py --k 200 --tag w4 --models $M > $L/full_w4_screen_redo.log 2>&1
echo "$(date +%T) screen rc=$?"; grep -a '"hr_cos"' $L/full_w4_screen_redo.log | cut -c1-200
