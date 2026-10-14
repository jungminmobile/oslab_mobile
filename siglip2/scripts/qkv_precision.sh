#!/usr/bin/env bash
# qkv_precision.sh — (10/15 쭈리 요청) U8의 정확도 손실(전부 QKV 정수 경로에서 나옴, ../results/siglip2/G3_cumulative/full_qkvu8)을
# PRD 3절의 대책으로 되돌릴 수 있는가: ① reduce_range(가중치 7비트) ② U8U8(가중치도 U8) ③ session.x64quantprecision=1(VNNI PC에선 영향 없을 것으로 예상)
# 모두 U8 fc2L2와 같은 보정·인자. 단계마다 공용 잠금. 결과: results/siglip2/G3_cumulative/qkv_precision/
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
ART="$SIGLIP2_ARTIFACTS"
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
R=../results/siglip2/G3_cumulative/qkv_precision; mkdir -p $R
S8=siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16_fc2L2.onnx
U8=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_fc2L2.onnx
RR=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
UU=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_wu8_fc2L2.onnx
Q=(--calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --overwrite --dump-ops
   --tag fc2L2 --act-type u8 --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/")
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=all -- . | tr '\n' ' ')]"

RN50_RESULTS=$R "${LOCK[@]}" "SigLIP2 quantize U8 reduce_range" -- "$PY" -u s2_quantize.py "${Q[@]}" --reduce-range > $ART/logs/qp_rr_quantize.log 2>&1
rc=$?; echo "$(date +%T) quantize rr rc=$rc"
RN50_RESULTS=$R "${LOCK[@]}" "SigLIP2 quantize U8U8" -- "$PY" -u s2_quantize.py "${Q[@]}" --weight-type u8 > $ART/logs/qp_wu8_quantize.log 2>&1
rc=$?; echo "$(date +%T) quantize wu8 rc=$rc"
ls "$ART/onnx/$RR" "$ART/onnx/$UU" || { echo "QUANT FAILED"; exit 1; }

RN50_RESULTS=$R "${LOCK[@]}" "SigLIP2 x64quantprecision check" -- "$PY" -u s2_sessopt_check.py --model $U8 --k 200 > $ART/logs/qp_sessopt.log 2>&1
rc=$?; echo "$(date +%T) sessopt rc=$rc $(tail -1 $ART/logs/qp_sessopt.log)"

RN50_RESULTS=$R "${LOCK[@]}" "SigLIP2 screen rr/wu8" -- "$PY" -u s2_screen.py --k 200 --tag qkv_precision --models $RR $UU > $ART/logs/qp_screen.log 2>&1
rc=$?; echo "$(date +%T) screen rc=$rc"
RN50_RESULTS=$R/full "${LOCK[@]}" "SigLIP2 paired_ci S8/U8/rr/wu8" -- "$PY" -u s2_paired_ci.py --models $S8 $U8 $RR $UU > $ART/logs/qp_paired_ci.log 2>&1
rc=$?; echo "$(date +%T) paired_ci rc=$rc"
echo "QKV_PRECISION DONE $(date +%T)"
