#!/usr/bin/env bash
# g8_b16_a5.sh — G8 A5: B/16 M2rot의 CPU 변환 형식 실험 (README A5, 판정 규칙은 결과 전 고정). A1 다음에 실행 (A1이 만든 onnx_g8의 M2rot FP32·INT8 사용)
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
unset S2_MODEL_ID
export SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2 RN50_ONNX=D:/mokerton/artifacts/siglip2/onnx_g8
PY="$SIGLIP2_PYTHON"; ART=$SIGLIP2_ARTIFACTS; L=$ART/logs; STOP=/d/mokerton/locks/siglip2.STOP
R=../results/siglip2/G8_candidates/b16/A5_format; mkdir -p $R
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
chk() { [ -f $STOP ] && { echo "$(date +%T) STOP 파일 — $1 전에 멈춤"; exit 3; }; return 0; }
step() { local name=$1; shift; chk "$name"; echo "$(date +%T) [$name] 시작"; "${LOCK[@]}" "SigLIP2 G8 A5 $name" -- "$@" > "$L/a5_$name.log" 2>&1; local rc=$?; echo "$(date +%T) [$name] rc=$rc"; tail -2 "$L/a5_$name.log"; return $rc; }
timed() { local job=$1; shift; while true; do chk "$job"; "${LOCK[@]}" "$job" -- "C:/Program Files/Git/bin/bash.exe" scripts/_idle_then.sh "$@"; local rc=$?; echo "$(date +%T) $job rc=$rc"; [ $rc -eq 75 ] && { sleep 180; continue; }; return $rc; done; }
FMR=siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx
BASE=(--src $FMR --calib-source highres --calib-n 256 --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --tag all --dump-ops --threads 4)
echo "$(date +%T) start commit=$(git rev-parse --short HEAD)"
step qa env RN50_RESULTS=$R/int8 "$PY" -u s2_quantize.py "${BASE[@]}" --op-types MatMul --quant-format qoperator || exit 1
step qb env RN50_RESULTS=$R/int8 "$PY" -u s2_quantize.py "${BASE[@]}" --op-types MatMul,Transpose,Reshape,Split,Squeeze,Concat,Gather || exit 1
step qc env RN50_RESULTS=$R/int8 "$PY" -u s2_quantize.py "${BASE[@]}" || exit 1
M0=$(ls $RN50_ONNX | grep -E "^siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx$")
MA=$(ls $RN50_ONNX | grep -E "rothad_.*_ops-MatMul_hc16_rr_qop_all.onnx$")
MB=$(ls $RN50_ONNX | grep -E "rothad_.*_ops-MatMul-Transpose-.*_rr_all.onnx$")
MC=$(ls $RN50_ONNX | grep -E "rothad_percentile99999_highres256_s0_u8_hc16_rr_all.onnx$")
echo "M0=$M0 MA=$MA MB=$MB MC=$MC" | tee $R/models.txt
step screen env RN50_RESULTS=$R/screen "$PY" -u s2_screen.py --k 200 --tag a5 --models $M0 $MA $MB $MC
timed "SigLIP2 G8 A5 speed screen t4 (needs idle PC)" env RN50_RESULTS=$R/speed_screen "$PY" -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads 4 \
  --models fp32=siglip2_image_fp32.onnx m2rot=$M0 qop=$MA datamove=$MB allops=$MC
echo "$(date +%T) A5 SCREEN DONE"
