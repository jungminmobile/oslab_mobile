#!/usr/bin/env bash
# g8_e5_calib_seed.sh — E5: 같은 M2rot FP32에서 보정 시드 0·1·2로 INT8 → dev 전체 (README 결과 전 고정)
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
unset S2_MODEL_ID
export SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2 RN50_ONNX=D:/mokerton/artifacts/siglip2/onnx_jy
PY="$SIGLIP2_PYTHON"; L=$SIGLIP2_ARTIFACTS/logs
R=../results/siglip2/G8_candidates/b16/E5_calib_seed
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
FMR=siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
M=()
for sd in 0 1 2; do
  f=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s${sd}_u8_ops-MatMul_hc16_rr_all.onnx
  if [ ! -f $RN50_ONNX/$f ]; then
    RN50_RESULTS=$R/int8 "${LOCK[@]}" "SigLIP2 G8 E5 quantize seed $sd" -- "$PY" -u s2_quantize.py --src $FMR --calib-source highres --calib-n 256 --seed $sd \
      --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --tag all --dump-ops --threads 4 > $L/e5_q$sd.log 2>&1
    echo "$(date +%T) quantize seed $sd rc=$?"
  fi
  M+=($f)
done
RN50_RESULTS=$R/dev_full "${LOCK[@]}" "SigLIP2 G8 E5 devfull 3 seeds" -- "$PY" -u s2_paired_ci.py --models "${M[@]}" > $L/e5_devfull.log 2>&1
echo "$(date +%T) devfull rc=$?"; grep -a '"hr_dtop1"' $L/e5_devfull.log | cut -c1-200
echo "$(date +%T) E5 DONE"
