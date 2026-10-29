#!/usr/bin/env bash
# E5 다시 (10/21 19시): 첫 실행의 시드 2 양자화와 dev 평가가 커밋 전 새 파일 때문에 dirty로 기록됨 → 깨끗한 상태에서
#  시드 2 양자화를 다시 해 이전 파일과 sha256 비교(같은 PC·같은 코드라 같아야 함) → dev 전체 3시드 → 이어서 E4b
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
unset S2_MODEL_ID
export SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2
PY="$SIGLIP2_PYTHON"; L=$SIGLIP2_ARTIFACTS/logs; J=$SIGLIP2_ARTIFACTS/onnx_jy
R=../results/siglip2/G8_candidates/b16/E5_calib_seed
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
sha() { "$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain -- . | tr '\n' ' ')]"
F2=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s2_u8_ops-MatMul_hc16_rr_all.onnx
mkdir -p $J/old && mv -f $J/$F2 $J/old/$F2
RN50_ONNX=$J RN50_RESULTS=$R/int8 "${LOCK[@]}" "SigLIP2 G8 E5 re-quantize seed 2" -- "$PY" -u s2_quantize.py --src siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx \
  --calib-source highres --calib-n 256 --seed 2 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --tag all --dump-ops --threads 4 > $L/e5_q2_redo.log 2>&1
a=$(sha $J/old/$F2); b=$(sha $J/$F2); echo "seed2 old ${a:0:16} new ${b:0:16} $([ "$a" = "$b" ] && echo SAME || echo DIFF)" | tee $R/seed2_requant_sha.txt
M=""; for sd in 0 1 2; do M="$M siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s${sd}_u8_ops-MatMul_hc16_rr_all.onnx"; done
RN50_ONNX=$J RN50_RESULTS=$R/dev_full "${LOCK[@]}" "SigLIP2 G8 E5 devfull 3 seeds (clean)" -- "$PY" -u s2_paired_ci.py --models $M > $L/e5_devfull_clean.log 2>&1
echo "$(date +%T) E5 devfull rc=$?"; grep -a '"hr_dtop1"' $L/e5_devfull_clean.log | cut -c1-220
RN50_ONNX=$SIGLIP2_ARTIFACTS/onnx_g8 RN50_RESULTS=../results/siglip2/G8_candidates/b16/E4_why_after "${LOCK[@]}" "SigLIP2 G8 E4b pct+SNR" -- "$PY" -u s2_why_after_pct.py --n 32 --threads 4 \
  --models 원래=siglip2_image_fp32.onnx 회전만=siglip2_image_fp32_rothad.onnx M2만=siglip2_image_fp32_rcmaskbcmlp55at5-13at9.onnx 회전+M2=siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx > $L/e4b.log 2>&1
echo "$(date +%T) E4b rc=$?"; tail -4 $L/e4b.log | cut -c1-600
