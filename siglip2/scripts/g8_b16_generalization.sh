#!/usr/bin/env bash
# g8_b16_generalization.sh — D-048 뒤 B/16은 "같은 처방이 다른 모델에도 통한다"는 일반화 근거. B/32에서 한 분석을 B/16에도 (2026-10-23)
#  H5 배포 눈금 층별 SNR (B16 2×2 + 배포 M2rot_ort + O-AB + R0) → H1 M2rot FP32 근사 손실 dev 전체 (준용·이 PC 빌드)
#  → B7 입력 변형 8종 (FP32·M2rot FP32·M2rot_ort·O-AB·R0). 정확도·분석만 (PC 써도 값 같음), dev만
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
unset S2_MODEL_ID
export SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2
PY="$SIGLIP2_PYTHON"; ART=$SIGLIP2_ARTIFACTS; L=$ART/logs; STOP=/d/mokerton/locks/siglip2.STOP
R=D:/mokerton/wt_g8/results/siglip2/G8_candidates/b16/generalization
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
chk() { [ -f $STOP ] && { echo "$(date +%T) STOP 파일 — $1 전에 멈춤"; exit 3; }; return 0; }
step() { local name=$1; shift; chk "$name"; echo "$(date +%T) [$name] 시작"; "${LOCK[@]}" "SigLIP2 B16 gen $name" -- "$@" > "$L/gen_$name.log" 2>&1; local rc=$?; echo "$(date +%T) [$name] rc=$rc"; tail -3 "$L/gen_$name.log"; return $rc; }
sha() { "$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }
need() { [ "$(sha "$1")" = "$2" ] || { echo "SHA256 다름: $1 (기대 ${2:0:16})"; exit 2; }; }
M2=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
OAB=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
R0=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
FMR=siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx
G=$ART/onnx_g8; H1=$ART/onnx_h1; JY=$ART/onnx_jy/siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad_pre.onnx
need $ART/onnx/$M2 e84c3727f51e3b97cd59d9ab9604e01acc78f81493926a0696fbf96dc9c60f9e
need $ART/onnx/$OAB db2c1fdd26e0c98fb2ff3bc25d11117dac483b5479dd5c9cb6db75804794a77e
need $H1/$FMR 16ef696e8016f4d1f85f1cafc1d98d32882a813d83e75c179f2cfffa55c36401
echo "$(date +%T) B16 GEN start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- . | wc -l)"
step H5 env RN50_RESULTS=$R/H5_qdq_snr "$PY" -u s2_why_after_qdq.py --pairs \
  b16_orig=$G/siglip2_image_fp32_pre.onnx::$G/siglip2_image_int8__percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
  b16_rot=$G/siglip2_image_fp32_rothad_pre.onnx::$G/siglip2_image_int8_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
  b16_m2=$G/siglip2_image_fp32_rcmaskbcmlp55at5-13at9_pre.onnx::$G/siglip2_image_int8_rcmaskbcmlp55at5-13at9_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
  b16_m2rot_a1=$G/siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad_pre.onnx::$G/$M2 \
  b16_m2rot_ort=$JY::$ART/onnx/$M2 \
  b16_oab=$JY::$ART/onnx/$OAB \
  b16_r0=$ART/onnx/siglip2_image_fp32_pre.onnx::$ART/onnx/$R0
[ -f $H1/${FMR%.onnx}_g8.onnx ] || cp $G/$FMR $H1/${FMR%.onnx}_g8.onnx
need $H1/${FMR%.onnx}_g8.onnx 015dd9b3fc2363c6ff88229d7eccc71fd92b7a7d8da51a8f6e93fbf548c13877
step H1 env RN50_ONNX=$H1 RN50_RESULTS=$R/m2rot_fp32_devfull "$PY" -u s2_paired_ci.py --models $FMR ${FMR%.onnx}_g8.onnx
step B7 env RN50_RESULTS=$R/input_shift "$PY" -u s2_input_shift.py --k 200 --models fp32=siglip2_image_fp32.onnx \
  m2rot_fp32=$H1/$FMR m2rot_ort=$M2 oab=$OAB r0=$R0
echo "$(date +%T) B16 GEN DONE dirty_end=$(git status --porcelain -- . | wc -l)"
