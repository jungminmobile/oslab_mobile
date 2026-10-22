#!/usr/bin/env bash
# g8_more_b16.sh — G8 결정 없이 할 수 있는 추가 실험 (2026-10-22, REVIEW_20261022 4절). 모두 dev·캐시만, test·제출 파일 변경 없음
#  B1 재분석(모델 실행 없음) → B2 지문 다시(VNNI 표시 고침) → H5 배포 눈금 층별 SNR(B16 7짝·B32 5짝) → H1 M2rot FP32 dev 전체(준용·이 PC 빌드)
#  → B7 입력 변형 8종 → H6 M2 캐시 상수 시드 1·2 (나머지는 A1과 같게) dev 전체
#  정확도·분석만이라 PC를 써도 값은 같다 (느려질 뿐). 무거운 단계는 공용 잠금 안. STOP 파일이면 단계 사이에서 멈춤
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
unset S2_MODEL_ID
PY="$SIGLIP2_PYTHON"; ART=D:/mokerton/artifacts/siglip2; ART32=D:/mokerton/artifacts/siglip2_b32; L=$ART/logs; STOP=/d/mokerton/locks/siglip2.STOP
R=D:/mokerton/wt_g8/results/siglip2/G8_candidates/b16; RR=D:/mokerton/wt_g8/results/siglip2
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
chk() { [ -f $STOP ] && { echo "$(date +%T) STOP 파일 — $1 전에 멈춤"; exit 3; }; return 0; }
step() { local name=$1; shift; chk "$name"; echo "$(date +%T) [$name] 시작"; "${LOCK[@]}" "SigLIP2 G8 more $name" -- "$@" > "$L/more_$name.log" 2>&1; local rc=$?; echo "$(date +%T) [$name] rc=$rc"; tail -3 "$L/more_$name.log"; return $rc; }
sha() { "$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }
need() { [ "$(sha "$1")" = "$2" ] || { echo "SHA256 다름: $1 (기대 ${2:0:16})"; exit 2; }; }
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- . | wc -l)"
M2=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
OAB=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
R0=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
FMR=siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx
need $ART/onnx/$M2 e84c3727f51e3b97cd59d9ab9604e01acc78f81493926a0696fbf96dc9c60f9e
need $ART/onnx/$OAB db2c1fdd26e0c98fb2ff3bc25d11117dac483b5479dd5c9cb6db75804794a77e
need $ART/onnx_h1/$FMR 16ef696e8016f4d1f85f1cafc1d98d32882a813d83e75c179f2cfffa55c36401
need $ART/onnx_g8/$M2 009ea9d5403e8cba09e98c168d12e0bcad3466898969df707d95ce0ecdb8e956

# B1 재분석 (가벼움, 잠금 없이)
chk B1; "$PY" -u scripts/g8_reanalysis.py --test-r0 D:/mokerton/oslab_mobile/results/siglip2/G6_test --test-m2 D:/mokerton/oslab_mobile/results/siglip2/G6_m2rot_ort/test \
  --dev-dirs $R/O_AB/dev_full $RR/G6_m2rot_ort/dev_full D:/mokerton/oslab_mobile/results/siglip2/G6_m2rot_ort/dev_full --results-root $RR --out $R/B1_reanalysis > $L/more_B1.log 2>&1
echo "$(date +%T) [B1] rc=$?"; tail -12 $L/more_B1.log
# B2 지문 (가벼움)
FP=$RR/G8_candidates/repro_fingerprint
"$PY" -u scripts/repro_fingerprint.py --art $ART --out $FP/fp_jjury_b16_v2.json > $L/more_B2.log 2>&1; echo "$(date +%T) [B2 b16] rc=$?"
S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 "$PY" -u scripts/repro_fingerprint.py --art $ART32 --out $FP/fp_jjury_b32_v2.json >> $L/more_B2.log 2>&1; echo "$(date +%T) [B2 b32] rc=$?"

# H5 배포 눈금 층별 SNR (B16: A1 2×2 + 배포 + O-AB + R0, B32: 2×2 + R0)
G=$ART/onnx_g8; JY=$ART/onnx_jy/siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad_pre.onnx
step H5_b16 env RN50_RESULTS=$R/H5_qdq_snr "$PY" -u s2_why_after_qdq.py --pairs \
  b16_orig=$G/siglip2_image_fp32_pre.onnx::$G/siglip2_image_int8__percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
  b16_rot=$G/siglip2_image_fp32_rothad_pre.onnx::$G/siglip2_image_int8_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
  b16_m2=$G/siglip2_image_fp32_rcmaskbcmlp55at5-13at9_pre.onnx::$G/siglip2_image_int8_rcmaskbcmlp55at5-13at9_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
  b16_m2rot_a1=$G/siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad_pre.onnx::$G/$M2 \
  b16_m2rot_ort=$JY::$ART/onnx/$M2 \
  b16_oab=$JY::$ART/onnx/$OAB \
  b16_r0=$ART/onnx/siglip2_image_fp32_pre.onnx::$ART/onnx/$R0
B=$ART32/onnx
step H5_b32 env S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_ARTIFACTS=$ART32 RN50_RESULTS=$RR/G8_candidates/b32/H5_qdq_snr "$PY" -u s2_why_after_qdq.py --pairs \
  b32_orig=$B/siglip2_image_fp32_pre.onnx::$B/siglip2_image_int8__percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
  b32_rot=$B/siglip2_image_fp32_rothad_pre.onnx::$B/siglip2_image_int8_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
  b32_m2=$B/siglip2_image_fp32_rcmaskbcmlp15at6u5-7at10u9_pre.onnx::$B/siglip2_image_int8_rcmaskbcmlp15at6u5-7at10u9_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
  b32_m2rot=$B/siglip2_image_fp32_rcmaskbcmlp15at6u5-7at10u9_rothad_pre.onnx::$B/siglip2_image_int8_rcmaskbcmlp15at6u5-7at10u9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
  b32_r0=$B/siglip2_image_fp32_pre.onnx::$B/siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx

# H1 M2rot FP32(양자화 없음) dev 전체: 준용 빌드(배포 INT8의 원본)와 이 PC 빌드(A1)
H1=$ART/onnx_h1
[ -f $H1/${FMR%.onnx}_g8.onnx ] || cp $G/$FMR $H1/${FMR%.onnx}_g8.onnx
need $H1/${FMR%.onnx}_g8.onnx 015dd9b3fc2363c6ff88229d7eccc71fd92b7a7d8da51a8f6e93fbf548c13877
step H1 env RN50_ONNX=$H1 RN50_RESULTS=$R/H1_m2rot_fp32 "$PY" -u s2_paired_ci.py --models $FMR ${FMR%.onnx}_g8.onnx

# B7 입력 변형 8종 × (FP32, M2rot FP32, M2rot_ort, O-AB, R0), 고화질 5종 × 200장
step B7 env RN50_RESULTS=$R/B7_input_shift "$PY" -u s2_input_shift.py --k 200 --models fp32=siglip2_image_fp32.onnx \
  m2rot_fp32=$H1/$FMR m2rot_ort=$M2 oab=$OAB r0=$R0

# H6 M2 캐시 상수 시드 1·2 (회전 행렬·FP32·양자화 옵션은 A1과 같게, 캐시 시드만 바꿈) → dev 전체, A1(시드 0)과 짝 비교
QOPT=(--calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --tag all --dump-ops --threads 4)
MS=$ART/onnx_m2seed; mkdir -p $MS
[ -f $MS/${M2%.onnx}_cs0.onnx ] || cp $G/$M2 $MS/${M2%.onnx}_cs0.onnx
for k in 1 2; do
  D=$ART/onnx_m2s$k; mkdir -p $D
  [ -f $D/rotq_rothad.npy ] || cp $G/rotq_rothad.npy $D/
  [ -f $D/siglip2_image_fp32.onnx ] || cp $ART/onnx/siglip2_image_fp32.onnx $D/
  step H6_cache_s$k env RN50_ONNX=$D RN50_RESULTS=$R/H6_m2_seed/regcache "$PY" -u s2_regcache.py cache --rot rothad --seed $k || exit 1
  step H6_ucache_s$k env RN50_ONNX=$D RN50_RESULTS=$R/H6_m2_seed/regcache "$PY" -u s2_regcache.py ucache --rot rothad --seed $k || exit 1
  step H6_export_s$k env RN50_ONNX=$D RN50_RESULTS=$R/H6_m2_seed/regcache "$PY" -u s2_regcache.py export --graph mask --mask-form bcast --mask-site mlp --rot rothad --seed $k || exit 1
  step H6_q_s$k env RN50_ONNX=$D RN50_RESULTS=$R/H6_m2_seed/int8 "$PY" -u s2_quantize.py --src $FMR "${QOPT[@]}" || exit 1
  cp $D/$M2 $MS/${M2%.onnx}_cs$k.onnx
done
step H6_devfull env RN50_ONNX=$MS RN50_RESULTS=$R/H6_m2_seed/dev_full "$PY" -u s2_paired_ci.py --models ${M2%.onnx}_cs0.onnx ${M2%.onnx}_cs1.onnx ${M2%.onnx}_cs2.onnx
echo "$(date +%T) MORE_B16 DONE dirty_end=$(git status --porcelain -- . | wc -l)"
