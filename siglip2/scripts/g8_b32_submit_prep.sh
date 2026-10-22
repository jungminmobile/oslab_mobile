#!/usr/bin/env bash
# g8_b32_submit_prep.sh — D-048(예선 = B/32) 제출 준비 실험 중 제출 파일 확정 없이 할 수 있는 것 (2026-10-22). dev·캐시만, test 없음
#  S1 모델 통계(FP32·M2rot·O-AB) → S2 ORT 연산별 프로파일(200장×1) → S3 배포 눈금 층별 SNR(2×2+R0) → S4 M2rot FP32(근사) dev 전체
#  → S5 입력 변형 8종 → S6 보정 시드 1·2 흔들림(M2rot, E5와 같은 방식) → S7 같은 실행 지연·메모리 3세션(FP32·M2rot·O-AB, PC 조용할 때)
#  STOP 파일이면 단계 사이에서 멈춤. S1~S6은 PC를 써도 값이 같다(느려질 뿐). S7만 PC를 쓰면 안 됨
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2_b32
PY="$SIGLIP2_PYTHON"; ART=$SIGLIP2_ARTIFACTS; B=$ART/onnx; L=$ART/logs; STOP=/d/mokerton/locks/siglip2.STOP; mkdir -p $L
R=D:/mokerton/wt_g8/results/siglip2/G8_candidates/b32/submit_prep
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
chk() { [ -f $STOP ] && { echo "$(date +%T) STOP 파일 — $1 전에 멈춤"; exit 3; }; return 0; }
step() { local name=$1; shift; chk "$name"; echo "$(date +%T) [$name] 시작"; "${LOCK[@]}" "SigLIP2 B32 prep $name" -- "$@" > "$L/prep_$name.log" 2>&1; local rc=$?; echo "$(date +%T) [$name] rc=$rc"; tail -3 "$L/prep_$name.log"; return $rc; }
sha() { "$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }
need() { [ "$(sha "$1")" = "$2" ] || { echo "SHA256 다름: $1 (기대 ${2:0:16})"; exit 2; }; }
FP=siglip2_image_fp32.onnx
FM=siglip2_image_fp32_rcmaskbcmlp15at6u5-7at10u9_rothad.onnx
M2=siglip2_image_int8_rcmaskbcmlp15at6u5-7at10u9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
OAB=siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
R0=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
need $B/$OAB 7a01044c6ba32bf4f5a12aba82eb19e06753f13c8bad43b9f6fcfcd9f484dbf6
echo "$(date +%T) B32 PREP start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- . | wc -l) M2rot sha $(sha $B/$M2 | cut -c1-16) FP32 sha $(sha $B/$FP | cut -c1-16)"

step S1_stats env RN50_RESULTS=$R/model_stats "$PY" -u s2_model_stats.py --models fp32=$FP m2rot=$M2 oab=$OAB
step S2_profile env RN50_RESULTS=$R/profile "$PY" -u s2_profile_ort.py --threads 1 4 --n 200 --warmup 20 --models FP32=$FP M2rot=$M2 OAB=$OAB
step S3_snr env RN50_RESULTS=$R/H5_qdq_snr "$PY" -u s2_why_after_qdq.py --pairs \
  b32_orig=$B/siglip2_image_fp32_pre.onnx::$B/siglip2_image_int8__percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
  b32_rot=$B/siglip2_image_fp32_rothad_pre.onnx::$B/siglip2_image_int8_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
  b32_m2=$B/siglip2_image_fp32_rcmaskbcmlp15at6u5-7at10u9_pre.onnx::$B/siglip2_image_int8_rcmaskbcmlp15at6u5-7at10u9_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
  b32_m2rot=$B/siglip2_image_fp32_rcmaskbcmlp15at6u5-7at10u9_rothad_pre.onnx::$B/$M2 \
  b32_r0=$B/siglip2_image_fp32_pre.onnx::$B/$R0
step S4_m2rot_fp32 env RN50_RESULTS=$R/m2rot_fp32_devfull "$PY" -u s2_paired_ci.py --models $FM
step S5_shift env RN50_RESULTS=$R/input_shift "$PY" -u s2_input_shift.py --k 200 --models fp32=$FP m2rot_fp32=$FM m2rot=$M2 oab=$OAB r0=$R0
QOPT=(--calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --tag all --dump-ops --threads 4)
MS=()
for sd in 1 2; do
  f=${M2/_s0_/_s${sd}_}
  [ -f $B/$f ] || step S6_q_s$sd env RN50_RESULTS=$R/seed/int8 "$PY" -u s2_quantize.py --src $FM "${QOPT[@]}" --seed $sd || exit 1
  MS+=($f)
done
step S6_devfull env RN50_RESULTS=$R/seed/dev_full "$PY" -u s2_paired_ci.py --models $M2 "${MS[@]}"

# S7 같은 실행 지연·메모리 (시작 전 CPU·속도 확인; B/16 FP32 4T 프로브가 아니라 이 모델 FP32 4T로 확인)
gate() {
  while true; do
    c=$("$PY" -c 'import psutil; print(round(psutil.cpu_percent(interval=30)))')
    pr=$("$PY" scripts/speed_probe.py D:/mokerton/artifacts/siglip2/onnx/siglip2_image_fp32.onnx 4); ms=$(echo "$pr" | sed -E 's/.*median=([0-9.]+).*/\1/')
    echo "$(date +%T) gate cpu30s=$c% (B16 FP32 probe) $pr"
    if [ "$c" -lt 25 ] && awk "BEGIN{exit !($ms < 200)}"; then return 0; fi
    echo "$(date +%T) gate FAIL → 3분 뒤 다시"; sleep 180
  done
}
export -f gate; export PY
chk S7; LAT=D:/mokerton/wt_g8/results/siglip2/G8_candidates/latency_this_pc/b32_submit_20261022; mkdir -p $LAT
"${LOCK[@]}" "SigLIP2 B32 prep S7 latency (needs idle PC)" -- "C:/Program Files/Git/bin/bash.exe" -c '
  set -e; gate
  "$PY" -u scripts/cross_latency.py --out "'$LAT'" --mem --models b32_fp32='$B'/'$FP' b32_m2rot='$B'/'$M2' b32_oab='$B'/'$OAB'
' > $LAT/latency_run.log 2>&1
echo "$(date +%T) [S7] rc=$?"; grep -E "gate|^session" $LAT/latency_run.log | tail -10
echo "$(date +%T) B32 PREP DONE dirty_end=$(git status --porcelain -- . | wc -l)"
