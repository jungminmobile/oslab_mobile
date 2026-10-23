#!/usr/bin/env bash
# g8_b32_cool_latency.sh — B/32 제출 모델 지연·메모리·연산별 프로파일을 PC가 식고 조용할 때 다시 (2026-10-23)
#  이유: 10/22 저녁 측정은 연속 실행 발열로 2·4T가 약 40% 느렸고, S2 프로파일은 게임 실행 중이라 무효
#  ① 같은 실행 지연·메모리 (FP32·M2rot·O-AB, 1·2·4T × 3세션) ② ORT 연산별 프로파일 200장 (1·4T)
#  단계마다 시작 전 확인: CPU 30초 평균 < 25%, B/16 FP32 4T 프로브 < 200 ms (아니면 3분마다 다시)
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2_b32
PY="$SIGLIP2_PYTHON"; B=$SIGLIP2_ARTIFACTS/onnx
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
FP=siglip2_image_fp32.onnx
M2=siglip2_image_int8_rcmaskbcmlp15at6u5-7at10u9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
OAB=siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
LAT=D:/mokerton/wt_g8/results/siglip2/G8_candidates/latency_this_pc/b32_cool_20261023
PROF=D:/mokerton/wt_g8/results/siglip2/G8_candidates/b32/submit_prep/profile_cool
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
echo "$(date +%T) B32 COOL start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- . | wc -l)"
mkdir -p $LAT
"${LOCK[@]}" "SigLIP2 B32 cool latency (needs idle PC)" -- "C:/Program Files/Git/bin/bash.exe" -c '
  set -e; gate
  "$PY" -u scripts/cross_latency.py --out "'$LAT'" --mem --models b32_fp32='$B'/'$FP' b32_m2rot='$B'/'$M2' b32_oab='$B'/'$OAB'
' > $LAT/latency_run.log 2>&1
echo "$(date +%T) [latency] rc=$?"; grep -E "gate|^session" $LAT/latency_run.log | tail -12
"${LOCK[@]}" "SigLIP2 B32 cool profile (needs idle PC)" -- "C:/Program Files/Git/bin/bash.exe" -c '
  set -e; gate
  RN50_RESULTS='$PROF' "$PY" -u s2_profile_ort.py --threads 1 4 --n 200 --warmup 20 --models FP32='$FP' M2rot='$M2' OAB='$OAB'
' > $LAT/profile_run.log 2>&1
echo "$(date +%T) [profile] rc=$?"; grep -E "gate|^t=" $LAT/profile_run.log | tail -8
echo "$(date +%T) B32 COOL DONE dirty_end=$(git status --porcelain -- . | wc -l)"
