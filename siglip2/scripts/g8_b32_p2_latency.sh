#!/usr/bin/env bash
# g8_b32_p2_latency.sh — 정민 E14 P2(e465d10d)를 이 PC 기본 세션으로 (2026-10-27, 쭈리 "P2로 교체" 조건 확인)
#  정민·준용 측정은 C3(mem pattern 끔) 세션이었다. 채점은 기본 세션이므로 보고서 대표값과 같은 방법(cross_latency.py, 기본 세션)으로 다시 잰다.
#  ① 같은 실행 지연·메모리 (FP32·O-AB·P2, 1·2·4T × 3세션) ② ORT 연산별 프로파일 200장 (1·4T, O-AB·P2)
#  P2 = make_models_e14.py(정민 jungmin/e14-graph e1f8428)로 이 PC에서 만든 파일, sha256 e465d10d = 정민 원본
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2_b32
PY="$SIGLIP2_PYTHON"; B=$SIGLIP2_ARTIFACTS/onnx
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
FP=siglip2_image_fp32.onnx
OAB=siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
P2=siglip2_image_int8_g6b_b32_m2rot_o1b_e14P2_fastgelu_patchint8.onnx
LAT=D:/mokerton/wt_g8/results/siglip2/G8_candidates/latency_this_pc/b32_p2_20261027
PROF=D:/mokerton/wt_g8/results/siglip2/G8_candidates/b32/p2/profile
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
echo "$(date +%T) B32 P2 start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- . | wc -l)"
sha256sum $B/$P2 | cut -c1-16
mkdir -p $LAT $PROF
"${LOCK[@]}" "SigLIP2 B32 P2 latency (needs idle PC)" -- "C:/Program Files/Git/bin/bash.exe" -c '
  set -e; gate
  "$PY" -u scripts/cross_latency.py --out "'$LAT'" --mem --models b32_fp32='$B'/'$FP' b32_oab='$B'/'$OAB' b32_p2='$B'/'$P2'
' > $LAT/latency_run.log 2>&1
echo "$(date +%T) [latency] rc=$?"; grep -E "gate|^session" $LAT/latency_run.log | tail -12
"${LOCK[@]}" "SigLIP2 B32 P2 profile (needs idle PC)" -- "C:/Program Files/Git/bin/bash.exe" -c '
  set -e; gate
  RN50_RESULTS='$PROF' "$PY" -u s2_profile_ort.py --threads 1 4 --n 200 --warmup 20 --models OAB='$OAB' P2='$P2'
' > $LAT/profile_run.log 2>&1
echo "$(date +%T) [profile] rc=$?"; grep -E "gate|^t=" $LAT/profile_run.log | tail -8
echo "$(date +%T) B32 P2 LATENCY DONE dirty_end=$(git status --porcelain -- . | wc -l)"
