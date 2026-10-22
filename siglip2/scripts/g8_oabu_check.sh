#!/usr/bin/env bash
# g8_oabu_check.sh — 준용 O2 요청(10/22): 통일 규칙 B16 O-AB(o2_b16_m2rotu, sha256 f5108c7b…)를 test 전에 이 PC에서 dev 전체·지연으로 확인
#  ① dev 전체: O-AB(통일) · O-AB(기존 55@5·13@9) · M2rot_ort 짝 비교 (준용 데스크탑 O-AB(통일)−O-AB −0.08 [−0.26, +0.09]와 같은 경향인지)
#  ② 지연: B16 FP32 · M2rot_ort · O-AB · O-AB(통일) 같은 실행 번갈아 1·2·4스레드 3세션 (시작 전 CPU·속도 확인)
#  dev만, test·제출 파일 변경 없음. 파일은 Release 그대로(재양자화 안 함)
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
unset S2_MODEL_ID
PY="$SIGLIP2_PYTHON"; ART=D:/mokerton/artifacts/siglip2; L=$ART/logs; STOP=/d/mokerton/locks/siglip2.STOP
R=D:/mokerton/wt_g8/results/siglip2/G8_candidates/b16/O2_unified; mkdir -p $R
LAT=D:/mokerton/wt_g8/results/siglip2/G8_candidates/latency_this_pc/oabu_20261022
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
sha() { "$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }
need() { [ "$(sha "$1")" = "$2" ] || { echo "SHA256 다름: $1 (기대 ${2:0:16})"; exit 2; }; }
M2=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
OAB=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
OABU=siglip2_image_int8_o2_b16_m2rotu_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
need $ART/onnx/$M2 e84c3727f51e3b97cd59d9ab9604e01acc78f81493926a0696fbf96dc9c60f9e
need $ART/onnx/$OAB db2c1fdd26e0c98fb2ff3bc25d11117dac483b5479dd5c9cb6db75804794a77e
need $ART/onnx/$OABU f5108c7b06daae81e510b936cb3be2913ed6fd5ca3e4d9b508c91688735e85a7
echo "$(date +%T) OABU start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- . | wc -l)"
[ -f $STOP ] && { echo STOP; exit 3; }
RN50_RESULTS=$R/dev_full "${LOCK[@]}" "SigLIP2 O-AB unified dev full (~30 min)" -- "$PY" -u s2_paired_ci.py --models $OABU $OAB $M2 > $L/oabu_devfull.log 2>&1
echo "$(date +%T) [dev] rc=$?"; grep -a '"hr_dtop1"\|^pair' $L/oabu_devfull.log | cut -c1-220
[ -f $STOP ] && { echo STOP; exit 3; }
FP=$ART/onnx/siglip2_image_fp32.onnx
gate() {
  while true; do
    c=$("$PY" -c 'import psutil; print(round(psutil.cpu_percent(interval=30)))')
    pr=$("$PY" scripts/speed_probe.py $FP 4); ms=$(echo "$pr" | sed -E 's/.*median=([0-9.]+).*/\1/')
    echo "$(date +%T) gate cpu30s=$c% $pr"
    if [ "$c" -lt 25 ] && awk "BEGIN{exit !($ms < 200)}"; then return 0; fi
    echo "$(date +%T) gate FAIL → 3분 뒤 다시"; sleep 180
  done
}
export -f gate; export PY FP
mkdir -p $LAT
"${LOCK[@]}" "SigLIP2 O-AB unified latency (needs idle PC)" -- "C:/Program Files/Git/bin/bash.exe" -c '
  set -e; gate
  "$PY" -u scripts/cross_latency.py --out "'$LAT'" --mem \
    --models b16_fp32='$FP' b16_m2rot='$ART'/onnx/'$M2' b16_oab='$ART'/onnx/'$OAB' b16_oabu='$ART'/onnx/'$OABU'
' > $LAT/latency_run.log 2>&1
echo "$(date +%T) [latency] rc=$?"; grep -E "gate|^session" $LAT/latency_run.log | tail -12
echo "$(date +%T) OABU DONE dirty_end=$(git status --porcelain -- . | wc -l)"
