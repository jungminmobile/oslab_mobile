#!/usr/bin/env bash
# g8_b32_test.sh — D-049: 예선 제출 B/32 O-AB(7a01044c…)로 test 7종 1회 (B/32 첫 test 사용, 쭈리 승인 10/22)
#  - 파일 sha256이 기대값과 다르면 아무것도 하지 않음. 데이터셋마다 done 파일(s2_test_eval의 1회 보장) — 이미 있으면 건너뜀
#  - 결과가 어떻든 그대로 보고 (고르는 데 쓰지 않음). 끝에 s2_test_summary로 고화질 Δ·95% 구간
#  STOP 파일(D:\mokerton\locks\siglip2.STOP)이면 데이터셋 사이에서 멈춤. 정확도만이라 PC를 써도 값은 같다
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2_b32
PY="$SIGLIP2_PYTHON"; ART=$SIGLIP2_ARTIFACTS; L=$ART/logs; STOP=/d/mokerton/locks/siglip2.STOP; mkdir -p $L
OAB=siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
WANT=7a01044c6ba32bf4f5a12aba82eb19e06753f13c8bad43b9f6fcfcd9f484dbf6
R=D:/mokerton/wt_g8/results/siglip2/G8_candidates/b32/test
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
H=$("$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$ART/onnx/$OAB")
echo "$(date +%T) B32 TEST start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- . | wc -l) sha256=${H:0:16}"
[ "$H" = "$WANT" ] || { echo "SHA256 MISMATCH (want ${WANT:0:16}) — 멈춤"; exit 2; }
mkdir -p $R
for s in pets caltech flowers cars food cifar_test imagenetv2; do
  [ -f $STOP ] && { echo "$(date +%T) STOP 파일 — test $s 전에 멈춤"; exit 3; }
  [ -f $R/done_$s.json ] && { echo "$(date +%T) skip $s (done)"; continue; }
  RN50_RESULTS=$R "${LOCK[@]}" "SigLIP2 B32 test O-AB $s" -- "$PY" -u s2_test_eval.py $s --models $OAB --threads 4 > $L/test_b32_$s.log 2>&1
  echo "$(date +%T) test $s rc=$? $(grep -a '"model"' $L/test_b32_$s.log | tail -1 | cut -c1-200)"
done
RN50_RESULTS=$R "$PY" -u s2_test_summary.py > $L/test_b32_summary.log 2>&1
echo "$(date +%T) summary rc=$?"; tail -20 $L/test_b32_summary.log
echo "$(date +%T) B32 TEST DONE dirty_end=$(git status --porcelain -- . | wc -l)"
