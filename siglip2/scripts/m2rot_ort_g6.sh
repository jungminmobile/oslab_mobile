#!/usr/bin/env bash
# m2rot_ort_g6.sh — (D-044, 쭈리 승인) 예선 최종 M2rot_ort: ① 이 PC dev 전체 재현 ② 지연 3세션 ③ test 1회
#  사용: bash scripts/m2rot_ort_g6.sh <M2rot_ort ONNX 경로> <기대 sha256>
#  - 파일 sha256이 기대값과 다르면 아무것도 하지 않는다 (준용이 dev로 평가한 바로 그 파일만 쓴다)
#  - ①: R0와 함께 s2_paired_ci (준용 기록: M2rot_ort −0.23 [−0.64, +0.18], R0 대비 +0.56 [+0.10, +1.01])
#  - ②: FP32·R0·M2rot_ort 번갈아, 스레드 1·2·4 × 3세션, 잠금 안 CPU 30초 25% 미만 (needs idle PC), 비공식
#  - ③: s2_test_eval.py로 test 7종 1회 (새 폴더 G6_test_m2rot, 데이터셋마다 done 파일). 결과가 어떻든 그대로 보고
#  STOP 파일(D:\mokerton\locks\siglip2.STOP)이면 단계 사이에서 멈춘다
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"; ART="$SIGLIP2_ARTIFACTS"; L=$ART/logs; STOP=/d/mokerton/locks/siglip2.STOP
SRC=$1; WANT=$2
NAME=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
R0=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
H=$("$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$SRC")
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) file sha256=$H"
[ "$H" = "$WANT" ] || { echo "SHA256 MISMATCH (want $WANT) — 멈춤"; exit 2; }
[ -f "$ART/onnx/$NAME" ] || cp "$SRC" "$ART/onnx/$NAME"
H2=$("$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$ART/onnx/$NAME"); [ "$H2" = "$WANT" ] || { echo "copy mismatch"; exit 2; }
chk() { [ -f $STOP ] && { echo "$(date +%T) STOP 파일 — $1 전에 멈춤"; exit 3; }; return 0; }
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
# ① dev 전체 재현
chk dev
R=../results/siglip2/G6_m2rot_ort; mkdir -p $R
RN50_RESULTS=$R/dev_full "${LOCK[@]}" "SigLIP2 M2rot_ort dev full (~40 min)" -- "$PY" -u s2_paired_ci.py --models $NAME $R0 > $L/m2rot_devfull.log 2>&1
echo "$(date +%T) dev full rc=$? $(grep -a '"hr_dtop1"' $L/m2rot_devfull.log | tr '\n' ' ' | cut -c1-400)"
# ② 지연 3세션
timed() {
  local job=$1; shift
  while true; do
    chk "$job"
    "$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "$job" -- "C:/Program Files/Git/bin/bash.exe" scripts/_idle_then.sh "$@"
    local rc=$?; echo "$(date +%T) $job rc=$rc"
    [ $rc -eq 75 ] && { sleep 180; continue; }
    return $rc
  done
}
for s in a b c; do for t in 1 2 4; do
  timed "SigLIP2 latency m2rot $s threads=$t (needs idle PC)" env RN50_RESULTS=$R/speed_$s "$PY" -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t \
    --models fp32=siglip2_image_fp32.onnx r0=$R0 m2rot=$NAME
done; done
# ③ test 1회
for s in pets caltech flowers cars cifar_test imagenetv2 food; do
  chk "test $s"
  [ -f $R/test/done_$s.json ] && { echo "$(date +%T) skip $s (done)"; continue; }
  RN50_RESULTS=$R/test "${LOCK[@]}" "SigLIP2 G6 test M2rot_ort $s" -- "$PY" -u s2_test_eval.py $s --models $NAME --threads 4 > $L/test_m2rot_$s.log 2>&1
  echo "$(date +%T) test $s rc=$? $(grep -a '"model"' $L/test_m2rot_$s.log | tail -1 | cut -c1-200)"
done
echo "M2ROT_G6 DONE $(date +%T)"
