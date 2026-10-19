#!/usr/bin/env bash
# G8 정확도: dev 스크리닝 6×200 → dev 전체 짝 비교 (FP32, S-all, S-const, D-m2rot). nice -n 10, ORT 4스레드.
# 단계마다 시작 전: 다른 세션이 지연 측정 중(s2_bench_speed·s2_model_stats·s2_profile_ort·hailortcli)이면 5분 간격으로 기다리고 기록 (용 10/20).
# 사용 (siglip2/):  bash scripts/g8_eval.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
PY="nice -n 10 $SIGLIP2_PYTHON -u"
R=../results/siglip2/G8_constb
WLOG=$R/eval_wait.log
M="siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_cbo_all.onnx siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_dynamic_ws8_pc_ops-MatMul_rr.onnx"
wait_measure() {
  while true; do
    n=$(bash scripts/g8_measure_busy.sh)
    [ "$n" -eq 0 ] && break
    echo "$(date '+%F %T') $1 대기 (다른 세션 지연 측정 프로세스 $n개)" | tee -a $WLOG; sleep 300
  done
  echo "$(date '+%F %T') $1 시작 (측정 프로세스 0)" | tee -a $WLOG
}
echo "$(date +%T) commit=$(git rev-parse --short HEAD)"
wait_measure screen
RN50_RESULTS=$R/screen $PY s2_screen.py --k 200 --threads 4 --tag g8 --models siglip2_image_fp32.onnx $M 2>&1 | grep -vE 'HF_TOKEN|Warning'
wait_measure full
RN50_RESULTS=$R/full $PY s2_paired_ci.py --threads 4 --n-boot 1000 --models siglip2_image_fp32.onnx $M 2>&1 | grep -vE 'HF_TOKEN|Warning'
echo "G8_EVAL_DONE $(date +%T)"
