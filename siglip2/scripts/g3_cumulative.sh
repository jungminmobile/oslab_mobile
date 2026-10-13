#!/usr/bin/env bash
# g3_cumulative.sh — G3 누적 구성: 보호하는 fc2를 줄여 크기·지연·정확도 곡선 (ROADMAP G3 "누적 구성")
# 출발점: 보정 방식 비교의 최선 = Percentile 99.999(비대칭, 16장씩 나눠 누적) + MatMul만 INT8 + 블록 5~11 fc2 FP32
#   (고화질 cos 0.959, Δtop-1 +0.5, 139.0 MiB, results/siglip2/G3_calib_method)
# 변형 (G1 진단의 fc2 입력 이상치 배율: 블록 5 124배, 9 286배, 10 35배, 11 15배 / 6~8 3~9배):
#   fc2L4 : 블록 5·9·10·11 fc2만 FP32
#   fc2L2 : 블록 5·9 fc2만 FP32
#   fc2L0 : fc2 보호 없음 (= mm, 비교용으로 이미 측정: cos 0.661)
# 스크리닝: s2_screen (dev 각 200장, 정수 커널). 지연: s2_bench_speed (PRD 5절 조건, 스레드 1·2·4, 비공식)
# 사용 (siglip2 폴더에서): bash scripts/g3_cumulative.sh screen   (양자화+스크리닝)
#                          bash scripts/g3_cumulative.sh speed    (지연만 — 다른 무거운 작업이 없을 때)
set -euo pipefail
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1
PY="$SIGLIP2_PYTHON"
R=../results/siglip2/G3_cumulative
fc2() { local s=""; for i in "$@"; do s="$s/blocks.$i/mlp/fc2/,"; done; echo "${s%,}"; }
q() {  # q <파일> <s2_quantize 인자...>
  local f=$1; shift
  RN50_RESULTS=$R "$PY" -u s2_quantize.py --calib-source highres --calib-n 256 --op-types MatMul --method percentile \
      --percentile 99.999 --hist-chunk 16 --overwrite --dump-ops "$@" >&2
  [ -f "$SIGLIP2_ARTIFACTS/onnx/$f" ] || { echo "QUANT FAILED $f" >&2; kill -TERM $$; exit 1; }
  echo "$f"
}
P=siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16
WHAT=${1:-screen}
if [ "$WHAT" = screen ]; then
M=()
M+=("$(q ${P}_fc2L4.onnx --tag fc2L4 --exclude "$(fc2 5 9 10 11)")")
M+=("$(q ${P}_fc2L2.onnx --tag fc2L2 --exclude "$(fc2 5 9)")")
RN50_RESULTS=$R "$PY" -u s2_screen.py --k 200 --tag cumulative --models "${M[@]}"
fi
if [ "$WHAT" = speed ]; then
# 지연: FP32, INT8 전부(참고), 보정 비교 최선(fc2 7개 보호), fc2L4, fc2L2
for t in 1 2 4; do
  RN50_RESULTS=$R/speed "$PY" -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t \
      --models fp32=siglip2_image_fp32.onnx int8_all=siglip2_image_int8_minmax_highres256_s0_s8_all.onnx \
               fc2L7=${P}_mm_nofc2L.onnx fc2L4=${P}_fc2L4.onnx fc2L2=${P}_fc2L2.onnx
done
fi
echo "G3CUM DONE $WHAT"
