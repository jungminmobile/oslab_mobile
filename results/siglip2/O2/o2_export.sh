#!/usr/bin/env bash
# O2 2단계: (회귀) B32 m2rot 배치 가변 export를 다시 만들어 sha256 23236123…과 같은지 → (본) B16 통일 규칙 M2rot FP32 export.
# s2_g6.py export는 ~/siglip2_work/g6/onnx/g6_<모델>_m2rot_dyn.onnx에 쓴다. B32 기존 파일은 백업 뒤 비교한다.
# 사용 (아무 곳):  bash results/siglip2/O2/o2_export.sh
set -euo pipefail
cd "$(dirname "$0")/../../../siglip2"
source ~/siglip2_env.sh
export OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
R=../results/siglip2/O2
G=$HOME/siglip2_work/g6/onnx
PY="nice -n 10 $SIGLIP2_PYTHON -u"
echo "commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
cp --preserve=timestamps $G/g6_b32_m2rot_dyn.onnx $G/g6_b32_m2rot_dyn.onnx.o2bak
RN50_RESULTS=$R/regress $PY s2_g6.py --model b32 export --cfg m2rot --dynamic --overwrite
sha256sum $G/g6_b32_m2rot_dyn.onnx $G/g6_b32_m2rot_dyn.onnx.o2bak | tee $R/regress/export_sha256.txt
[ "$(sha256sum < $G/g6_b32_m2rot_dyn.onnx)" = "$(sha256sum < $G/g6_b32_m2rot_dyn.onnx.o2bak)" ] && rm $G/g6_b32_m2rot_dyn.onnx.o2bak && echo REGRESS_SAME
RN50_RESULTS=$R/export $PY s2_g6.py --model b16 export --cfg m2rot --dynamic
cp --preserve=timestamps $G/g6_b16_m2rot_dyn.onnx $SIGLIP2_ARTIFACTS/onnx/siglip2_image_fp32_o2_b16_m2rotu.onnx
sha256sum $SIGLIP2_ARTIFACTS/onnx/siglip2_image_fp32_o2_b16_m2rotu.onnx | tee $R/export/fp32_sha256.txt
echo O2_EXPORT_DONE
