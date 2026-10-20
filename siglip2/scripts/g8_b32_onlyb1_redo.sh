#!/usr/bin/env bash
# g8_b32_onlyb1_redo.sh — B/32 민감도 only_b1 다시 (독립 검토 10/20: 그 줄만 dirty=true로 기록됨 — 코드가 아니라 결과 문서가 커밋 전이었던 때로 보이나 확인 못 해 다시 잼)
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2_b32
export RN50_RESULTS=../results/siglip2/G8_candidates/b32/G3_sensitivity
PY="$SIGLIP2_PYTHON"; L=$SIGLIP2_ARTIFACTS/logs
echo "commit=$(git rev-parse --short HEAD) dirty=[$(git status --porcelain | tr '\n' ' ')]"
"$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "SigLIP2 G8 B/32 only_b1 redo" -- "C:/Program Files/Git/bin/bash.exe" -c \
 "\"$PY\" -u s2_quantize.py --calib-source highres --calib-n 256 --tag only_b1 --overwrite --only /trunk/blocks/blocks.1/ && \"$PY\" -u s2_screen.py --k 200 --tag block --models siglip2_image_int8_minmax_highres256_s0_s8_only_b1.onnx" > $L/onlyb1_redo.log 2>&1
echo "rc=$?"; grep -a '"hr_cos"' $L/onlyb1_redo.log | tail -1 | cut -c1-200
