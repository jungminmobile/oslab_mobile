#!/usr/bin/env bash
# test_g6.sh — (10/16, 쭈리 승인) G6 최종 test 1회 평가: FP32 + ORT 최종(fc2L2 U8 + reduce_range, D-030)
#  데이터셋마다 공용 잠금을 따로 잡는다(개인연구 세션이 사이에 들어올 수 있게). STOP 파일이면 다음 데이터셋 전에 멈춘다.
#  s2_test_eval.py가 데이터셋마다 done_<set>.json을 남겨 두 번 평가하지 않는다. 작은 것부터 (오류를 일찍 보려고).
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"; L=$SIGLIP2_ARTIFACTS/logs; STOP=/d/mokerton/locks/siglip2.STOP
R=../results/siglip2/G6_test; mkdir -p $R
FINAL=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=no -- . | tr '\n' ' ')]"
# ImageNetV2 matched-frequency (test 전용, rn50/datasets.imagenetv2_rows가 $RN50_EXTRA/imagenetv2-mf.tar를 읽음)
IV2="$SIGLIP2_DATA/extra/imagenetv2-mf.tar"
if [ ! -f "$IV2" ]; then
  curl -sSL -o "$IV2.part" "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz" && mv "$IV2.part" "$IV2"
  echo "$(date +%T) imagenetv2 download rc=$? $(ls -la "$IV2" 2>&1)"
fi
"$PY" -c "import hashlib,sys; print('imagenetv2 sha256', hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$IV2" | tee -a $R/data_sha256.txt
for s in pets caltech flowers cars cifar_test imagenetv2 food; do
  [ -f $STOP ] && { echo "$(date +%T) STOP 파일 — $s 전에 멈춤"; exit 3; }
  [ -f $R/done_$s.json ] && { echo "$(date +%T) skip $s (done)"; continue; }
  HF_HUB_OFFLINE=0 RN50_RESULTS=$R "$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "SigLIP2 G6 test $s (FP32+final, 3 GB)" -- \
    "$PY" -u s2_test_eval.py $s --models $FINAL --threads 4 > $L/test_$s.log 2>&1
  echo "$(date +%T) test $s rc=$? $(grep -a '"model"' $L/test_$s.log | tail -2 | tr '\n' ' ')"
done
echo "TEST_G6 DONE $(date +%T)"
