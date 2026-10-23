#!/usr/bin/env bash
# b32t_data_run.sh — README B-2: test 7종 데이터 준비 + FP32 ONNX 대조 → 사진 대조(b32t_check_data). 데스크탑 WSL.
# 사용: setsid nohup bash b32t_data_run.sh > ~/siglip2_work/b32t/data_run.log 2>&1 < /dev/null &
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd)
source ~/siglip2_env.sh
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_ARTIFACTS=$HOME/siglip2_work/b32t/art_b32 HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1
O=$HOME/siglip2_work/b32t; CODE=$HOME/oslab_b32p
REF=$O/junhyuk_test.jsonl
git -C $CODE show 092a4ab:results/siglip2/G8_candidates/b32/test/test.jsonl > $REF
echo "$(date +%T) start commit=$(git -C $H rev-parse --short HEAD) dirty=$(git -C $H status --porcelain --untracked-files=no | wc -l) code=$(git -C $CODE rev-parse --short HEAD) code_dirty=$(git -C $CODE status --porcelain --untracked-files=no | wc -l)"
for s in pets caltech flowers cars food cifar_test imagenetv2; do
  [ -f $O/done_data_$s.json ] && { echo "$(date +%T) skip $s (done)"; continue; }
  timeout 7200 nice -n 5 $SIGLIP2_PYTHON -u $H/b32t_data.py $s --code $CODE --out $O --log $H > $O/data_$s.log 2>&1
  echo "$(date +%T) data $s rc=$? $(tail -1 $O/data_$s.log | cut -c1-300)"
done
$SIGLIP2_PYTHON $H/b32t_check_data.py --ref $REF --log $H > $O/data_check.log 2>&1; rc=$?
echo "$(date +%T) check rc=$rc"; grep -E '"ok"|all_ok' $H/data_check.json | tr -d ' ' | tr '\n' ' '; echo
echo "$(date +%T) DATA DONE"
