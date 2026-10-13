#!/usr/bin/env bash
# g4_run.sh — G4 첫 구성들: optimize(모델 스크립트) → SDK_QUANTIZED 에뮬레이션 (Pets dev 64장). WSL에서 실행
set -uo pipefail
source ~/hailo10_venv/bin/activate
SRC=~/siglip2_dfc/20261013_124213
W=~/siglip2_dfc/g4_$(date +%Y%m%d_%H%M%S); mkdir -p $W; cd $W
cp $SRC/siglip2_b16_224.har $SRC/calib_highres64_s0_nhwc.npy .
cp /mnt/d/mokerton/oslab_mobile/siglip2/hailo/g4_*.alls .
tr -d '\r' < /mnt/d/mokerton/oslab_mobile/siglip2/hailo/s2_dfc_emulate.py > s2_dfc_emulate.py
for cfg in A_ewadd_patch16 C_official B_latefc2; do
  t0=$(date +%s)
  hailo optimize siglip2_b16_224.har --hw-arch hailo10h --model-script g4_$cfg.alls --calib-set-path calib_highres64_s0_nhwc.npy \
      --output-har-path opt_$cfg.har > optimize_$cfg.log 2>&1
  echo "$cfg optimize rc=$? sec=$(( $(date +%s) - t0 ))"
  python s2_dfc_emulate.py --har opt_$cfg.har --data pets_dev --k 64 --quantized-only \
      --out /mnt/d/mokerton/artifacts/siglip2/dfc/g4_emulate.jsonl > emulate_$cfg.log 2>&1
  echo "$cfg emulate rc=$? $(grep '^quantized' emulate_$cfg.log)"
done
echo G4RUN DONE $W
