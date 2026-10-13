#!/usr/bin/env bash
# g4_run.sh — G4 구성들: optimize(모델 스크립트) → SDK_QUANTIZED 에뮬레이션 (Pets dev 앞 64장). WSL Ubuntu-22.04에서 실행
# 사용: bash g4_run.sh [에뮬레이션 배치] [구성 이름...]   예: bash g4_run.sh 2 D_official_b9fc2 E_fc2_b0b5b9
#   구성 이름 X → siglip2/hailo/g4_X.alls. 배치를 줄이면 메모리 사용이 준다(배치 8에서 B 구성이 11.6 GB + 스왑).
#   인자가 없으면 첫 실행(2026-10-13)과 같은 배치 8, 구성 A·C·B.
set -uo pipefail
source ~/hailo10_venv/bin/activate
BS=${1:-8}; shift || true
CFGS=("$@"); [ ${#CFGS[@]} -eq 0 ] && CFGS=(A_ewadd_patch16 C_official B_latefc2)
SRC=~/siglip2_dfc/20261013_124213
W=~/siglip2_dfc/g4_$(date +%Y%m%d_%H%M%S); mkdir -p $W; cd $W
cp $SRC/siglip2_b16_224.har $SRC/calib_highres64_s0_nhwc.npy .
cp /mnt/d/mokerton/oslab_mobile/siglip2/hailo/g4_*.alls .
tr -d '\r' < /mnt/d/mokerton/oslab_mobile/siglip2/hailo/s2_dfc_emulate.py > s2_dfc_emulate.py
for f in g4_*.alls; do tr -d '\r' < $f > $f.tmp && mv $f.tmp $f; done
echo "commit=$(git -C /mnt/d/mokerton/oslab_mobile rev-parse --short HEAD) bs=$BS cfgs=${CFGS[*]}" > run_info.txt
for cfg in "${CFGS[@]}"; do
  t0=$(date +%s)
  hailo optimize siglip2_b16_224.har --hw-arch hailo10h --model-script g4_$cfg.alls --calib-set-path calib_highres64_s0_nhwc.npy \
      --output-har-path opt_$cfg.har > optimize_$cfg.log 2>&1
  echo "$cfg optimize rc=$? sec=$(( $(date +%s) - t0 ))"
  python s2_dfc_emulate.py --har opt_$cfg.har --data pets_dev --k 64 --bs $BS --quantized-only \
      --out /mnt/d/mokerton/artifacts/siglip2/dfc/g4_emulate.jsonl > emulate_$cfg.log 2>&1
  echo "$cfg emulate rc=$? $(grep '^quantized' emulate_$cfg.log)"
done
O=/mnt/d/mokerton/oslab_mobile/results/siglip2/G4_dfc_first/$(basename $W); mkdir -p $O
cp run_info.txt g4_*.alls optimize_*.log emulate_*.log $O/ 2>/dev/null; sha256sum opt_*.har > $O/har_sha256.txt
cp /mnt/d/mokerton/artifacts/siglip2/dfc/g4_emulate.jsonl /mnt/d/mokerton/oslab_mobile/results/siglip2/G4_dfc_first/emulate.jsonl
echo G4RUN DONE $W
