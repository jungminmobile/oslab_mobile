#!/usr/bin/env bash
# g4_more2_wsl.sh — G4 계획 항목: DFC 최적화 수준 0 → 1 (구성 D, 보정 256장) (10/15)
# GPU가 없어 수준 1이 되는지부터 확인한다. 되면 스크리닝(dev 6×200)으로 D256(수준 0)과 비교.
# 보정 1024장은 고화질 보정 후보 풀이 311장이라 지금은 불가 (풀 확대는 데이터 구성 변경 → 팀 결정).
set -uo pipefail
source ~/hailo10_venv/bin/activate
cd ~
LOCK="python3 /mnt/d/mokerton/locks/heavy_lock.py run --owner siglip2 --job"
REPO=/mnt/d/mokerton/oslab_mobile
R=$REPO/results/siglip2
W0=~/siglip2_dfc/20261013_124213
W3=~/siglip2_dfc/g4_calib256_20261014
tr -d '\r' < $REPO/siglip2/hailo/s2_dfc_screen.py > ~/s2_dfc_screen.py
echo "$(date +%T) start commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -c core.autocrlf=true -C $REPO status --porcelain --untracked-files=no -- . ':(exclude)results' | wc -l)"
W4=~/siglip2_dfc/g4_lvl1_20261015; mkdir -p $W4
cp $W0/siglip2_b16_224.har $W3/calib_highres256_s0_nhwc.npy $W4/
tr -d '\r' < $REPO/siglip2/hailo/g4_D256L1_official_b9fc2.alls > $W4/g4_D256L1_official_b9fc2.alls
t0=$(date +%s)
( cd $W4 && $LOCK "DFC optimize D256 level1" -- hailo optimize siglip2_b16_224.har --hw-arch hailo10h --model-script g4_D256L1_official_b9fc2.alls \
    --calib-set-path calib_highres256_s0_nhwc.npy --output-har-path opt_D256L1_official_b9fc2.har > optimize_D256L1.log 2>&1 )
rc=$?; echo "$(date +%T) optimize D256L1 rc=$rc sec=$(( $(date +%s) - t0 ))"
mkdir -p $R/G4_dfc_screen/level1; cp $W4/optimize_D256L1.log $W4/g4_D256L1_official_b9fc2.alls $R/G4_dfc_screen/level1/ 2>/dev/null
grep -a -iE "error|gpu|level|warning" $W4/optimize_D256L1.log | tail -20 > $R/G4_dfc_screen/level1/optimize_summary.txt
if [ -f $W4/opt_D256L1_official_b9fc2.har ]; then
  sha256sum $W4/opt_D256L1_official_b9fc2.har > $R/G4_dfc_screen/level1/sha256.txt
  $LOCK "G4 screen D256L1" -- python ~/s2_dfc_screen.py --har $W4/opt_D256L1_official_b9fc2.har --tag D256L1_official_b9fc2 --k 200 --bs 8 \
      --out $R/G4_dfc_screen > $R/G4_dfc_screen/log_D256L1_official_b9fc2.txt 2>&1
  rc=$?; echo "$(date +%T) screen D256L1 rc=$rc $(grep '"hr_cos"' $R/G4_dfc_screen/log_D256L1_official_b9fc2.txt | tail -1)"
else
  echo "$(date +%T) optimize level1 produced no HAR (see optimize_summary.txt)"
fi
echo "G4_MORE2 DONE $(date +%T)"
