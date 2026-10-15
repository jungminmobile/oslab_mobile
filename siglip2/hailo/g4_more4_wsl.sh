#!/usr/bin/env bash
# g4_more4_wsl.sh — (10/15 오전) 결정 지원용 DFC 실험
#  1) C + 보정 256장 + 최적화 수준 1: D256L1(cos 0.967)과 같은 조건에서 C(HEF 97.1 MB·16 컨텍스트)를 비교 → D-032 판단 자료
#  2) D256L1을 dev 전체(5,923장)로 평가: 200장 표본보다 좁은 구간으로 ORT 후보와 비교 (results/siglip2/G4_dfc_full)
set -uo pipefail
source ~/hailo10_venv/bin/activate
cd ~
LOCK="python3 /mnt/d/mokerton/locks/heavy_lock.py run --owner siglip2 --job"
REPO=/mnt/d/mokerton/oslab_mobile
R=$REPO/results/siglip2
W0=~/siglip2_dfc/20261013_124213
W3=~/siglip2_dfc/g4_calib256_20261014
W4=~/siglip2_dfc/g4_lvl1_20261015
tr -d '\r' < $REPO/siglip2/hailo/s2_dfc_screen.py > ~/s2_dfc_screen.py
echo "$(date +%T) start commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -c core.autocrlf=true -C $REPO status --porcelain --untracked-files=no -- . ':(exclude)results' | wc -l)"
tr -d '\r' < $REPO/siglip2/hailo/g4_C256L1_official.alls > $W4/g4_C256L1_official.alls
t0=$(date +%s)
( cd $W4 && $LOCK "DFC optimize C256L1" -- hailo optimize siglip2_b16_224.har --hw-arch hailo10h --model-script g4_C256L1_official.alls \
    --calib-set-path calib_highres256_s0_nhwc.npy --output-har-path opt_C256L1_official.har > optimize_C256L1.log 2>&1 )
rc=$?; echo "$(date +%T) optimize C256L1 rc=$rc sec=$(( $(date +%s) - t0 ))"
cp $W4/optimize_C256L1.log $W4/g4_C256L1_official.alls $R/G4_dfc_screen/level1/ 2>/dev/null
if [ -f $W4/opt_C256L1_official.har ]; then
  sha256sum $W4/opt_C256L1_official.har >> $R/G4_dfc_screen/level1/sha256.txt
  $LOCK "G4 screen C256L1" -- python ~/s2_dfc_screen.py --har $W4/opt_C256L1_official.har --tag C256L1_official --k 200 --bs 8 \
      --out $R/G4_dfc_screen > $R/G4_dfc_screen/log_C256L1_official.txt 2>&1
  rc=$?; echo "$(date +%T) screen C256L1 rc=$rc $(grep '"hr_cos"' $R/G4_dfc_screen/log_C256L1_official.txt | tail -1)"
fi
mkdir -p $R/G4_dfc_full
# dev 전체를 200장 구간으로 나눠 따로 실행 (메모리 대책). [0, 200)은 G4_dfc_screen의 D256L1 결과를 그대로 쓴다
for st in 200 400 600 800 1000; do
  $LOCK "G4 full-dev D256L1 start=$st" -- python ~/s2_dfc_screen.py --har $W4/opt_D256L1_official_b9fc2.har --tag D256L1_official_b9fc2_s$st       --k 200 --start $st --bs 8 --out $R/G4_dfc_full > $R/G4_dfc_full/log_D256L1_s$st.txt 2>&1
  rc=$?; echo "$(date +%T) full-dev D256L1 start=$st rc=$rc"
done
echo "G4_MORE4 DONE $(date +%T)"
