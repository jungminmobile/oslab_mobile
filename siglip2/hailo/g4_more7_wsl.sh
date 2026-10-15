#!/usr/bin/env bash
# g4_more7_wsl.sh — (10/15 밤) C256L1을 dev 전체(5,923장)로 평가: D256L1 dev 전체(results/siglip2/G4_dfc_full)와 같은 사진으로 비교 (D-032 판단 자료)
#  [0, 200)은 G4_dfc_screen의 C256L1 결과를 쓴다. 나머지는 200장 구간마다 데이터셋 3종씩 두 프로세스로 (start=800 OOM 대책, g4_more6과 같음).
set -uo pipefail
source ~/hailo10_venv/bin/activate
cd ~
LOCK="python3 /mnt/d/mokerton/locks/heavy_lock.py run --owner siglip2 --job"
REPO=/mnt/d/mokerton/oslab_mobile
R=$REPO/results/siglip2
W4=~/siglip2_dfc/g4_lvl1_20261015
tr -d '\r' < $REPO/siglip2/hailo/s2_dfc_screen.py > ~/s2_dfc_screen.py
echo "$(date +%T) start commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -c core.autocrlf=true -C $REPO status --porcelain --untracked-files=no -- . ':(exclude)results' | wc -l)"
run() {  # $1 start, $2 이름, 나머지: 데이터셋
  local st=$1 nm=$2; shift 2
  $LOCK "G4 full-dev C256L1 start=$st $nm" -- python ~/s2_dfc_screen.py --har $W4/opt_C256L1_official.har \
      --tag C256L1_official_s$st$nm --k 200 --start $st --bs 8 --sets "$@" --out $R/G4_dfc_full > $R/G4_dfc_full/log_C256L1_s$st$nm.txt 2>&1
  rc=$?; echo "$(date +%T) full-dev C256L1 start=$st $nm rc=$rc"
}
for st in 200 400 600 800; do
  run $st a cifar_A pets_dev flowers_dev
  run $st b cars_dev food_dev caltech_dev
done
run 1000 b caltech_dev
echo "G4_MORE7 DONE $(date +%T)"
