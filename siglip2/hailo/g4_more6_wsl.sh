#!/usr/bin/env bash
# g4_more6_wsl.sh — (10/15 저녁) D256L1 dev 전체 평가의 start=800 구간 재실행
#  g4_more4_wsl.sh의 start=800 구간이 마지막 데이터셋(caltech)에서 OOM으로 죽었다 (rc=247, dmesg: python anon-rss 11.7 GB).
#  같은 HAR·같은 구간을 데이터셋 3종씩 두 프로세스로 나눠 돌린다 (--sets, infer 호출 수를 줄여 메모리 대책). 계산 내용은 같다.
set -uo pipefail
source ~/hailo10_venv/bin/activate
cd ~
LOCK="python3 /mnt/d/mokerton/locks/heavy_lock.py run --owner siglip2 --job"
REPO=/mnt/d/mokerton/oslab_mobile
R=$REPO/results/siglip2
W4=~/siglip2_dfc/g4_lvl1_20261015
tr -d '\r' < $REPO/siglip2/hailo/s2_dfc_screen.py > ~/s2_dfc_screen.py
echo "$(date +%T) start commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -c core.autocrlf=true -C $REPO status --porcelain --untracked-files=no -- . ':(exclude)results' | wc -l)"
run() {  # $1 이름, 나머지: 데이터셋
  local nm=$1; shift
  $LOCK "G4 full-dev D256L1 start=800 $nm" -- python ~/s2_dfc_screen.py --har $W4/opt_D256L1_official_b9fc2.har \
      --tag D256L1_official_b9fc2_s800$nm --k 200 --start 800 --bs 8 --sets "$@" --out $R/G4_dfc_full > $R/G4_dfc_full/log_D256L1_s800$nm.txt 2>&1
  rc=$?; echo "$(date +%T) full-dev D256L1 start=800 $nm rc=$rc"
}
run a cifar_A pets_dev flowers_dev
run b cars_dev food_dev caltech_dev
echo "G4_MORE6 DONE $(date +%T)"
