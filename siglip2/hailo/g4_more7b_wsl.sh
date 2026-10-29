#!/usr/bin/env bash
# g4_more7b_wsl.sh — g4_more7_wsl.sh(C256L1 dev 전체) 이어 하기. 10/15 21:40 개인연구 세션 요청으로 WSL을 잠시 내렸다가 다시 띄울 때 쓴다
#  (개인연구 E20이 메모리 2~3 GB를 쓰는 동안 WSL VM이 메모리를 잡고 있으면 폰 adb가 끊길 수 있음).
#  이미 끝난 조각(같은 로그에 "start=<st> <nm> rc=0")은 건너뛰고, 나머지만 같은 명령으로 돌린다. 로그는 g4_more7_wsl.log에 이어 쓴다.
set -uo pipefail
source ~/hailo10_venv/bin/activate
cd ~
LOCK="python3 /mnt/d/mokerton/locks/heavy_lock.py run --owner siglip2 --job"
REPO=/mnt/d/mokerton/oslab_mobile
R=$REPO/results/siglip2
W4=~/siglip2_dfc/g4_lvl1_20261015
LOG=/mnt/d/mokerton/artifacts/siglip2/logs/g4_more7_wsl.log
tr -d '\r' < $REPO/siglip2/hailo/s2_dfc_screen.py > ~/s2_dfc_screen.py
echo "$(date +%T) resume (g4_more7b) commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -c core.autocrlf=true -C $REPO status --porcelain --untracked-files=no -- . ':(exclude)results' | wc -l)"
run() {  # $1 start, $2 이름, 나머지: 데이터셋
  local st=$1 nm=$2; shift 2
  if grep -q "full-dev C256L1 start=$st $nm rc=0" $LOG; then echo "$(date +%T) skip start=$st $nm (done)"; return; fi
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
