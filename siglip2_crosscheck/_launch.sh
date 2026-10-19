#!/bin/bash
# 정민 노트북 WSL Ubuntu-22.04에서 준혁 교차 확인 실행 (Claude, 2026-10-17)
source ~/hailo10_venv/bin/activate
SRC="/mnt/c/study/3/2/2026 모바일 학술대회/oslab_mobile/siglip2_crosscheck"
W=~/siglip2_crosscheck
mkdir -p "$W"
cp -n "$SRC"/*.onnx "$SRC"/*.npy "$SRC"/*.npz "$SRC"/*.alls "$SRC"/*.py "$SRC"/*.sh "$SRC"/SHA256SUMS.txt "$W"/
cd "$W"
echo "$(date +%T) launch" > launch.out
COMPILE=1 bash run_crosscheck.sh > run_all.out 2>&1
echo "$(date +%T) exit $?" >> launch.out
