#!/bin/bash
# usage: run_e12.sh <outname> <modelname> <onnx>
set -e
source ~/siglip2_env.sh
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
PY=$HOME/venvs/siglip2/bin/python
cd ~/oslab_e12
OUT="results/E12-1_ORT세션메모리설정/$1"; NM=$2; M=$3
mkdir -p "$OUT"
echo "[$(date +%T)] G1 start"
$PY -u e12/check_equal.py --model "$M" --combos C0 C1 C2 C3 C4 --n 200 --threads 4 --out "$OUT/equal.json"
echo "[$(date +%T)] speed s1"
$PY -u e12/bench_speed_session.py --session 1 --models "$NM=$M" --combos C0 C1 C2 C3 C4 --threads 1 2 4 --warmup 20 --n 200 --out "$OUT/speed.jsonl"
sleep 120
echo "[$(date +%T)] speed s2"
$PY -u e12/bench_speed_session.py --session 2 --models "$NM=$M" --combos C2 C1 C4 C3 C0 --threads 1 2 4 --warmup 20 --n 200 --out "$OUT/speed.jsonl"
sleep 120
echo "[$(date +%T)] speed s3"
$PY -u e12/bench_speed_session.py --session 3 --models "$NM=$M" --combos C4 C3 C2 C1 C0 --threads 1 2 4 --warmup 20 --n 200 --out "$OUT/speed.jsonl"
echo "[$(date +%T)] done"
