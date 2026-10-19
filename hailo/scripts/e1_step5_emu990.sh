#!/bin/bash
# E1 5단계: 레벨0 기준값 확정 — Pets dev 전체(990장)로 에뮬레이터 정확도 재평가
# 사용: bash e1_step5_emu990.sh   (진행 상황: ~/hailo10_work/logs/step5_progress.txt)
source ~/hailo10_venv/bin/activate
W=~/hailo10_work; S="$(dirname "$0")"; P=$W/logs/step5_progress.txt
mkdir -p $W/logs; : > $P
RUNS=("fp opt fp" "base8 opt quantized" "c4 opt_c4 quantized" "official opt_official quantized" \
      "off8L2 opt_off8L2 quantized" "off8L3 opt_off8L3 quantized" "off8L4 opt_off8L4 quantized")
i=0
for r in "${RUNS[@]}"; do
  set -- $r; i=$((i+1))
  echo "$(date +%T) [$i/7] 시작: $1" >> $P
  cd $W/$2 && python3 "$S/emu_eval.py" --har $W/$2/rn50_image_fp32_q1_optimized.har --ctx $3 --data pets_dev --n 990 --tag lv0_990_$1 >> $W/logs/step5_$1.log 2>&1
  echo "$(date +%T) [$i/7] 끝: $1 (exit $?)  $(tail -n1 $W/results/emu_eval.jsonl | cut -c1-250)" >> $P
done
echo "$(date +%T) ALL DONE" >> $P
