#!/bin/bash
# E1 7단계: layer4(·layer3) 잔차 덧셈을 16비트로 두면 붕괴가 사라지는지 (레벨 0, 노트북, Pets 990장)
set -u
S="$(cd "$(dirname "$0")" && pwd)"; W=~/hailo10_work; P=$W/logs/step7_progress.txt
source ~/hailo10_venv/bin/activate; mkdir -p $W/logs; echo "=== $(date +%T) 새 실행: ${RUN_LIST:-기본}" >> $P
for n in ${RUN_LIST:-t1_off8L4add16 t3_c3eqadd16 t2_off8L4out16}; do
  echo "$(date +%T) $n 양자화 시작" >> $P
  rm -rf $W/opt_$n; mkdir -p $W/opt_$n; cd $W/opt_$n
  hailo optimize $W/parse/rn50.har --hw-arch hailo10h --calib-set-path $W/calib/calib_rn50_highres64_s0_nhwc_u8.npy \
     --model-script "$S/rn50_$n.alls" > $W/logs/rn50_${n}_optimize.log 2>&1
  rc=$?; echo "$(date +%T) $n 양자화 exit=$rc" >> $P
  if [ $rc -ne 0 ]; then grep -iE "error|exception" $W/logs/rn50_${n}_optimize.log | tail -3 | cut -c1-200 >> $P; continue; fi
  python3 "$S/emu_eval.py" --har $(ls $W/opt_$n/*_optimized.har | head -1) --ctx quantized --data pets_dev --n 990 --tag lv0_990_$n > $W/logs/step7_$n.log 2>&1
  echo "$(date +%T) $n 끝: $(tail -n1 $W/results/emu_eval.jsonl | grep -o '"top1.*cos_p1": [0-9.]*')" >> $P
done
echo "$(date +%T) ALL DONE" >> $P
