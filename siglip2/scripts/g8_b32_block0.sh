#!/usr/bin/env bash
# g8_b32_block0.sh — G8 4절: B/32 주 후보에서 블록 0 MatMul 일부를 FP32로 남겨 손실이 줄어드는지 (판정은 결과 전 고정, README 4절)
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2_b32
PY="$SIGLIP2_PYTHON"; ART=$SIGLIP2_ARTIFACTS; L=$ART/logs; STOP=/d/mokerton/locks/siglip2.STOP
R=../results/siglip2/G8_candidates/b32/block0; mkdir -p $R
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
chk() { [ -f $STOP ] && { echo "$(date +%T) STOP"; exit 3; }; return 0; }
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
FMR=siglip2_image_fp32_rcmaskbcmlp15at6u5-7at10u9_rothad.onnx
P=siglip2_image_int8_rcmaskbcmlp15at6u5-7at10u9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr
MAIN=${P}_all.onnx
B0=/trunk/blocks.0
declare -A EX=([X1]="$B0/mlp/fc2/" [X2]="$B0/mlp/fc1/,$B0/mlp/fc2/" [X3]="$B0/attn/,$B0/mlp/")
export RN50_RESULTS=$R/int8
for k in X1 X2 X3; do chk q$k
  "${LOCK[@]}" "SigLIP2 G8 B/32 block0 q$k" -- "$PY" -u s2_quantize.py --src $FMR --calib-source highres --calib-n 256 --op-types MatMul --method percentile \
    --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --exclude "${EX[$k]}" --tag b0$k --dump-ops --threads 4 --overwrite > $L/block0_q$k.log 2>&1
  echo "$(date +%T) q$k rc=$? $(grep -a 'quantize .* nodes' $L/block0_q$k.log | tail -1)"
done
export RN50_RESULTS=$R/screen; chk screen
"${LOCK[@]}" "SigLIP2 G8 B/32 block0 screen" -- "$PY" -u s2_screen.py --k 200 --tag block0 --models $MAIN ${P}_b0X1.onnx ${P}_b0X2.onnx ${P}_b0X3.onnx > $L/block0_screen.log 2>&1
echo "$(date +%T) screen rc=$?"; grep -a '"hr_cos"' $L/block0_screen.log | cut -c1-220
BEST=$("$PY" -c "
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8')]
rows=[r for r in rows if '_b0X' in r['model']]
print(max(rows,key=lambda r:r['hr_cos'])['model'])" $R/screen/screen.jsonl)
echo "best=$BEST" | tee $R/best.txt
export RN50_RESULTS=$R/dev_full; chk devfull
"${LOCK[@]}" "SigLIP2 G8 B/32 block0 devfull" -- "$PY" -u s2_paired_ci.py --models $MAIN $BEST > $L/block0_devfull.log 2>&1
echo "$(date +%T) devfull rc=$?"; grep -a "pair\|hr_dtop1" $L/block0_devfull.log | tail -3 | cut -c1-250
export RN50_RESULTS=$R/speed
while true; do chk speed
  "${LOCK[@]}" "SigLIP2 G8 B/32 block0 speed t4 (needs idle PC)" -- "C:/Program Files/Git/bin/bash.exe" scripts/_idle_then.sh "$PY" -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads 4 \
    --models main=$MAIN best=$BEST > $L/block0_speed.log 2>&1
  rc=$?; [ $rc -eq 75 ] && { sleep 180; continue; }; break; done
echo "$(date +%T) speed rc=$rc"; tail -4 $L/block0_speed.log
echo "$(date +%T) BLOCK0 DONE"
