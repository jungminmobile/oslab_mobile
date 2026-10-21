#!/usr/bin/env bash
# g8_rn50_clean.sh — RN50 B2를 깨끗한 상태로 다시 잰다 (정민 Release rn50-b2-xcheck-20261021, 코드 511de3c, sha256 26개 확인)
#  ① 정확도: 정민 코드 그대로 (evaluate.py --exec int + bootstrap.py, dev 6종) — 묶음 폴더에서, 저장소는 건드리지 않음
#  ② 지연·메모리: 후보 3개(RN50·B/16·B/32) 6개 모델을 같은 실행에서 번갈아 (cross_latency.py, 고성능, PC 조용할 때만)
set -uo pipefail
B=D:/mokerton/artifacts/rn50_g8/dl
OUT=D:/mokerton/artifacts/rn50_g8/results
R=D:/mokerton/wt_g8/results/siglip2/G8_candidates/rn50_clean   # Windows python에 넘기므로 D:/ 형식 (MSYS 경로 변환 끔)
PY="$SIGLIP2_PYTHON"; LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8
mkdir -p $OUT $R
echo "$(date +%T) start repo=$(git -C D:/mokerton/wt_g8 rev-parse --short HEAD) dirty=$(git -C D:/mokerton/wt_g8 status --porcelain -- siglip2 | wc -l)"
# 묶음 zip에 rn50/prompts/cifar100_templates.json이 빠져 있어 정민 브랜치(=main과 같음)에서 넣었다 (10/22 첫 실행 FileNotFoundError)
[ -f $B/code/rn50/prompts/cifar100_templates.json ] || { echo "prompts 없음"; exit 2; }
B2=rn50_image_int8_q1_percentile99999_highres256_s0_B2
HR="pets_dev flowers_dev cars_dev food_dev caltech_dev"
cd $B/code/rn50
RN50_ONNX=$B/onnx RN50_CACHE=$B/cache RN50_RESULTS=$OUT "${LOCK[@]}" "RN50 B2 clean accuracy" -- "$PY" evaluate.py --exec int --threads 4 --save-pred --data cifar_A $HR --models fp32 $B2.onnx > $OUT/eval.log 2>&1
rc=$?; echo "$(date +%T) evaluate rc=$rc"; tail -3 $OUT/eval.log; [ $rc -eq 0 ] || exit 1
RN50_ONNX=$B/onnx RN50_CACHE=$B/cache RN50_RESULTS=$OUT "$PY" bootstrap.py --a fp32 --b $B2 --data $HR > $OUT/boot.log 2>&1
rc=$?; echo "$(date +%T) bootstrap rc=$rc"; tail -3 $OUT/boot.log; [ $rc -eq 0 ] || exit 1
cp $OUT/*.jsonl $OUT/*.log $R/ 2>/dev/null
cd /d/mokerton/wt_g8/siglip2
A=D:/mokerton/artifacts
while true; do
  "${LOCK[@]}" "Candidates 3 latency+mem (needs idle PC)" -- "C:/Program Files/Git/bin/bash.exe" scripts/_idle_then.sh "$PY" -u scripts/cross_latency.py --out $R/latency --mem \
    --models rn50_fp32=$B/onnx/rn50_image_fp32_q1.onnx rn50_b2=$B/onnx/$B2.onnx \
             b16_fp32=$A/siglip2/onnx/siglip2_image_fp32.onnx b16_m2rot=$A/siglip2/onnx/siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx \
             b32_fp32=$A/siglip2_b32/onnx/siglip2_image_fp32.onnx b32_m2rot=$A/siglip2_b32/onnx/siglip2_image_int8_rcmaskbcmlp15at6u5-7at10u9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx > $R/latency_run.log 2>&1
  rc=$?; echo "$(date +%T) latency rc=$rc"; [ $rc -eq 75 ] && { sleep 180; continue; }; break
done
tail -40 $R/latency_run.log | grep -E "session|rn50|b16|b32" | tail -12
echo "$(date +%T) RN50 CLEAN DONE"
