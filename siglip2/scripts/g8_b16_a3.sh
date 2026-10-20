#!/usr/bin/env bash
# g8_b16_a3.sh — G8 A3: 최종 B/16(M2rot_ort 배포 파일)용 소규모 평가 묶음 (보고서 6.3) + 이 PC 기대값
#  묶음: $ART/share/g8_b16_final/ (dev 6종 × 200장, FP32·M2rot_ort, portable_eval.py, SHA256SUMS.txt)
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' OPENBLAS_NUM_THREADS=1
unset S2_MODEL_ID RN50_ONNX
export SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2
PY="$SIGLIP2_PYTHON"; ART=$SIGLIP2_ARTIFACTS; L=$ART/logs
R=../results/siglip2/G8_candidates/b16/A3_bundle; mkdir -p $R
F=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
B=$ART/share/g8_b16_final
"$PY" -u scripts/make_xplat_bundle.py --name g8_b16_final --k 200 --models siglip2_image_fp32.onnx $F > $L/a3_bundle.log 2>&1 || { tail $L/a3_bundle.log; exit 1; }
cp $B/SHA256SUMS.txt $R/SHA256SUMS.txt
grep -q "e84c3727f51e3b97cd59d9ab9604e01acc78f81493926a0696fbf96dc9c60f9e  models/$F" $B/SHA256SUMS.txt || { echo "묶음 안 모델 sha256이 배포 파일과 다름"; exit 2; }
# 묶음만으로 실행 (저장소 코드 없이 — 학술부·다른 기기와 같은 방식)
"$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "SigLIP2 G8 A3 portable_eval" -- \
  "$PY" -I $B/portable_eval.py --data $B/dev200 --out $R/out_this_pc --threads 4 --models fp32=$B/models/siglip2_image_fp32.onnx m2rot_ort=$B/models/$F > $L/a3_eval.log 2>&1
echo "rc=$?"; tail -5 $L/a3_eval.log
du -sh $B
