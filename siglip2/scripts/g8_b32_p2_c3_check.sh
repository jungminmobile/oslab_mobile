#!/usr/bin/env bash
# g8_b32_p2_c3_check.sh — P2가 C3(enable_mem_pattern=False)에서도 출력이 같은지 + P2 입력 변형 8종 (2026-10-27, test 뒤)
#  ① E12-1 check_equal(정민 스크립트 그대로): P2 C0 vs C3, 고화질 5종 × 200장 비트 동일 확인
#  ② portable_eval --no-mem-pattern: dev200 1,200장 사진별 예측이 기본 세션 결과(b32/p2/xplat_i7)와 같은지
#  ③ s2_input_shift.py: P2 입력 변형 8종 (O-AB 때와 같은 명령, 보고서 6.2 견고성)
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2_b32
PY="$SIGLIP2_PYTHON"; B=$SIGLIP2_ARTIFACTS/onnx
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
P2=siglip2_image_int8_g6b_b32_m2rot_o1b_e14P2_fastgelu_patchint8.onnx
R=D:/mokerton/wt_g8/results/siglip2/G8_candidates/b32/p2
SH=$SIGLIP2_ARTIFACTS/share/20261023_b32_submit
echo "$(date +%T) P2 C3 CHECK start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- . | wc -l)"
mkdir -p $R/c3_equal $R/xplat_i7_c3 $R/input_shift
( cd D:/mokerton/wt_e12 && "${LOCK[@]}" "SigLIP2 B32 P2 C3 equal" -- "$PY" -u e12/check_equal.py --model "$B/$P2" --combos C0 C3 --n 200 --threads 4 --out $R/c3_equal/equal.json ) > $R/c3_equal/run.log 2>&1
echo "$(date +%T) equal rc=$?"; tail -4 $R/c3_equal/run.log
"${LOCK[@]}" "SigLIP2 B32 P2 portable C3" -- "$PY" -I scripts/portable_eval.py --data $SH/dev200 --out $R/xplat_i7_c3 --threads 4 --no-mem-pattern \
  --models fp32=$B/siglip2_image_fp32.onnx p2=$B/$P2 > $R/xplat_i7_c3/run.log 2>&1
echo "$(date +%T) portable rc=$?"
"$PY" -c "
import numpy as np
for m in ('fp32','p2'):
    a=np.load('$R/xplat_i7/pred_'+m+'.npz'); b=np.load('$R/xplat_i7_c3/pred_'+m+'.npz')
    print(m, 'C3 = 기본 세션 (사진별 예측·cos 비트 동일):', all(np.array_equal(a[k], b[k]) for k in a.files))
" | tee $R/xplat_i7_c3/compare_default.txt
FP=siglip2_image_fp32.onnx; OAB=siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
RN50_RESULTS=$R/input_shift "${LOCK[@]}" "SigLIP2 B32 P2 input shift" -- "$PY" -u s2_input_shift.py --k 200 --models fp32=$FP oab=$OAB p2=$P2 > $R/input_shift/run.log 2>&1
echo "$(date +%T) input_shift rc=$?"; tail -3 $R/input_shift/run.log
echo "$(date +%T) P2 C3 CHECK DONE dirty_end=$(git status --porcelain -- . | wc -l)"
