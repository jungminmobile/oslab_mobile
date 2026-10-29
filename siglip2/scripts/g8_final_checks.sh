#!/usr/bin/env bash
# g8_final_checks.sh — 보고서 v4 검토에서 나온 추가 확인 (10/21)
#  E1: 최종 배포 파일(e84c3727…) dev 전체를 다시 돌려 사진별 예측 저장 → 오류 클래스 분석 (지금 보고서는 R0로만 분석)
#  E2: 준용이 Release에 올린 M2rot FP32·회전 행렬·캐시로 ① 이 PC에서 INT8 양자화만 → e84c3727과 비교 ② 회전 행렬·캐시로 M2rot FP32 export → 16ef696e과 비교
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
unset S2_MODEL_ID
export SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2
PY="$SIGLIP2_PYTHON"; ART=$SIGLIP2_ARTIFACTS; L=$ART/logs
R=../results/siglip2/G8_candidates/b16/final_checks; mkdir -p $R
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
sha() { "$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }
F=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
FMR=siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
[ "$(sha $ART/onnx/$F)" = e84c3727f51e3b97cd59d9ab9604e01acc78f81493926a0696fbf96dc9c60f9e ] || { echo "배포 파일 sha 불일치"; exit 2; }
# E1
export RN50_RESULTS=$R/E1_devfull
"${LOCK[@]}" "SigLIP2 G8 final E1 devfull" -- "$PY" -u s2_paired_ci.py --models $F > $L/final_E1.log 2>&1; echo "$(date +%T) E1 devfull rc=$?"; grep -a '"hr_dtop1"' $L/final_E1.log | cut -c1-250
export RN50_RESULTS=$R/E1_error_classes
"$PY" -u s2_error_classes.py --preds final=$R/E1_devfull/pred_${F%.onnx}.npz > $L/final_E1_err.log 2>&1; echo "E1 error rc=$?"
# E2 — 준용 Release 파일 받기 (별도 폴더 onnx_jy)
J=$ART/onnx_jy; mkdir -p $J
cd /d/mokerton/oslab_mobile && gh release download siglip2-m2rot-ort-20261019 --repo jungminmobile/oslab_mobile -D $J --skip-existing -p "$FMR" -p "rotq_rothad.npy" -p "regcache_*" -p "SHA256SUMS.txt" ; cd /d/mokerton/wt_g8/siglip2
for f in $FMR rotq_rothad.npy regcache_55at5-13at9_rothad_highres256_s0.npz regcache_uc_55at5-13at9_rothad_highres256_s0.npz; do echo "jy $f $(sha $J/$f | cut -c1-16)"; done | tee $R/E2_jy_files.txt
cat $J/SHA256SUMS.txt >> $R/E2_jy_files.txt
# ② 준용 FP32 → 이 PC 양자화
export RN50_ONNX=$J RN50_RESULTS=$R/E2_quant
"${LOCK[@]}" "SigLIP2 G8 final E2 quantize jy fp32" -- "$PY" -u s2_quantize.py --src $FMR --calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --tag all --dump-ops --threads 4 --overwrite > $L/final_E2q.log 2>&1
H=$(sha $J/$F); echo "E2 quantize: this PC INT8 from jy FP32 = ${H:0:16}  release e84c3727f51e3b97  $([ "$H" = e84c3727f51e3b97cd59d9ab9604e01acc78f81493926a0696fbf96dc9c60f9e ] && echo SAME || echo DIFF)" | tee -a $R/E2_result.txt
# ③ 준용 회전 행렬·캐시 → 이 PC에서 M2rot FP32 export (회전 적용은 torch 계산)
mkdir -p $ART/onnx_jy_export && cp $J/rotq_rothad.npy $J/regcache_*rothad*.npz $ART/onnx_jy_export/
export RN50_ONNX=$ART/onnx_jy_export RN50_RESULTS=$R/E2_export
"${LOCK[@]}" "SigLIP2 G8 final E2 export from jy caches" -- "$PY" -u s2_regcache.py export --graph mask --mask-form bcast --mask-site mlp --rot rothad --overwrite > $L/final_E2e.log 2>&1
H2=$(sha $ART/onnx_jy_export/$FMR); echo "E2 export: this PC FP32 from jy rotq+caches = ${H2:0:16}  jy 16ef696e8016f4d1  $([ "${H2:0:16}" = 16ef696e8016f4d1 ] && echo SAME || echo DIFF)" | tee -a $R/E2_result.txt
echo "$(date +%T) FINAL CHECKS DONE"
