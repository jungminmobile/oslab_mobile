#!/usr/bin/env bash
# g8_b32_p2_eval.sh — P2(e465d10d) 교체 확인: Release 파일 sha, dev 전체 O-AB vs P2 짝 비교, portable_eval(dev200) i7 결과 (2026-10-27)
#  정확도만이라 PC를 써도 값은 같다. 무거운 단계는 공용 잠금 안
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2_b32
PY="$SIGLIP2_PYTHON"; B=$SIGLIP2_ARTIFACTS/onnx
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
OAB=siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
P2=siglip2_image_int8_g6b_b32_m2rot_o1b_e14P2_fastgelu_patchint8.onnx
WANT=e465d10da92cf0f830b95d29d72cf3251ea34da90f54b6b5c1f3e16510efd6b2
R=D:/mokerton/wt_g8/results/siglip2/G8_candidates/b32/p2
REL=$SIGLIP2_ARTIFACTS/release_p2_20261027
echo "$(date +%T) P2 EVAL start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- . | wc -l)"
mkdir -p $R/dev_full $R/xplat_i7 $REL
# ① Release 파일 sha (준용 Release siglip2-p2-20261027) = 이 PC에서 만든 파일
[ -f $REL/e14_4_P2_patch_int8.onnx ] || gh release download siglip2-p2-20261027 --repo jungminmobile/oslab_mobile -p e14_4_P2_patch_int8.onnx -D $REL
HR=$(sha256sum $REL/e14_4_P2_patch_int8.onnx | cut -d' ' -f1); HL=$(sha256sum $B/$P2 | cut -d' ' -f1)
echo "release=${HR:0:16} local=${HL:0:16} want=${WANT:0:16}" | tee $R/sha_check.txt
[ "$HR" = "$WANT" ] && [ "$HL" = "$WANT" ] || { echo "SHA256 MISMATCH — 멈춤"; exit 2; }
# ② dev 전체: O-AB와 P2를 같은 실행에서 (FP32 대비 + 두 모델 짝 차이)
RN50_RESULTS=$R/dev_full "${LOCK[@]}" "SigLIP2 B32 P2 dev full" -- "$PY" -u s2_paired_ci.py --models $OAB $P2 > $R/dev_full/run.log 2>&1
echo "$(date +%T) dev rc=$?"; grep -a '"hr_dtop1"\|"pair"' $R/dev_full/paired_ci.jsonl | cut -c1-260
# ③ portable_eval (Release 묶음 dev200, 10/23과 같은 명령, 모델 P2만 추가)
SH=$SIGLIP2_ARTIFACTS/share/20261023_b32_submit
"${LOCK[@]}" "SigLIP2 B32 P2 portable_eval" -- "$PY" -I $SH/portable_eval.py --data $SH/dev200 --out $R/xplat_i7 --threads 4 \
  --models fp32=$B/siglip2_image_fp32.onnx oab=$B/$OAB p2=$B/$P2 > $R/xplat_i7/run.log 2>&1
echo "$(date +%T) portable rc=$?"; tail -3 $R/xplat_i7/run.log
echo "$(date +%T) P2 EVAL DONE dirty_end=$(git status --porcelain -- . | wc -l)"
