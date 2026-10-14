#!/usr/bin/env bash
# g3_repro_venv.sh — ROADMAP G3 완료 기준 "새 venv에서 같은 정수 커널 정확도가 재현된다" 확인
# 1) 새 가상환경 D:\mokerton\venvs\siglip2_repro 에 siglip2/requirements.txt 그대로 설치 (pip 캐시 D:\mokerton\cache\pip)
# 2) 새 venv로 FP32 ONNX export → 원본 sha256과 비교 (RN50_ONNX=$ART/onnx_repro, 원본 폴더는 건드리지 않음)
# 3) 새 venv로 fc2L2(S8) 양자화를 같은 인자로 → 원본 sha256과 비교
# 4) 새 venv로 그 모델의 dev 전체 사진별 예측(s2_paired_ci) → 원본 G3 full의 pred npz와 사진별로 비교
# 단계마다 공용 잠금. 결과: results/siglip2/G3_repro/
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY0="$SIGLIP2_PYTHON"                                     # 잠금 도구·비교용 (원래 venv)
BASE='C:\Users\qaz43\AppData\Local\Programs\Python\Python311\python.exe'
VENV=D:/mokerton/venvs/siglip2_repro
PYR=$VENV/Scripts/python.exe
ART="$SIGLIP2_ARTIFACTS"
R=../results/siglip2/G3_repro; mkdir -p $R
LOCK=("$PY0" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
P=siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=all -- . | tr '\n' ' ')]"

# 1) 새 venv
if [ ! -f "$PYR" ]; then
  "${LOCK[@]}" "SigLIP2 repro venv install" -- "$BASE" -m venv "$VENV"
fi
"${LOCK[@]}" "SigLIP2 repro venv pip install" -- "$PYR" -m pip install --cache-dir 'D:\mokerton\cache\pip' \
    -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu > $ART/logs/repro_pip.log 2>&1
echo "$(date +%T) pip rc=$?"
"$PYR" -m pip freeze > $R/pip_freeze_repro.txt
"$PY0" -m pip freeze > $R/pip_freeze_original.txt
echo "freeze diff lines: $(diff $R/pip_freeze_original.txt $R/pip_freeze_repro.txt | grep -c '^[<>]')"

# 2) export
export RN50_ONNX="$ART/onnx_repro"
mkdir -p "$RN50_ONNX"
RN50_RESULTS=$R/export "${LOCK[@]}" "SigLIP2 repro export" -- "$PYR" -u s2_export_onnx.py --out $R/export --overwrite \
    > $ART/logs/repro_export.log 2>&1
echo "$(date +%T) export rc=$?"
# 3) 양자화 (g3_cumulative.sh의 fc2L2와 같은 인자)
RN50_RESULTS=$R "${LOCK[@]}" "SigLIP2 repro quantize fc2L2" -- "$PYR" -u s2_quantize.py --calib-source highres --calib-n 256 \
    --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --overwrite --dump-ops --tag fc2L2 \
    --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/" > $ART/logs/repro_quantize.log 2>&1
echo "$(date +%T) quantize rc=$?"
( cd "$ART" && sha256sum onnx/siglip2_image_fp32.onnx onnx_repro/siglip2_image_fp32.onnx onnx/siglip2_image_fp32_b1.onnx onnx_repro/siglip2_image_fp32_b1.onnx \
    onnx/${P}_fc2L2.onnx onnx_repro/${P}_fc2L2.onnx ) > $R/sha256.txt 2>&1
cat $R/sha256.txt
# 4) 사진별 예측 (새 venv, 새로 만든 모델)
RN50_RESULTS=$R/full "${LOCK[@]}" "SigLIP2 repro paired_ci fc2L2" -- "$PYR" -u s2_paired_ci.py --models "$ART/onnx_repro/${P}_fc2L2.onnx" \
    > $ART/logs/repro_paired_ci.log 2>&1
echo "$(date +%T) paired_ci rc=$?"
"$PY0" - > $R/pred_compare.txt <<'EOF'
import numpy as np, glob
a = np.load("../results/siglip2/G3_cumulative/full/pred_siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16_fc2L2.npz")
b = np.load(glob.glob("../results/siglip2/G3_repro/full/pred_*fc2L2.npz")[0])
for d in ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]:
    same = (a[f"{d}_pred"] == b[f"{d}_pred"]).mean(); dc = float(np.abs(a[f"{d}_cos"] - b[f"{d}_cos"]).max())
    print(f"{d}: n={len(a[d+'_pred'])} pred_same={same:.4f} max|dcos|={dc:.2e}")
EOF
cat $R/pred_compare.txt
echo "REPRO DONE $(date +%T)"
