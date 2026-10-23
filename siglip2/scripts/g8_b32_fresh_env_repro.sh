#!/usr/bin/env bash
# g8_b32_fresh_env_repro.sh — 보고서 9장·README v4 명령을 새 clone·새 가상환경에서 그대로 (2026-10-23, 학술부가 할 순서)
#  ① GitHub에서 새로 clone (브랜치 siglip2/report-b32) ② 시스템 Python으로 새 venv + requirements 설치 ③ Release 내려받기·sha256
#  ④ (가) 빠른 확인 portable_eval → 이 PC 기준값과 사진별 비교 ⑤ (나) dev 캐시를 새 산출물 폴더에 처음부터 만들고 dev 전체 → 기존 결과와 사진별 비교
#  절약을 위해 원본 데이터(SIGLIP2_DATA)와 HF 가중치 캐시는 이 PC 것을 다시 쓴다(내려받기만 생략, 전처리·캐시·평가는 새로). 무거운 단계는 공용 잠금 안
set -uo pipefail
W=D:/mokerton/repro_fresh_b32_$(date +%m%d_%H%M); mkdir -p $W && cd $W
SYSPY=C:/Users/qaz43/AppData/Local/Programs/Python/Python311/python.exe
LOCKPY="$SIGLIP2_PYTHON"; LOCK=("$LOCKPY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
OAB=siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
REF=D:/mokerton/wt_g8/results/siglip2/G8_candidates/b32
echo "$(date +%T) 폴더 $W"
gh repo clone jungminmobile/oslab_mobile oslab_mobile -- --branch siglip2/report-b32 -q || exit 1
echo "$(date +%T) clone $(git -C oslab_mobile rev-parse --short HEAD)"
"$SYSPY" -m venv venv || exit 1; VPY=$W/venv/Scripts/python.exe
"$VPY" -m pip install -q --upgrade pip > pip.log 2>&1
"$VPY" -m pip install -r oslab_mobile/siglip2/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu >> pip.log 2>&1; echo "$(date +%T) pip rc=$?"
"$VPY" -m pip freeze > freeze.txt; "$VPY" --version
gh release download siglip2-b32-submit-20261023 --repo jungminmobile/oslab_mobile -D release || exit 1
(cd release && sha256sum -c SHA256SUMS.txt > ../sha_check.txt 2>&1); echo "$(date +%T) sha256 rc=$?"; tail -2 sha_check.txt
# (가) 빠른 확인
"${LOCK[@]}" "SigLIP2 B32 fresh env quick check" -- "$VPY" -I release/portable_eval.py --data release --out out_quick --threads 4 \
  --models fp32=release/siglip2_image_fp32.onnx oab=release/$OAB > quick.log 2>&1
echo "$(date +%T) quick rc=$?"
# (나) dev 처음부터
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_DATA=D:/mokerton/data SIGLIP2_ARTIFACTS=$W/art HF_HOME=D:/mokerton/cache/huggingface \
       HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 PYTHONIOENCODING=utf-8 MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
mkdir -p $W/art/onnx && cp release/*.onnx $W/art/onnx/
cd oslab_mobile/siglip2
"${LOCK[@]}" "SigLIP2 B32 fresh env build dev (~1h)" -- "$VPY" -u s2_build_dev.py all --threads 4 > $W/build_dev.log 2>&1
echo "$(date +%T) build_dev rc=$?"; tail -3 $W/build_dev.log
RN50_RESULTS=$W/art/results_repro "${LOCK[@]}" "SigLIP2 B32 fresh env dev full" -- "$VPY" -u s2_paired_ci.py --models $OAB > $W/devfull.log 2>&1
echo "$(date +%T) devfull rc=$?"; grep -a '"hr_dtop1"' $W/devfull.log | tail -2
cd $W
"$VPY" - <<EOF
import json, numpy as np
q = json.load(open("out_quick/result.json", encoding="utf-8"))["models"]; r = json.load(open("release/result_i7-1165G7.json", encoding="utf-8"))["models"]
print("quick oab", q["oab"]["hr_dtop1"], q["oab"]["hr_cos"], "| ref", r["oab"]["hr_dtop1"], r["oab"]["hr_cos"])
a = np.load("out_quick/pred_oab.npz"); b = np.load("$REF/xplat/i7-1165G7/pred_oab.npz")
print("quick pred identical:", all(np.array_equal(a[k], b[k]) for k in a.files if k.endswith("_pred")))
A = np.load("art/results_repro/pred_$OAB".replace(".onnx", ".npz")); B = np.load("$REF/O_AB/dev_full/pred_$OAB".replace(".onnx", ".npz"))
print("dev full pred identical:", {k: bool(np.array_equal(A[k], B[k])) for k in A.files if k.endswith("_pred")})
EOF
echo "$(date +%T) FRESH ENV REPRO DONE ($W)"
