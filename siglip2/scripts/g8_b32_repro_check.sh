#!/usr/bin/env bash
# g8_b32_repro_check.sh — 제출 모델(B/32 O-AB) 빠른 재현 절차를 깨끗한 폴더에서 그대로 (Release 내려받기 → sha256 → portable_eval)
#  학술부가 하는 순서. 저장소 코드·데이터 없이 Release 파일만 씀 (B/16 E3 g8_repro_procedure_check.sh와 같은 방식)
#  기대: 이 PC 기준값 result_i7-1165G7.json과 사진별 예측까지 같음 (같은 PC·같은 파일)
set -uo pipefail
W=D:/mokerton/repro_check_b32_$(date +%m%d_%H%M); mkdir -p $W && cd $W
PY="$SIGLIP2_PYTHON"
echo "$(date +%T) 폴더 $W"
gh release download siglip2-b32-submit-20261023 --repo jungminmobile/oslab_mobile -D release || exit 1
cd release && sha256sum -c SHA256SUMS.txt > ../sha_check.txt 2>&1; rc=$?; cd ..; echo "$(date +%T) sha256 check rc=$rc"; tail -3 sha_check.txt
[ $rc -eq 0 ] || exit 2
OAB=siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
M2=siglip2_image_int8_rcmaskbcmlp15at6u5-7at10u9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
"$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "SigLIP2 B32 repro procedure check" -- \
  "$PY" -I release/portable_eval.py --data release --out out_quick --threads 4 \
    --models fp32=release/siglip2_image_fp32.onnx oab=release/$OAB m2rot=release/$M2 > run.log 2>&1
echo "$(date +%T) eval rc=$?"; tail -3 run.log
"$PY" - <<'EOF'
import json, numpy as np
a = json.load(open("out_quick/result.json", encoding="utf-8"))["models"]
b = json.load(open("release/result_i7-1165G7.json", encoding="utf-8"))["models"]
for m in a:
    print(m, "hr_dtop1", a[m]["hr_dtop1"], "vs 기준", b[m]["hr_dtop1"], "hr_cos", a[m]["hr_cos"], "vs", b[m]["hr_cos"])
    z = np.load(f"out_quick/pred_{m}.npz")
    print("   ", {k: int(z[k].shape[0]) for k in z.files if k.endswith("_pred")})
EOF
echo "$(date +%T) REPRO CHECK DONE ($W)"
