#!/usr/bin/env bash
# g8_repro_procedure_check.sh — 보고서 9.2 (가) 빠른 확인 절차를 깨끗한 폴더에서 그대로 (Release 내려받기 → portable_eval)
#  학술부가 하는 것과 같은 순서. 저장소 코드·데이터 없이 Release 파일만 씀
set -uo pipefail
W=D:/mokerton/repro_check_$(date +%m%d_%H%M); mkdir -p $W && cd $W
PY="$SIGLIP2_PYTHON"
echo "$(date +%T) 폴더 $W"
gh release download siglip2-xplat-20261015 --repo jungminmobile/oslab_mobile -D release -p "*.npz" -p "portable_eval.py" -p "siglip2_image_fp32.onnx" -p "SHA256SUMS.txt" || exit 1
gh release download siglip2-m2rot-ort-20261019 --repo jungminmobile/oslab_mobile -D release -p "siglip2_image_int8_*" --skip-existing || exit 1
ls -la release | tee files.txt
F=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
"$PY" -c "import hashlib,sys; [print(hashlib.sha256(open(p,'rb').read()).hexdigest()[:16], p) for p in sys.argv[1:]]" release/siglip2_image_fp32.onnx release/$F | tee -a files.txt
"$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "SigLIP2 repro procedure check" -- \
  "$PY" -I release/portable_eval.py --data release --out out_quick --threads 4 --models fp32=release/siglip2_image_fp32.onnx int8=release/$F > run.log 2>&1
echo "$(date +%T) rc=$?"; tail -4 run.log
"$PY" -c "import json; r=json.load(open('out_quick/result.json',encoding='utf-8')); m=r['models']['int8']; print('int8 hr_cos', m['hr_cos'], 'hr_dtop1', m['hr_dtop1'])"
