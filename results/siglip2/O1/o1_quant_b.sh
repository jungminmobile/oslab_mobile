#!/usr/bin/env bash
# O1 2단계: O-B FP32 그래프 정리(s2_ortfuse, FP32 동치 64장 확인) → O-B·O-AB 양자화, B16·B32. O-A는 깨끗한 커밋에서 다시 만들어 sha 비교.
# 사용 (아무 곳):  bash results/siglip2/O1/o1_quant_b.sh
set -euo pipefail
cd "$(dirname "$0")/../../../siglip2"
source ~/siglip2_env.sh
export OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
R=../results/siglip2/O1
A=$SIGLIP2_ARTIFACTS
PY="nice -n 10 $SIGLIP2_PYTHON -u"
COM="--calib-source highres --calib-n 256 --seed 0 --method percentile --percentile 99.999 --act-type u8 --reduce-range --op-types MatMul --hist-chunk 16 --tag all --dump-ops --threads 4"
NPY32="--calib-npy $HOME/siglip2_work/g6pre/cache_zoo256/calib256_s0_x.npy"
echo "commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
for s in rcmaskbcmlp55at5-13at9_rothad g6b_b32_m2rot; do
  RN50_RESULTS=$R/ortfuse $PY s2_ortfuse.py --src siglip2_image_fp32_$s.onnx --skip-ln --fold-shape --check 64
done
# FP32 동치 기준: cos 최소 ≥ 0.99999 (배치 1·4), 아니면 멈춤
$SIGLIP2_PYTHON - $R/ortfuse/ortfuse_log.jsonl <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])][-2:]
for r in rows:
    e = r["equiv"]; ok = min(e["b1"]["cos_min"], e["b4"]["cos_min"]) >= 0.99999 and not r["dirty"]
    print(r["dst"], e["b1"], e["b4"], "dirty", r["dirty"], "OK" if ok else "FAIL")
    assert ok
PY
for s in "rcmaskbcmlp55at5-13at9_rothad" "g6b_b32_m2rot"; do
  X=""; [ $s = g6b_b32_m2rot ] && X="$NPY32"
  RN50_RESULTS=$R/quant $PY s2_quantize.py --src siglip2_image_fp32_$s.onnx $X $COM --no-matmul-out-q --overwrite
  RN50_RESULTS=$R/quant $PY s2_quantize.py --src siglip2_image_fp32_${s}_o1b.onnx $X $COM
  RN50_RESULTS=$R/quant $PY s2_quantize.py --src siglip2_image_fp32_${s}_o1b.onnx $X $COM --no-matmul-out-q
done
sha256sum $A/onnx/siglip2_image_int8_*_moq0_all.onnx $A/onnx/siglip2_image_int8_*_o1b_*_all.onnx $A/onnx/siglip2_image_fp32_*_o1b.onnx | tee $R/quant/sha256.txt
echo O1_QUANT_B_DONE
