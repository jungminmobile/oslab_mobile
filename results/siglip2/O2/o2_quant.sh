#!/usr/bin/env bash
# O2 3단계: B16 통일 FP32(siglip2_image_fp32_o2_b16_m2rotu.onnx)를 기존 M2rot_ort 설정과 O-AB 설정으로 양자화. 보정은 기존 B16과 같은 calib_images(highres 256, 시드 0).
# 사용 (아무 곳):  bash results/siglip2/O2/o2_quant.sh
set -euo pipefail
cd "$(dirname "$0")/../../../siglip2"
source ~/siglip2_env.sh
export OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
R=../results/siglip2/O2
PY="nice -n 10 $SIGLIP2_PYTHON -u"
COM="--calib-source highres --calib-n 256 --seed 0 --method percentile --percentile 99.999 --act-type u8 --reduce-range --op-types MatMul --hist-chunk 16 --tag all --dump-ops --threads 4"
echo "commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
RN50_RESULTS=$R/quant $PY s2_quantize.py --src siglip2_image_fp32_o2_b16_m2rotu.onnx $COM
RN50_RESULTS=$R/ortfuse $PY s2_ortfuse.py --src siglip2_image_fp32_o2_b16_m2rotu.onnx --skip-ln --fold-shape --check 64
$SIGLIP2_PYTHON - $R/ortfuse/ortfuse_log.jsonl <<'PY'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])][-1]; e = r["equiv"]
ok = min(e["b1"]["cos_min"], e["b4"]["cos_min"]) >= 0.99999 and not r["dirty"]
print(r["dst"], r["skip_ln"], r["fold_shape"]["reshape_folded"], e["b1"], "OK" if ok else "FAIL"); assert ok
PY
RN50_RESULTS=$R/quant $PY s2_quantize.py --src siglip2_image_fp32_o2_b16_m2rotu_o1b.onnx $COM --no-matmul-out-q
sha256sum $SIGLIP2_ARTIFACTS/onnx/siglip2_image_int8_o2_b16_m2rotu*_all.onnx $SIGLIP2_ARTIFACTS/onnx/siglip2_image_fp32_o2_b16_m2rotu*.onnx | tee $R/quant/sha256.txt
echo O2_QUANT_DONE
