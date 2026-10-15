#!/usr/bin/env bash
# 텍스트 인코더 INT8 스모크 전체 (x86 PC). 저장소 루트에서 실행:
#   nohup bash textenc/run_smoke.sh > textenc_smoke.log 2>&1 &
# 필요: SIGLIP2_DATA, SIGLIP2_ARTIFACTS (siglip2/CLAUDE.md 10절), Pets·Food dev 캐시(s2_build_dev.py), Hugging Face 접속
set -euo pipefail
PY="${SIGLIP2_PYTHON:-python}"
OUT="${OUT:-results/textenc/smoke}"
TH="${TH:-4}"
A=(--threads "$TH" --out "$OUT")
echo "[1/6] export $(date)";   "$PY" textenc/te_export.py "${A[@]}"
echo "[2/6] massive $(date)";  "$PY" textenc/te_massive.py "${A[@]}"
echo "[3/6] quantize $(date)"
"$PY" textenc/te_quantize.py --variant A "${A[@]}"
"$PY" textenc/te_quantize.py --variant B --act-type u8 "${A[@]}"
"$PY" textenc/te_quantize.py --variant B --act-type s8 "${A[@]}"
MODELS=(siglip2_text_int8_dyn.onnx siglip2_text_int8_qdq_p99999_u8.onnx siglip2_text_int8_qdq_p99999_s8.onnx)
if "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$OUT/protect.json'))['nodes'] else 1)"; then
  for t in u8 s8; do
    "$PY" textenc/te_quantize.py --variant C --act-type $t --exclude-json "$OUT/protect.json" "${A[@]}"
    MODELS+=(siglip2_text_int8_qdq_p99999_${t}_protect.onnx)
  done
else
  echo "protect.json 비어 있음 → C 생략 (B와 같음)"
fi
"$PY" textenc/te_quantize.py --variant D "${A[@]}"
COMBO=()
for m in "${MODELS[@]}"; do
  "$PY" textenc/te_quantize.py --variant D --base "$m" "${A[@]}"
  COMBO+=("${m%.onnx}_embed8.onnx")
done
echo "[4/6] eval $(date)";     "$PY" textenc/te_eval.py --models "${MODELS[@]}" siglip2_text_int8_embed8.onnx "${COMBO[@]}" "${A[@]}"
echo "[5/6] speed $(date)"
BM=(fp32=siglip2_text_fp32.onnx D=siglip2_text_int8_embed8.onnx)
for m in "${MODELS[@]}"; do k="${m#siglip2_text_int8_}"; BM+=("${k%.onnx}=$m"); done
"$PY" textenc/te_bench.py --dir "$SIGLIP2_ARTIFACTS/textenc" --threads ${BENCH_THREADS:-1 2 4} --models "${BM[@]}" --out "$OUT/speed_x86"
echo "[6/6] done $(date)"
