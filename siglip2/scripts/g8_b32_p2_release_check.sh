#!/usr/bin/env bash
# g8_b32_p2_release_check.sh — 최종 제출 후보 Release(siglip2-b32-p2-submit-20261027)만으로 새 폴더에서 빠른 확인 (2026-10-27)
#  보고서 9.2 (가)·Release 설명의 명령 그대로: 받기 → sha256 확인 → portable_eval --no-mem-pattern → Release 안 기준 결과와 사진별 비교
set -uo pipefail
W=D:/mokerton/repro_p2_release_$(date +%m%d_%H%M); mkdir -p $W && cd $W
PY="$SIGLIP2_PYTHON"; LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
OUT=D:/mokerton/wt_g8/results/siglip2/G8_candidates/b32/p2/release_check
echo "$(date +%T) 폴더 $W"
gh release download siglip2-b32-p2-submit-20261027 --repo jungminmobile/oslab_mobile -D release || exit 1
(cd release && sha256sum -c SHA256SUMS.txt) > sha_check.txt 2>&1; echo "$(date +%T) sha256 rc=$? $(grep -c ': OK' sha_check.txt) OK"
"${LOCK[@]}" "SigLIP2 B32 P2 release check" -- "$PY" -I release/portable_eval.py --data release --out out_quick --threads 4 --no-mem-pattern \
  --models fp32=release/siglip2_image_fp32.onnx p2=release/e14_4_P2_patch_int8.onnx > quick.log 2>&1
echo "$(date +%T) quick rc=$?"
"$PY" - <<PYEOF
import json, numpy as np
q = json.load(open("out_quick/result.json", encoding="utf-8")); r = json.load(open("release/result_i7-1165G7_p2.json", encoding="utf-8"))
print("hr_dtop1 / cos:", q["models"]["p2"]["hr_dtop1"], q["models"]["p2"]["hr_cos"], "| 기준", r["models"]["p2"]["hr_dtop1"], r["models"]["p2"]["hr_cos"], "| mem_pattern 기록:", q.get("mem_pattern"))
for m in ("fp32", "p2"):
    a = np.load(f"out_quick/pred_{m}.npz"); b = np.load(f"release/pred_i7-1165G7_{m}.npz")
    print(m, "사진별 예측·cos 비트 동일:", all(np.array_equal(a[k], b[k]) for k in a.files))
PYEOF
mkdir -p $OUT && cp sha_check.txt quick.log out_quick/result.json $OUT/ && echo "$(date +%T) RELEASE CHECK DONE ($W)"
