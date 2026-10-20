#!/usr/bin/env bash
# g1_dfc.sh — G1 컴파일 게이트: SigLIP2 이미지 인코더 float ONNX → DFC parse → optimize(기본, 보정 64장) → compile (Hailo-10H)
# WSL Ubuntu-22.04 안에서 실행 (D-011):
#   wsl -d Ubuntu-22.04 -e bash -lc "bash /mnt/d/mokerton/oslab_mobile/siglip2/hailo/g1_dfc.sh"
# 입력: $ART/onnx/siglip2_image_fp32_b1.onnx (1×3×224×224 고정, D-020), $ART/dfc/calib_highres64_s0_nhwc.npy (s2_dfc_calib.py)
# 출력: ~/siglip2_dfc/<날짜>/ 에서 작업 → $ART/dfc/<날짜>/ 로 HAR·HEF·로그 복사, 요약은 results/siglip2/G1_dfc_gate/
# 모델 스크립트(.alls) 없이 기본값(S-C의 출발점). GPU가 없어 최적화 레벨은 DFC가 자동으로 낮춘다(로그에 기록됨).
set -uo pipefail
ART=/mnt/d/mokerton/artifacts/siglip2
REPO=/mnt/d/mokerton/oslab_mobile
TAG=$(date +%Y%m%d_%H%M%S)
W=~/siglip2_dfc/$TAG
OUT=$ART/dfc/$TAG
RES=$REPO/results/siglip2/G1_dfc_gate
NET=siglip2_b16_224
mkdir -p "$W" "$OUT" "$RES"
source ~/hailo10_venv/bin/activate
cp "$ART/onnx/siglip2_image_fp32_b1.onnx" "$ART/dfc/calib_highres64_s0_nhwc.npy" "$W/"
cd "$W"
{
  echo "tag=$TAG"; echo "commit=$(git -C $REPO rev-parse --short HEAD)"; hailo --version 2>&1 | tail -1
  python --version; uname -r; nproc; free -g | head -2
  sha256sum siglip2_image_fp32_b1.onnx calib_highres64_s0_nhwc.npy
} > env.txt 2>&1

step() {  # step <이름> <명령...> : 시간·종료 코드를 기록하고 실패해도 다음 정보는 남긴다
  local name=$1; shift
  local t0=$(date +%s)
  "$@" > "$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc sec=$(( $(date +%s) - t0 ))" | tee -a steps.txt
  return $rc
}

step parse hailo parser onnx siglip2_image_fp32_b1.onnx --hw-arch hailo10h --net-name $NET --har-path $NET.har -y \
     --parsing-report-path parsing_report.json \
&& step optimize hailo optimize $NET.har --hw-arch hailo10h --calib-set-path calib_highres64_s0_nhwc.npy \
     --output-har-path ${NET}_optimized.har \
&& step compile hailo compiler ${NET}_optimized.har --hw-arch hailo10h --output-dir . --output-har-path ${NET}_compiled.har
RC=$?

ls -la > files.txt
sha256sum *.har *.hef >> files.txt 2>/dev/null
cp -f env.txt steps.txt files.txt *.log parsing_report.json "$OUT/" 2>/dev/null
cp -f *.har *.hef "$OUT/" 2>/dev/null
# results/에는 작은 텍스트만 (로그가 크면 앞뒤만)
mkdir -p "$RES/$TAG"
cp -f env.txt steps.txt files.txt parsing_report.json "$RES/$TAG/" 2>/dev/null
for f in *.log; do
  if [ $(stat -c %s "$f") -gt 300000 ]; then { head -c 100000 "$f"; echo; echo "... (중간 생략, 전체: $OUT/$f) ..."; tail -c 150000 "$f"; } > "$RES/$TAG/$f"
  else cp -f "$f" "$RES/$TAG/"; fi
done
echo "G1DFC DONE rc=$RC out=$OUT"
exit $RC
