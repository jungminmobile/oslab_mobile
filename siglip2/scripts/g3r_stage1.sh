#!/usr/bin/env bash
# G3R 1단계 (README 7절 0~1): 환경 확인 → selftest → --src identity → cache → stats(G0) → fp32(G1a·G1b 스크리닝)
# 사용 (siglip2/ 폴더, Git Bash):  bash scripts/g3r_stage1.sh
# 관문에서 실패하면 그 자리에서 멈춘다. 전체 로그: $R/stage1_<시각>.log
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
R=../results/siglip2/G3R_regcache
mkdir -p "$R"
LOG="$R/stage1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }
die()  { echo "!!! 멈춤: $*"; exit 1; }

step "0. 환경"
for v in SIGLIP2_DATA SIGLIP2_ARTIFACTS SIGLIP2_PYTHON; do
  [ -n "${!v:-}" ] || die "$v 비어 있음 (siglip2/README.md 환경 변수)"; echo "$v=${!v}"
done
PY="$SIGLIP2_PYTHON"
echo "git $(git rev-parse --short HEAD) ($(git rev-parse --abbrev-ref HEAD)), dirty: $(git status --porcelain | wc -l)"
[ "$(git rev-parse --abbrev-ref HEAD)" = "siglip2/g3r-regcache" ] || die "브랜치가 siglip2/g3r-regcache가 아님"
C="$SIGLIP2_ARTIFACTS/cache"
miss=0
for d in cifar_A pets_dev flowers_dev cars_dev food_dev caltech_dev; do
  for s in x y clf ref_fp32; do [ -f "$C/${d}_${s}.npy" ] || { echo "없음: $C/${d}_${s}.npy"; miss=1; }; done
done
ncal=$(ls "$C"/calib_*_f32.npy 2>/dev/null | wc -l)
echo "보정 후보 calib_*_f32.npy: $ncal개 (기대 5)"
[ "$miss" = 0 ] && [ "$ncal" -ge 5 ] || die "dev 캐시/보정 후보 부족 → s2_build_dev.py all 먼저 (시간 확인 후)"
[ -f "$SIGLIP2_ARTIFACTS/onnx/siglip2_image_fp32.onnx" ] || die "siglip2_image_fp32.onnx 없음 (s2_export_onnx.py)"

step "0-a. selftest (무작위 가중치)"
"$PY" s2_regcache.py selftest

step "0-b. --src 꺼진 기본 경로 identity (기대 sha256 5eee7839…)"
RN50_RESULTS=$R/identity "$PY" s2_quantize.py --calib-source highres --calib-n 256 --op-types MatMul \
  --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range \
  --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/" --tag idcheck_fc2L2
H=$("$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" \
  "$SIGLIP2_ARTIFACTS/onnx/siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_idcheck_fc2L2.onnx")
echo "sha256 $H"
[ "$H" = "5eee783955591d1de818b532ea73b2bb2ea8a008e327cc92d323992c0f9083ac" ] || die "identity 불일치 → s2_quantize.py 기본 경로가 바뀜"

step "1. 평균 캐시 (보정 train 256장 s0)"
RN50_RESULTS=$R "$PY" s2_regcache.py cache

step "1-b. stats (G0 + G1b 부가)"
RN50_RESULTS=$R "$PY" s2_regcache.py stats --n 64

step "1-c. fp32 스크리닝 (G1a·G1b, dev 6종 × 200)"
RN50_RESULTS=$R "$PY" s2_regcache.py fp32 --k 200 --modes oracle mean none

step "판정 (README 5절)"
"$PY" - "$R/fp32_screen.jsonl" <<'EOF'
import json, sys
rows = {}
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line); rows[r["mode"]] = r["result"]   # 마지막 실행 기준
DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
o = rows["oracle"]; omin = min(o[d]["cos_min"] for d in DATA)
g1a = omin >= 0.99999
print(f"G1a oracle cos 최소 {omin:.6f} (≥ 0.99999) → {'통과' if g1a else '실패: 구현 버그, 다음 단계 금지'}")
for m in ("mean", "none"):
    r = rows[m]
    print(f"{m:5s} 고화질 cos {r['hr_cos']:.5f}  Δtop-1 {r['hr_dtop1']:+.2f}%p  "
          f"CIFAR A Δ {r['cifar_A']['top1'] - r['cifar_A']['fp32_top1']:+.2f}%p  "
          f"→ 스크리닝 cos ≥ 0.99: {'통과' if r['hr_cos'] >= 0.99 else '실패'}")
print("※ G1b 최종은 dev 전체 Δtop-1 95% 구간 하한 ≥ −0.5%p (README 7절 2단계, s2_paired_ci.py)")
sys.exit(0 if g1a else 1)
EOF
echo; echo "끝. 로그: $LOG"
