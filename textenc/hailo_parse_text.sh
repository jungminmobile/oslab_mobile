#!/usr/bin/env bash
# hailo_parse_text.sh — SigLIP2 텍스트 블록 12개 ONNX를 Hailo DFC(5.x, --hw-arch hailo10h)로 parse → optimize → compile 시험
# WSL Ubuntu-22.04, DFC 가상환경 ~/hailo10_venv 에서. 저장소 루트에서:
#   nohup bash textenc/hailo_parse_text.sh <ONNX·보정 파일 폴더> > hailo_text.log 2>&1 &
# 입력 (te_export_blocks.py가 만든 것):
#   siglip2_text_blocks_b1.onnx       (1,64,768) → (1,64,768)  블록 12개 + ln_final
#   siglip2_text_blocks_pool_b1.onnx  (1,64,768) → (1,768)     + 위치 63 Slice + projection
#   dfc_text_calib64_s0.npy           (64,64,768) 보정 입력
# 이번 시험의 질문은 "NPU에 올라가는가"뿐이다. optimize는 기본값(모델 스크립트 없음)이고, 정확도는 보지 않는다.
# 보정 파일 모양이 DFC가 기대하는 것과 다르면 무작위 보정(--use-random-calib-set)으로 한 번 더 시도해 compile 가능 여부만 본다.
set -uo pipefail
SRC=$(realpath "${1:?ONNX 폴더를 인자로}")
REPO=$(cd "$(dirname "$0")/.." && pwd)
TAG=$(date +%Y%m%d_%H%M%S)
W=~/textenc_dfc/$TAG
RES=$REPO/results/textenc/hailo_parse/$TAG
mkdir -p "$W" "$RES"
source ~/hailo10_venv/bin/activate
# WSL에서 NVIDIA 드라이버는 보이는데 CUDA·cuDNN이 없으면 TensorFlow가 GPU를 잡으려다 CUDA_ERROR_INVALID_HANDLE로 죽는다.
# CUDA 12.5+·cuDNN 9.3+를 깔기 전까지는 GPU를 숨기고 CPU로 돌린다 (HAILO_USE_GPU=1이면 숨기지 않음)
if [ "${HAILO_USE_GPU:-0}" != 1 ]; then export CUDA_VISIBLE_DEVICES=-1; fi
cp "$SRC"/siglip2_text_blocks_b1.onnx "$SRC"/dfc_text_calib64_s0.npy "$W/"     # 필수
cp "$SRC"/siglip2_text_blocks_pool_b1.onnx "$W/" 2>/dev/null || echo "pool 변형 없음 — 건너뜀"   # 선택
cd "$W"
{
  echo "tag=$TAG"; echo "commit=$(git -C "$REPO" rev-parse --short HEAD) dirty=$(git -C "$REPO" status --porcelain -- textenc | wc -l)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"; hailo --version 2>&1 | tail -1; python --version; uname -r; nproc; free -g | head -2
  sha256sum *.onnx *.npy
} > env.txt 2>&1

step() {  # step <이름> <명령...>
  local name=$1; shift
  local t0=$(date +%s)
  "$@" > "$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc sec=$(( $(date +%s) - t0 ))" | tee -a steps.txt
  return $rc
}

for NET in siglip2_text_blocks_b1 siglip2_text_blocks_pool_b1; do
  [ -f $NET.onnx ] || continue
  step "${NET}_parse" hailo parser onnx $NET.onnx --hw-arch hailo10h --net-name $NET --har-path $NET.har -y \
       --parsing-report-path ${NET}_parsing_report.json || continue
  if ! step "${NET}_optimize" hailo optimize $NET.har --hw-arch hailo10h --calib-set-path dfc_text_calib64_s0.npy \
         --output-har-path ${NET}_optimized.har; then
    step "${NET}_optimize_random" hailo optimize $NET.har --hw-arch hailo10h --use-random-calib-set \
         --output-har-path ${NET}_optimized.har || continue
  fi
  step "${NET}_compile" hailo compiler ${NET}_optimized.har --hw-arch hailo10h --output-dir . --output-har-path ${NET}_compiled.har
done

ls -la > files.txt
sha256sum *.har *.hef >> files.txt 2>/dev/null
cp -f env.txt steps.txt files.txt *_parsing_report.json "$RES/" 2>/dev/null
for f in *.log; do   # 큰 로그는 앞뒤만 저장소에
  if [ "$(stat -c %s "$f")" -gt 300000 ]; then { head -c 100000 "$f"; echo; echo "... (중간 생략, 전체: $W/$f) ..."; tail -c 150000 "$f"; } > "$RES/$f"
  else cp -f "$f" "$RES/"; fi
done
echo "TEXT DFC DONE work=$W results=$RES"
cat steps.txt
