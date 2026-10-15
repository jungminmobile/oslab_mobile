#!/usr/bin/env bash
# run_text_q.sh — 텍스트 블록 HAR을 구성별(.alls)로 optimize → DFC 에뮬레이션 정확도 (WSL Ubuntu-22.04, ~/hailo10_venv)
#   nohup bash textenc/hailo/run_text_q.sh <작업 폴더> > text_q.log 2>&1 &
# 작업 폴더에 필요한 것: siglip2_text_blocks_b1.har (parse 결과), dfc_text_calib64_s0.npy (보정, 3차원이면 자동 변환),
#                      dfc_eval_pets.npz · dfc_eval_food.npz · dfc_eval_holdout.npz (te_dfc_evalset.py)
# 구성 (textenc/hailo/*.alls, 모두 레벨 0·보정 64 — GPU 없음):
#   t0_base         전부 8비트 (DFC 기본)
#   t1_cproj0       블록 0 c_proj(conv4)만 16비트 — CPU 스모크의 "거대 활성값 층 1개 보호"와 같은 처방
#   t2_cproj0_ewadd t1 + 잔차 합(ew_add) 전부 16비트 — 잔차 흐름 채널 556의 거대 값(약 434)을 위해. 이미지 쪽 A 구성과 같은 처방
#   t3_cproj0_ewadd_ln t2 + LayerNorm 전부 16비트 (잔차 흐름을 읽는 층)
#   t4_all16        conv·ew_add·LayerNorm·normalization·matmul·softmax 전부 16비트 — 레벨 0의 상한 확인용
# compile은 하지 않는다 (정확도가 나오는 구성만 나중에 compile).
set -uo pipefail
W=$(realpath "${1:?작업 폴더}")
HERE=$(cd "$(dirname "$0")" && pwd)
source ~/hailo10_venv/bin/activate
if [ "${HAILO_USE_GPU:-0}" != 1 ]; then export CUDA_VISIBLE_DEVICES=-1; fi
cd "$W"
python -c "import numpy as np; x=np.load('dfc_text_calib64_s0.npy'); x=x[:, None] if x.ndim == 3 else x; np.save('dfc_text_calib64_s0.npy', x); print('calib', x.shape)"
{ echo "date=$(date -Is)"; echo "commit=$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null) dirty=$(git -C "$HERE" status --porcelain -- . 2>/dev/null | wc -l)"
  hailo --version 2>&1 | tail -1; sha256sum siglip2_text_blocks_b1.har dfc_text_calib64_s0.npy dfc_eval_*.npz; } > run_info.txt 2>&1
HARS=()
for cfg in ${CFGS:-t0_base t1_cproj0 t2_cproj0_ewadd}; do
  tr -d '\r' < "$HERE/$cfg.alls" > $cfg.alls
  t0=$(date +%s)
  hailo optimize siglip2_text_blocks_b1.har --hw-arch hailo10h --model-script $cfg.alls \
      --calib-set-path dfc_text_calib64_s0.npy --output-har-path opt_$cfg.har > optimize_$cfg.log 2>&1
  rc=$?; echo "optimize $cfg rc=$rc sec=$(( $(date +%s) - t0 ))" | tee -a steps.txt
  [ $rc = 0 ] && HARS+=(opt_$cfg.har)
done
python "$HERE/te_dfc_emulate.py" --har "${HARS[@]}" --data-dir . --out emulate_text.jsonl 2>&1 | grep -v -i warn | tee emulate_text.log
echo "TEXT Q DONE"; cat steps.txt
