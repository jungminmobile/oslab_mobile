#!/usr/bin/env bash
# after_speed_win.sh — 지연 재측정(스레드 1·2·4)이 끝난 뒤 Windows 쪽 실험 (단계마다 공용 잠금)
#  1) 후보 모델 지표: 크기·peak 메모리(세션 생성·추론, 새 프로세스) — 후보표의 PRD 필수 열 (results/siglip2/G3_cumulative/model_stats)
#  2) 새 venv 재현 테스트 (scripts/g3_repro_venv.sh, ROADMAP G3 완료 기준)
#  3) DFC 보정 256장 사진 만들기 (가벼움, WSL 쪽 g4_more_wsl.sh가 사용)
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
ART="$SIGLIP2_ARTIFACTS"
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
P=siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16
until grep -a -q "SPEED_RERUN4_DONE" $ART/logs/speed_rerun_t4.log 2>/dev/null; do sleep 30; done
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=all -- . | tr '\n' ' ')]"
"$PY" hailo/s2_dfc_calib.py --n 256 > $ART/logs/dfc_calib256.log 2>&1
rc=$?; echo "$(date +%T) dfc calib256 rc=$rc $(tail -1 $ART/logs/dfc_calib256.log)"
RN50_RESULTS=../results/siglip2/G3_cumulative/model_stats "${LOCK[@]}" "SigLIP2 candidate model stats (peak mem)" -- \
    "$PY" -u s2_model_stats.py --models fp32=siglip2_image_fp32.onnx int8_all=siglip2_image_int8_minmax_highres256_s0_s8_all.onnx \
      fc2L7=${P}_mm_nofc2L.onnx fc2L4=${P}_fc2L4.onnx fc2L2=${P}_fc2L2.onnx \
      fc2L2_u8=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_fc2L2.onnx > $ART/logs/model_stats_candidates.log 2>&1
rc=$?; echo "$(date +%T) model stats rc=$rc"
bash scripts/g3_repro_venv.sh > $ART/logs/repro_venv.log 2>&1
rc=$?; echo "$(date +%T) repro rc=$rc $(tail -1 $ART/logs/repro_venv.log)"
echo "AFTER_SPEED_WIN DONE $(date +%T)"
