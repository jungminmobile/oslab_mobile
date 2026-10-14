#!/usr/bin/env bash
# speed_rerun_t12.sh — 지연 측정 스레드 1·2 다시 재기 (10/14 11:45~12:10 개인연구 세션의 pip 설치가 잠금 없이 겹쳐 오염)
# g3_cumulative.sh speed와 같은 모델·같은 조건(PRD 5절: 워밍업 20, 200장 × 3회 번갈아, 배치 1). 스레드마다 공용 잠금.
# 잠금 도구가 python을 바로 실행하므로 WSL bash 문제(speed_after_g4.sh 주석)가 없다.
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"
R=../results/siglip2/G3_cumulative
P=siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16
until grep -a -q "SPEED_DONE" <(tail -n 2 $SIGLIP2_ARTIFACTS/logs/speed_run.log); do sleep 30; done
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=all -- . | tr '\n' ' ')]"
for t in 1 2; do
  RN50_RESULTS=$R/speed "$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "SigLIP2 latency rerun threads=$t (needs idle PC)" -- \
    "$PY" -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t \
      --models fp32=siglip2_image_fp32.onnx int8_all=siglip2_image_int8_minmax_highres256_s0_s8_all.onnx \
               fc2L7=${P}_mm_nofc2L.onnx fc2L4=${P}_fc2L4.onnx fc2L2=${P}_fc2L2.onnx \
               fc2L2_u8=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_fc2L2.onnx
  echo "$(date +%T) rerun threads=$t rc=$?"
done
echo "SPEED_RERUN_DONE $(date +%T)"
