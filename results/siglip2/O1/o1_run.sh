#!/usr/bin/env bash
# O1 실행 순서: 정확도(o1_eval.sh) → 지연 3세션 × 스레드 1·2·4 + peak 메모리(o1_ort_speed.sh, 10개 모델 번갈아)
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
bash $D/o1_eval.sh
source ~/siglip2_env.sh; source $D/o1_models.sh
M=""; SZ=""; for k in $ORDER; do M="$M $k=${F[$k]}:$(size_of $k)"; SZ="$SZ $k=$(size_of $k)"; done
cd $D/../../../siglip2 && M="$M" SZ="$SZ" bash ../results/siglip2/O1/o1_ort_speed.sh
