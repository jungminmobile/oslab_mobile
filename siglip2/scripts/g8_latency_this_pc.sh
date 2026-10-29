#!/usr/bin/env bash
# g8_latency_this_pc.sh — 이 PC(i7-1165G7)에서 후보들의 지연·FPS·p50/p95·peak 메모리를 같은 실행으로 (cross_latency.py)
#  사용: bash scripts/g8_latency_this_pc.sh <결과 폴더 이름> 이름=onnx ...   (공용 잠금 + PC 조용할 때만: 잠금 안 CPU 30초 평균 25% 미만)
#  쭈리 규칙: 지연은 고성능 모드에서만 — 전원 구성표는 결과 json(device.power_scheme)에 자동 기록
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' OPENBLAS_NUM_THREADS=1
NAME=$1; shift
R=../results/siglip2/G8_candidates/latency_this_pc/$NAME; mkdir -p $R
PY="$SIGLIP2_PYTHON"; L=D:/mokerton/artifacts/siglip2/logs; STOP=/d/mokerton/locks/siglip2.STOP
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)" | tee $R/run.log
while true; do
  [ -f $STOP ] && { echo "STOP 파일 — 멈춤"; exit 3; }
  "$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "SigLIP2 G8 latency this PC $NAME (needs idle PC)" -- \
    "C:/Program Files/Git/bin/bash.exe" scripts/_idle_then.sh "$PY" -u scripts/cross_latency.py --out $R --mem --models "$@" >> $R/run.log 2>&1
  rc=$?; echo "$(date +%T) rc=$rc" | tee -a $R/run.log
  [ $rc -eq 75 ] && { sleep 180; continue; }
  exit $rc
done
