#!/usr/bin/env bash
# g6_dfc.sh — G6 구성 하나를 DFC 5.4.0으로 parse → (optimize → compile). G5c g5c_dfc.sh와 같은 흐름을 인자로 일반화.
# 사용: TAG=b32_V ONNX=<onnx> NET=<이름> ALLS=<.alls 경로> CALIB=<NHWC float32 0~255 npy> NCAL=64 SIZE=256 \
#       [START=<노드> END=<노드>] [PARSE_ONLY=1] nohup bash g6_dfc.sh > ~/siglip2_work/g6/dfc/<TAG>/nohup.log 2>&1 &
# WSL Ubuntu-22.04, ~/hailo10_venv, CUDA_VISIBLE_DEVICES=-1. 모든 DFC 단계는 flock ~/siglip2_work/dfc.lock 안에서,
# 잠금을 잡은 뒤 여유 메모리 ≥ 20 GB와 다른 세션의 CPU 측정(G8 g8_speed.sh 등) 없음을 확인하고 시작한다.
set -uo pipefail
source ~/hailo10_venv/bin/activate
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1
H=$(cd "$(dirname "$0")" && pwd); REPO=$(git -C $H rev-parse --show-toplevel)
TAG=${TAG:?}; ONNX=${ONNX:?}; NET=${NET:?}; SIZE=${SIZE:?}
PARSE_ONLY=${PARSE_ONLY:-0}; START=${START:-}; END=${END:-}
W=$HOME/siglip2_work/g6/dfc/$TAG; LOCK=$HOME/siglip2_work/dfc.lock; mkdir -p $W
L=$W/run_info.txt; cd $W
log() { echo "$(date +%T) $*" | tee -a $L; }
log "tag=$TAG commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -C $REPO status --porcelain --untracked-files=no | wc -l) dfc=$(pip show hailo-dataflow-compiler 2>/dev/null | awk '/^Version/{print $2}') onnx=$ONNX onnx_sha256=$(sha256sum $ONNX | cut -c1-64) parse_only=$PARSE_ONLY"
busy() { ps -eo cmd | grep -E "g8_speed.sh|s2_bench_speed.py|s2_model_stats.py|s2_profile_ort.py" | grep -v grep | wc -l; }
step() {
  local n=$1; shift
  log "$n 잠금 대기"
  ( flock 9
    while [ $(free -g | awk '/Mem:/{print $7}') -lt 20 ] || [ $(busy) -gt 0 ]; do
      log "$n 대기: free=$(free -g | awk '/Mem:/{print $7}')G 다른 CPU 측정=$(busy)"; sleep 120; done
    log "$n 잠금 잡음 free=$(free -g | awk '/Mem:/{print $7}')G load=$(cut -d' ' -f1-3 /proc/loadavg)"
    t0=$(date +%s); /usr/bin/time -v -o time_$n.txt "$@" > $n.log 2>&1; rc=$?
    log "$n rc=$rc sec=$(( $(date +%s) - t0 )) maxrss_kb=$(awk -F': ' '/Maximum resident/{print $2}' time_$n.txt)"; exit $rc ) 9>$LOCK
}
NODES=""; [ -n "$START" ] && NODES="--start-node-names $START --end-node-names $END"
step parse hailo parser onnx $ONNX --net-name $NET --hw-arch hailo10h $NODES --har-path $NET.har -y \
  || { log "PARSE FAIL"; tail -30 parse.log | tee -a $L; exit 1; }
python $REPO/siglip2/hailo/s2_har_layers.py $NET.har layer_map.csv | tee -a $L
if [ "$PARSE_ONLY" = 1 ]; then log "G6 DFC PARSE DONE $TAG"; exit 0; fi
ALLS=${ALLS:?}; CALIB=${CALIB:?}; NCAL=${NCAL:?}
log "alls=$ALLS alls_sha256=$(sha256sum $ALLS | cut -c1-64) calib=$CALIB calib_sha256=$(sha256sum $CALIB | cut -c1-64) ncal=$NCAL"
cp $ALLS model.alls; cp $CALIB calib.npy
python - <<PY || { log "CALIB FILE CHECK FAIL"; exit 1; }
import numpy as np
x = np.load("calib.npy", mmap_mode="r"); assert x.shape == ($NCAL, $SIZE, $SIZE, 3) and x.dtype == np.float32, (x.shape, x.dtype)
assert 0 <= float(x.min()) and float(x.max()) <= 255 and float(x.max()) > 2, (x.min(), x.max())
print("calib", x.shape, float(x.min()), float(x.max()), round(float(x.mean()), 2))
PY
step optimize hailo optimize $NET.har --hw-arch hailo10h --model-script model.alls --calib-set-path calib.npy \
     --output-har-path ${NET}_optimized.har || { log "OPTIMIZE FAIL"; tail -30 optimize.log | tee -a $L; exit 1; }
grep -E "Using dataset with [0-9]+ entries" optimize.log | tee -a $L
python - <<PY || { log "CALIB ASSERT FAIL"; exit 1; }
import re
n = [int(v) for v in re.findall(r"Using dataset with ([0-9]+) entries for calibration", open("optimize.log").read())]
print("log counts", n); assert n and all(v == $NCAL for v in n), n
PY
log "calib assert OK ($NCAL)"
mkdir -p comp && step compile bash -c "cd comp && hailo compiler ../${NET}_optimized.har --hw-arch hailo10h --output-dir . --output-har-path ${NET}_compiled.har" \
     || { log "COMPILE FAIL"; tail -30 compile.log | tee -a $L; exit 1; }
HEF=$(ls comp/*.hef | head -1)
log "hef=$HEF bytes=$(stat -c %s $HEF) sha256=$(sha256sum $HEF | cut -c1-64) contexts=$(grep -oE "_context_[0-9]+" compile.log | sort -u | wc -l)"
log "G6 DFC DONE $TAG"
