#!/usr/bin/env bash
# g5c_dfc.sh — Zoo v5.4.0 공개 ONNX + 공개 .alls(그대로)를 DFC 5.4.0으로 parse → optimize → compile (G5c, G5b g5b_dfc.sh와 같은 흐름)
# 사용: TAG=R-our CALIB=<NHWC 0~255 float32 npy, 64장> nohup bash g5c_dfc.sh > ~/siglip2_work/g5c/<TAG>/nohup.log 2>&1 &
# WSL Ubuntu-22.04, ~/hailo10_venv, CUDA_VISIBLE_DEVICES=-1, 모든 DFC 단계는 flock ~/siglip2_work/dfc.lock 안에서.
set -uo pipefail
source ~/hailo10_venv/bin/activate
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1
H=$(cd "$(dirname "$0")" && pwd); REPO=$(git -C $H rev-parse --show-toplevel)
TAG=${TAG:?}; CALIB=${CALIB:?}
W=$HOME/siglip2_work/g5c/$TAG; LOCK=$HOME/siglip2_work/dfc.lock; mkdir -p $W
ONNX=$HOME/siglip2_work/g5c/onnx/siglip_base_patch16_224_vision_encoder.onnx
ALLS=zoo_v540_siglip_b_16_image_encoder.alls; NET=siglip_b_16_image_encoder
L=$W/run_info.txt; cd $W
log() { echo "$(date +%T) $*" | tee -a $L; }
log "tag=$TAG commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -C $REPO status --porcelain --untracked-files=no | wc -l) dfc=$(pip show hailo-dataflow-compiler 2>/dev/null | awk '/^Version/{print $2}') onnx_sha256=$(sha256sum $ONNX | cut -c1-64) alls_sha256=$(sha256sum $H/$ALLS | cut -c1-64) calib=$CALIB calib_sha256=$(sha256sum $CALIB | cut -c1-64)"
cp $H/$ALLS .; cp $CALIB calib.npy
python - <<PY || { log "CALIB FILE CHECK FAIL"; exit 1; }
import numpy as np
x = np.load("calib.npy", mmap_mode="r"); assert x.shape == (64, 224, 224, 3) and x.dtype == np.float32, (x.shape, x.dtype)
assert 0 <= float(x.min()) and float(x.max()) <= 255 and float(x.max()) > 2, (x.min(), x.max())
print("calib", x.shape, float(x.min()), float(x.max()), round(float(x.mean()), 2))
PY
step() {
  local n=$1; shift
  log "$n 잠금 대기"
  ( flock 9; log "$n 잠금 잡음 free=$(free -g | awk '/Mem:/{print $7}')G"
    t0=$(date +%s); /usr/bin/time -v -o time_$n.txt "$@" > $n.log 2>&1; rc=$?
    log "$n rc=$rc sec=$(( $(date +%s) - t0 )) maxrss_kb=$(awk -F': ' '/Maximum resident/{print $2}' time_$n.txt)"; exit $rc ) 9>$LOCK
}
step parse hailo parser onnx $ONNX --net-name $NET --hw-arch hailo10h --start-node-names input.1 --end-node-names 1725 \
     --har-path $NET.har -y || { log "PARSE FAIL"; tail -30 parse.log | tee -a $L; exit 1; }
step optimize hailo optimize $NET.har --hw-arch hailo10h --model-script $ALLS --calib-set-path calib.npy \
     --output-har-path ${NET}_optimized.har || { log "OPTIMIZE FAIL"; tail -30 optimize.log | tee -a $L; exit 1; }
grep -E "Using dataset with [0-9]+ entries" optimize.log | tee -a $L
python - <<PY || { log "CALIB ASSERT FAIL"; exit 1; }
import re
n = [int(v) for v in re.findall(r"Using dataset with ([0-9]+) entries for calibration", open("optimize.log").read())]
print("log counts", n); assert n and all(v == 64 for v in n), n
PY
log "calib assert OK (64)"
mkdir -p comp && step compile bash -c "cd comp && hailo compiler ../${NET}_optimized.har --hw-arch hailo10h --output-dir . --output-har-path ${NET}_compiled.har" \
     || { log "COMPILE FAIL"; tail -30 compile.log | tee -a $L; exit 1; }
HEF=$(ls comp/*.hef | head -1)
log "hef=$HEF bytes=$(stat -c %s $HEF) sha256=$(sha256sum $HEF | cut -c1-64)"
log "G5C DFC DONE $TAG"
