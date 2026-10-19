#!/usr/bin/env bash
# g5b_dfc.sh — 벤더 레시피(Zoo v5.0.0 siglip2_b_16_image_encoder .alls)를 DFC 5.4.0으로 parse → optimize → compile (G5b 2단계)
# WSL Ubuntu-22.04, ~/hailo10_venv, CUDA_VISIBLE_DEVICES=-1, 모든 DFC 단계는 flock ~/siglip2_work/dfc.lock 안에서.
# 사용: nohup bash g5b_dfc.sh > ~/siglip2_work/g5b/dfc_nohup.log 2>&1 &
set -uo pipefail
source ~/hailo10_venv/bin/activate
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1
H=$(cd "$(dirname "$0")" && pwd); REPO=$(git -C $H rev-parse --show-toplevel)
W=${W:-$HOME/siglip2_work/g5b}; LOCK=$HOME/siglip2_work/dfc.lock
ONNX=$W/onnx/siglip2-base-patch16-224_vision_encoder.sim.onnx
POOL=${POOL:-$HOME/siglip2_work/artifacts/dfc/calib_highres256_s0_nhwc.npy}
ALLS=siglip2_b_16_image_encoder_v500.alls; NET=siglip2_b_16_image_encoder
L=$W/run_info.txt; cd $W
log() { echo "$(date +%T) $*" | tee -a $L; }
log "commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -C $REPO status --porcelain --untracked-files=no | wc -l) dfc=$(pip show hailo-dataflow-compiler 2>/dev/null | awk '/^Version/{print $2}') onnx_sha256=$(sha256sum $ONNX | cut -c1-64) alls_sha256=$(sha256sum $H/$ALLS | cut -c1-64) pool_sha256=$(sha256sum $POOL | cut -c1-64)"
cp $H/$ALLS .
python $H/g5b_calib.py --src $POOL --n 64 --out calib_pix64_s0_nhwc.npy | tee -a $L || exit 1
step() {   # $1 이름, 나머지 = 명령. 잠금 안에서 /usr/bin/time -v로 시간·최대 RSS
  local n=$1; shift
  log "$n 잠금 대기"
  ( flock 9; log "$n 잠금 잡음 free=$(free -g | awk '/Mem:/{print $7}')G"
    t0=$(date +%s); /usr/bin/time -v -o time_$n.txt "$@" > $n.log 2>&1; rc=$?
    log "$n rc=$rc sec=$(( $(date +%s) - t0 )) maxrss_kb=$(awk -F': ' '/Maximum resident/{print $2}' time_$n.txt)"; exit $rc ) 9>$LOCK
}
step parse hailo parser onnx $ONNX --net-name $NET --hw-arch hailo10h --start-node-names onnx::Cast_0 --end-node-names 1393 \
     --har-path $NET.har -y || { log "PARSE FAIL"; exit 1; }
step optimize hailo optimize $NET.har --hw-arch hailo10h --model-script $ALLS --calib-set-path calib_pix64_s0_nhwc.npy \
     --output-har-path ${NET}_optimized.har || { log "OPTIMIZE FAIL"; tail -30 optimize.log | tee -a $L; exit 1; }
# 실제로 쓰인 보정 장수 확인 (로그에서 찾은 줄을 남기고, 64가 아니면 멈춤)
grep -E "Using dataset with [0-9]+ entries" optimize.log | tee -a $L
python - <<PY || { log "CALIB ASSERT FAIL"; exit 1; }
import numpy as np, re
x = np.load("calib_pix64_s0_nhwc.npy", mmap_mode="r"); assert x.shape[0] == 64, x.shape
t = open("optimize.log").read()
n = [int(v) for v in re.findall(r"Using dataset with ([0-9]+) entries for calibration", t)]
print("calib file rows", x.shape[0], "log counts", n)
assert n and all(v == 64 for v in n), n
PY
log "calib assert OK (64)"
mkdir -p comp && step compile bash -c "cd comp && hailo compiler ../${NET}_optimized.har --hw-arch hailo10h --output-dir . --output-har-path ${NET}_compiled.har" \
     || { log "COMPILE FAIL"; tail -30 compile.log | tee -a $L; exit 1; }
HEF=$(ls comp/*.hef | head -1)
log "hef=$HEF bytes=$(stat -c %s $HEF) sha256=$(sha256sum $HEF | cut -c1-64)"
log "G5B DFC DONE"
