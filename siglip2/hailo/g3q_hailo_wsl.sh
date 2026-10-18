#!/usr/bin/env bash
# g3q_hailo_wsl.sh — G3Q-H: 회전(rothad)·원래(D) HAR → optimize(D256L1 .alls) → SDK_QUANTIZED 스크리닝(dev 6×200) → compile → profiler
# WSL Ubuntu-22.04 (연구실 데스크탑), ~/hailo10_venv, CUDA_VISIBLE_DEVICES=-1. 작업 폴더 ~/siglip2_work/hailo_g3q (저장소 밖, DFC 로그가 저장소를 dirty로 만들지 않게).
# optimize·screen·compile은 다른 세션(G3R)과 메모리 충돌을 막으려고 flock ~/siglip2_work/dfc.lock 안에서, 시작 전 free ≥ MIN_FREE_GB 대기.
# 사용: bash g3q_hailo_wsl.sh <optimize|screen|compile> <rothad|D> ...      (단계 하나, 구성 여러 개를 순서대로)
#   HAR: rothad=siglip2_rothad_b16_224.har, D=siglip2_b16_224.har (hailo parser onnx <b1> --hw-arch hailo10h --net-name ...로 미리 만든 것)
set -uo pipefail
source ~/hailo10_venv/bin/activate
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 S2_DFC_CACHE=$HOME/siglip2_work/artifacts/cache S2_DFC_REPO=$HOME/oslab_rot
MIN_FREE_GB=${MIN_FREE_GB:-15}
SELF=$(readlink -f "$0"); H=$(dirname "$SELF")
W=$HOME/siglip2_work/hailo_g3q; cd $W
O=$S2_DFC_REPO/results/siglip2/G3Q_rotate/hailo; mkdir -p $O
LOCK=$HOME/siglip2_work/dfc.lock
ALLS=g4_D256L1_official_b9fc2.alls
CAL=$HOME/siglip2_work/artifacts/dfc/calib_highres256_s0_nhwc.npy
log() { echo "$(date +%T) $*" | tee -a $O/run_info.txt; }
har() { [ "$1" = rothad ] && echo siglip2_rothad_b16_224.har || echo siglip2_b16_224.har; }
waitmem() { while true; do f=$(free -g | awk '/Mem:/{print $7}'); [ "$f" -ge "$MIN_FREE_GB" ] && { log "$1 free=${f}G 시작"; return; }; sleep 60; done; }

locked() {   # 잠금 안에서 실행되는 한 단계 (인자: 단계 구성)
  local step=$1 c=$2 tag=${2}_D256L1 t0 rc
  waitmem "$step $tag"; t0=$(date +%s)
  case $step in
    optimize)
      hailo optimize $(har $c) --hw-arch hailo10h --model-script $ALLS --calib-set-path $CAL --output-har-path opt_$tag.har > optimize_$tag.log 2>&1; rc=$?
      cp optimize_$tag.log $O/; [ $rc -eq 0 ] && sha256sum opt_$tag.har >> $O/har_sha256.txt ;;
    screen)
      python $H/s2_dfc_screen.py --har opt_$tag.har --tag $tag --k 200 --bs 8 --out $O > screen_$tag.log 2>&1; rc=$?
      cp screen_$tag.log $O/ ;;
    compile)
      mkdir -p comp_$tag && cd comp_$tag
      hailo compiler ../opt_$tag.har --hw-arch hailo10h --output-dir . --output-har-path compiled_$tag.har > compile.log 2>&1; rc=$?
      cd $W ;;
  esac
  log "$tag $step rc=$rc sec=$(( $(date +%s) - t0 ))"
  return $rc
}

if [ "${1:-}" = _locked ]; then locked "$2" "$3"; exit $?; fi
step=$1; shift
cp $H/$ALLS .
log "step=$step cfgs=$* commit=$(git -C $S2_DFC_REPO rev-parse --short HEAD) dirty=$(git -C $S2_DFC_REPO status --porcelain --untracked-files=no | wc -l) alls_sha=$(sha256sum $ALLS | cut -c1-16) calib_sha=$(sha256sum $CAL | cut -c1-16)"
for c in "$@"; do
  tag=${c}_D256L1
  log "$tag $step 잠금 대기"
  flock $LOCK bash "$SELF" _locked $step $c || { log "$tag $step 실패"; continue; }
  if [ $step = screen ]; then log "$tag $(tail -1 screen_$tag.log)"; fi
  if [ $step = compile ]; then
    cd comp_$tag; hef=$(ls *.hef 2>/dev/null | head -1)
    if [ -n "$hef" ]; then
      hailo profiler compiled_$tag.har --hef $hef --out-path profiler.html --csv profiler.csv > prof.log 2>&1
      mkdir -p $O/$tag; cp profiler.csv prof.log $O/$tag/; tail -c 200000 compile.log > $O/$tag/compile_tail.log
      sha256sum $hef > $O/$tag/files.txt; ls -l $hef >> $O/$tag/files.txt
      log "$tag hef=$hef bytes=$(stat -c %s $hef) sha256=$(sha256sum $hef | cut -c1-16) $(grep -i -m1 'Number of Contexts' prof.log)"
    fi
    cd $W
  fi
done
log "G3Q-H $step DONE $*"
