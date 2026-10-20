#!/usr/bin/env bash
# G3RQ-QFT: M2rot_D256L1 + CPU 미세조정 (dataset_size 256, epochs 4, batch 2) optimize(최대 2시간) → 스크리닝 → (cos ≥ 0.9896일 때만) compile.
# flock ~/siglip2_work/dfc.lock 안, free ≥ 20 GB 대기, CUDA_VISIBLE_DEVICES=-1. 작업 폴더 ~/siglip2_work/hailo_qft_m2rot (저장소 밖)
set -uo pipefail
source ~/hailo10_venv/bin/activate
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 S2_DFC_CACHE=$HOME/siglip2_work/artifacts/cache
H=$(cd "$(dirname "$0")" && pwd); REPO=$(cd $H/../.. && pwd); export S2_DFC_REPO=$REPO
O=$REPO/results/siglip2/G3RQ_m2rot/hailo_qft; mkdir -p $O
W=$HOME/siglip2_work/hailo_qft_m2rot; mkdir -p $W; cd $W
CAL=$HOME/siglip2_work/artifacts/dfc/calib_highres256_s0_nhwc.npy
cp $HOME/siglip2_work/g3b_20261018_215006/siglip2_m2rot.har $H/g3rq_m2rot_qft.alls .
log() { echo "$(date +%T) $*" | tee -a $O/run_info.txt; }
log "commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -C $REPO status --porcelain --untracked-files=no | wc -l) $(sha256sum siglip2_m2rot.har g3rq_m2rot_qft.alls $CAL | awk '{printf "%s=%s ", $2, substr($1,1,16)}')"
locked() { flock $HOME/siglip2_work/dfc.lock bash -c "while [ \$(free -g | awk '/Mem:/{print \$7}') -lt 20 ]; do sleep 60; done; echo \"\$(date +%T) $1 잠금·메모리 OK free=\$(free -g | awk '/Mem:/{print \$7}')G load=\$(cut -d' ' -f1 /proc/loadavg)\" >> $O/run_info.txt; $2"; }
t0=$(date +%s)
locked optimize "timeout 2h /usr/bin/time -v hailo optimize siglip2_m2rot.har --hw-arch hailo10h --model-script g3rq_m2rot_qft.alls --calib-set-path $CAL --output-har-path opt_M2rot_qft.har > optimize_qft.log 2> optimize_qft.time"
rc=$?; log "optimize rc=$rc sec=$(( $(date +%s) - t0 )) maxrss_kb=$(grep 'Maximum resident' optimize_qft.time | awk '{print $NF}') $(grep -h 'SNR:' optimize_qft.log | tail -1 | tr -s '\t ' ' ')"
cp optimize_qft.log optimize_qft.time $O/
[ $rc -eq 124 ] && { log "2시간 넘김 — 멈춤 (보고)"; exit 1; }
[ $rc -ne 0 ] && { log "optimize 실패 — 멈춤"; exit 1; }
sha256sum opt_M2rot_qft.har >> $O/har_sha256.txt
t0=$(date +%s)
locked screen "python $H/s2_dfc_screen.py --har opt_M2rot_qft.har --tag M2rot_D256L1_qft --k 200 --bs 8 --out $O > screen_qft.log 2>&1"
log "screen rc=$? sec=$(( $(date +%s) - t0 )) $(tail -1 screen_qft.log)"; cp screen_qft.log $O/
cos=$(tail -1 $O/screen.jsonl | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['hr_cos'])")
if python3 -c "import sys; sys.exit(0 if float('$cos') >= 0.9896 else 1)"; then
  log "hr_cos $cos ≥ 0.9896 → 개선 후보 → compile"
  mkdir -p comp; t0=$(date +%s)
  locked compile "cd comp && hailo compiler ../opt_M2rot_qft.har --hw-arch hailo10h --output-dir . --output-har-path compiled_qft.har > compile.log 2>&1"
  log "compile rc=$? sec=$(( $(date +%s) - t0 ))"
  cd comp; hef=$(ls *.hef | head -1); hailo profiler compiled_qft.har --hef $hef --out-path profiler.html --csv profiler.csv > prof.log 2>&1
  cp $hef ../M2rotqft_D256L1.hef; mkdir -p $O/compile; cp profiler.csv prof.log $O/compile/; tail -c 200000 compile.log > $O/compile/compile_tail.log
  log "hef=$hef bytes=$(stat -c %s $hef) sha256=$(sha256sum $hef | cut -c1-64) $(grep -i -m1 'Number of Contexts' prof.log | tr -s '\t ' ' ')"
else
  log "hr_cos $cos < 0.9896 → 개선 없음 (끝)"
fi
log "G3RQ QFT DONE"
