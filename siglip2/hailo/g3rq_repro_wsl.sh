#!/usr/bin/env bash
# 빌드 재현성 (G3RQ repro): rothad_D256L1·M2rot_D256L1 2차 빌드 = 1차와 같은 HAR·.alls·보정·명령으로 optimize → compile → profiler. + rothad HAR SDK_NATIVE 동치.
# 작업 폴더 ~/siglip2_work/hailo_repro (저장소 밖). DFC는 flock ~/siglip2_work/dfc.lock 안, free ≥ 15 GB 대기, CUDA_VISIBLE_DEVICES=-1.
set -uo pipefail
source ~/hailo10_venv/bin/activate
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1
H=$(cd "$(dirname "$0")" && pwd); REPO=$(cd $H/../.. && pwd)
O=$REPO/results/siglip2/G3RQ_m2rot/hailo/repro; mkdir -p $O
W=$HOME/siglip2_work/hailo_repro; mkdir -p $W; cd $W
CAL=$HOME/siglip2_work/artifacts/dfc/calib_highres256_s0_nhwc.npy
cp $HOME/siglip2_work/hailo_g3q/siglip2_rothad_b16_224.har $HOME/siglip2_work/g3b_20261018_215006/siglip2_m2rot.har .
cp $REPO/siglip2/hailo/g4_D256L1_official_b9fc2.alls $H/g3r_M2rot_D256L1equiv.alls .
log() { echo "$(date +%T) $*" | tee -a $O/run_info.txt; }
log "commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -C $REPO status --porcelain --untracked-files=no | wc -l) $(sha256sum siglip2_rothad_b16_224.har siglip2_m2rot.har g4_D256L1_official_b9fc2.alls g3r_M2rot_D256L1equiv.alls $CAL | awk '{printf "%s=%s ", $2, substr($1,1,16)}')"
run_locked() { flock $HOME/siglip2_work/dfc.lock bash -c "while [ \$(free -g | awk '/Mem:/{print \$7}') -lt 15 ]; do sleep 60; done; echo \"\$(date +%T) $1 잠금·메모리 OK free=\$(free -g | awk '/Mem:/{print \$7}')G\" >> $O/run_info.txt; $2"; }
for c in rothad M2rot; do
  if [ $c = rothad ]; then HAR=siglip2_rothad_b16_224.har; ALLS=g4_D256L1_official_b9fc2.alls; else HAR=siglip2_m2rot.har; ALLS=g3r_M2rot_D256L1equiv.alls; fi
  t0=$(date +%s)
  run_locked "optimize $c" "hailo optimize $HAR --hw-arch hailo10h --model-script $ALLS --calib-set-path $CAL --output-har-path opt2_$c.har > optimize2_$c.log 2>&1"
  log "$c optimize rc=$? sec=$(( $(date +%s) - t0 )) har2=$(sha256sum opt2_$c.har 2>/dev/null | cut -c1-16) $(grep -h 'SNR:' optimize2_$c.log | tail -1 | tr -s '\t ' ' ')"
  mkdir -p comp2_$c; t0=$(date +%s)
  run_locked "compile $c" "cd comp2_$c && hailo compiler ../opt2_$c.har --hw-arch hailo10h --output-dir . --output-har-path compiled2_$c.har > compile.log 2>&1"
  log "$c compile rc=$? sec=$(( $(date +%s) - t0 ))"
  cd comp2_$c; hef=$(ls *.hef | head -1)
  hailo profiler compiled2_$c.har --hef $hef --out-path profiler.html --csv profiler.csv > prof.log 2>&1
  cp $hef ../${c}_D256L1_build2.hef
  mkdir -p $O/$c; cp profiler.csv prof.log $O/$c/; cp ../optimize2_$c.log $O/$c/; tail -c 200000 compile.log > $O/$c/compile_tail.log
  log "$c hef=$hef bytes=$(stat -c %s $hef) sha256=$(sha256sum $hef | cut -c1-64) $(grep -i -m1 'Number of Contexts' prof.log | tr -s '\t ' ' ')"
  cd $W
done
run_locked "native rothad" "python $H/s2_dfc_native_check.py --har siglip2_rothad_b16_224.har --out $O/native_check.jsonl > native_rothad.log 2>&1"
log "native rothad rc=$? $(tail -1 native_rothad.log | cut -c1-200)"
log "REPRO BUILD DONE"
