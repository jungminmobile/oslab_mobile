#!/usr/bin/env bash
# overnight_wsl2.sh — overnight_wsl.sh의 나머지(구성 D 스크리닝 뒤)를 순서를 바꿔 이어서 실행
# 바꾼 이유: overnight_wsl.sh는 한 단계가 끝나자마자 다음 단계 잠금을 바로 잡아, Windows 쪽 짧은 작업(깨끗한 재실행·원인 분석 2단계,
#   합계 약 35분)이 G4 스크리닝 5개(약 5시간 30분) 뒤로 밀린다. → Windows의 overnight_win3.sh가 끝난 뒤 시작하고,
#   Hailo 실기기 가능성을 보여 주는 D 컴파일을 앞당긴다. 각 단계 내용은 overnight_wsl.sh와 같다.
# 순서: 스크리닝 E, C → 컴파일 D → analyze-noise(G1 기본, D) → 스크리닝 B, A → 컴파일 E
# 사용 (WSL Ubuntu-22.04, 홈에서): bash overnight_wsl2.sh
set -uo pipefail
source ~/hailo10_venv/bin/activate
cd ~
LOCK="python3 /mnt/d/mokerton/locks/heavy_lock.py run --owner siglip2 --job"
REPO=/mnt/d/mokerton/oslab_mobile
R=$REPO/results/siglip2
W0=~/siglip2_dfc/20261013_124213        # G1 (기본 8비트)
W1=~/siglip2_dfc/g4_20261013_165132     # A·B·C
W2=~/siglip2_dfc/g4_20261013_221931     # D·E
DATA=/mnt/d/mokerton/artifacts/siglip2/dfc/pets_dev64_nhwc.npy
CALIB=$W0/calib_highres64_s0_nhwc.npy
until grep -q "OVERNIGHT_WIN3 DONE" /mnt/d/mokerton/artifacts/siglip2/logs/overnight_win3.log 2>/dev/null; do sleep 60; done
tr -d '\r' < $REPO/siglip2/hailo/s2_dfc_screen.py > ~/s2_dfc_screen.py
echo "$(date +%T) start commit=$(git -C $REPO rev-parse --short HEAD)"

screen() {   # $1 har, $2 tag
  $LOCK "G4 screen $2" -- python ~/s2_dfc_screen.py --har $1 --tag $2 --k 200 --bs 8 --out $R/G4_dfc_screen > $R/G4_dfc_screen/log_$2.txt 2>&1
  echo "$(date +%T) screen $2 rc=$? $(grep '"hr_cos"' $R/G4_dfc_screen/log_$2.txt | tail -1)"
}
compile() {  # $1 har, $2 tag
  C=~/siglip2_dfc/compile_$(date +%Y%m%d); mkdir -p $C/$2 $R/G4_compile/$2
  t0=$(date +%s)
  ( cd $C/$2 && $LOCK "DFC compile $2" -- hailo compiler $1 --hw-arch hailo10h --output-dir . --output-har-path compiled_$2.har > compile.log 2>&1 )
  rc=$?; echo "$(date +%T) compile $2 rc=$rc sec=$(( $(date +%s) - t0 ))"
  ( cd $C/$2 && $LOCK "DFC profiler $2" -- hailo profiler compiled_$2.har --hef *.hef --out-path profiler.html --csv profiler.csv > prof.log 2>&1 )
  echo "$(date +%T) profiler compiled $2 rc=$?"
  ( cd $C/$2 && ls -la > $R/G4_compile/$2/files.txt; sha256sum *.hef >> $R/G4_compile/$2/files.txt 2>/dev/null
    grep -a -E "contexts|Number of Contexts|Successful|FPS|rror" compile.log prof.log | tail -40 > $R/G4_compile/$2/summary.txt
    cp profiler.csv $R/G4_compile/$2/ 2>/dev/null; tail -c 200000 compile.log > $R/G4_compile/$2/compile_tail.log )
}
noise() {    # $1 har, $2 tag  (HAR을 복사해서 분석 — 원본 HAR은 그대로 둔다)
  N=~/siglip2_dfc/noise_$(date +%Y%m%d); mkdir -p $N $R/G4_noise
  cp $1 $N/noise_$2.har
  ( cd $N && $LOCK "DFC analyze-noise $2" -- hailo analyze-noise noise_$2.har --data-path $DATA --calib-data-path $CALIB \
      --batch-size 2 --data-count 64 > analyze_$2.log 2>&1 )
  echo "$(date +%T) analyze-noise $2 rc=$?"
  ( cd $N && $LOCK "DFC profiler $2" -- hailo profiler noise_$2.har --out-path profiler_$2.html --csv profiler_$2.csv > prof_$2.log 2>&1 )
  echo "$(date +%T) profiler $2 rc=$?"
  cp $N/analyze_$2.log $N/profiler_$2.csv $N/prof_$2.log $R/G4_noise/ 2>/dev/null
}

screen $W2/opt_E_fc2_b0b5b9.har E_fc2_b0b5b9
screen $W1/opt_C_official.har C_official
compile $W2/opt_D_official_b9fc2.har D_official_b9fc2
noise $W0/siglip2_b16_224_optimized.har G1_default
noise $W2/opt_D_official_b9fc2.har D_official_b9fc2
screen $W1/opt_B_latefc2.har B_latefc2
screen $W1/opt_A_ewadd_patch16.har A_ewadd_patch16
compile $W2/opt_E_fc2_b0b5b9.har E_fc2_b0b5b9
echo "OVERNIGHT_WSL2 DONE $(date +%T)"
