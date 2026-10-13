#!/usr/bin/env bash
# overnight_wsl.sh — 밤샘 WSL(DFC) 작업. 단계마다 공용 잠금(D:\mokerton\locks\heavy_lock.py)을 잡고 놓아
# 개인연구 세션과 무거운 작업이 겹치지 않게 한다.
#  1) G4 확대 평가: 구성 A~E를 dev 6종 × 앞 200장 DFC 양자화 에뮬레이션 (ORT 스크리닝과 같은 형식)  → results/siglip2/G4_dfc_screen
#  2) 층별 잡음 분석(hailo analyze-noise): G1 기본 8비트, 구성 D                                    → results/siglip2/G4_noise
#  3) 컴파일: 구성 D, E (HEF 크기·컨텍스트 수, 프로파일러)                                            → results/siglip2/G4_compile
# 사용 (WSL Ubuntu-22.04): bash overnight_wsl.sh
set -uo pipefail
source ~/hailo10_venv/bin/activate
LOCK="python3 /mnt/d/mokerton/locks/heavy_lock.py run --owner siglip2 --job"
REPO=/mnt/d/mokerton/oslab_mobile
R=$REPO/results/siglip2
W0=~/siglip2_dfc/20261013_124213        # G1 (기본 8비트)
W1=~/siglip2_dfc/g4_20261013_165132     # A·B·C
W2=~/siglip2_dfc/g4_20261013_221931     # D·E
DATA=/mnt/d/mokerton/artifacts/siglip2/dfc/pets_dev64_nhwc.npy
CALIB=$W0/calib_highres64_s0_nhwc.npy
tr -d '\r' < $REPO/siglip2/hailo/s2_dfc_screen.py > ~/s2_dfc_screen.py
echo "$(date +%T) start commit=$(git -C $REPO rev-parse --short HEAD)"

# 1) G4 확대 평가
mkdir -p $R/G4_dfc_screen
for item in "$W2/opt_D_official_b9fc2.har D_official_b9fc2" "$W2/opt_E_fc2_b0b5b9.har E_fc2_b0b5b9" \
            "$W1/opt_C_official.har C_official" "$W1/opt_B_latefc2.har B_latefc2" "$W1/opt_A_ewadd_patch16.har A_ewadd_patch16"; do
  set -- $item
  $LOCK "G4 screen $2" -- python ~/s2_dfc_screen.py --har $1 --tag $2 --k 200 --bs 8 --out $R/G4_dfc_screen > $R/G4_dfc_screen/log_$2.txt 2>&1
  echo "$(date +%T) screen $2 rc=$? $(grep '"hr_cos"' $R/G4_dfc_screen/log_$2.txt | tail -1)"
done

# 2) 층별 잡음 분석 (HAR을 복사해서 분석 — 원본 HAR은 그대로 둔다)
N=~/siglip2_dfc/noise_$(date +%Y%m%d); mkdir -p $N $R/G4_noise
for item in "$W0/siglip2_b16_224_optimized.har G1_default" "$W2/opt_D_official_b9fc2.har D_official_b9fc2"; do
  set -- $item
  cp $1 $N/noise_$2.har
  ( cd $N && $LOCK "DFC analyze-noise $2" -- hailo analyze-noise noise_$2.har --data-path $DATA --calib-data-path $CALIB \
      --batch-size 2 --data-count 64 > analyze_$2.log 2>&1 )
  echo "$(date +%T) analyze-noise $2 rc=$?"
  ( cd $N && $LOCK "DFC profiler $2" -- hailo profiler noise_$2.har --out-path profiler_$2.html --csv profiler_$2.csv > prof_$2.log 2>&1 )
  echo "$(date +%T) profiler $2 rc=$?"
  cp $N/analyze_$2.log $N/profiler_$2.csv $N/prof_$2.log $R/G4_noise/ 2>/dev/null
done

# 3) 컴파일 (구성당 2시간+, 단계마다 잠금)
C=~/siglip2_dfc/compile_$(date +%Y%m%d); mkdir -p $C $R/G4_compile
for item in "$W2/opt_D_official_b9fc2.har D_official_b9fc2" "$W2/opt_E_fc2_b0b5b9.har E_fc2_b0b5b9"; do
  set -- $item
  mkdir -p $C/$2
  t0=$(date +%s)
  ( cd $C/$2 && $LOCK "DFC compile $2" -- hailo compiler $1 --hw-arch hailo10h --output-dir . --output-har-path compiled_$2.har > compile.log 2>&1 )
  rc=$?; echo "$(date +%T) compile $2 rc=$rc sec=$(( $(date +%s) - t0 ))"
  ( cd $C/$2 && $LOCK "DFC profiler $2" -- hailo profiler compiled_$2.har --hef *.hef --out-path profiler.html --csv profiler.csv > prof.log 2>&1 )
  mkdir -p $R/G4_compile/$2
  ( cd $C/$2 && ls -la > $R/G4_compile/$2/files.txt; sha256sum *.hef >> $R/G4_compile/$2/files.txt 2>/dev/null
    grep -a -E "contexts|Number of Contexts|Successful|FPS|rror" compile.log prof.log | tail -40 > $R/G4_compile/$2/summary.txt
    cp profiler.csv $R/G4_compile/$2/ 2>/dev/null; tail -c 200000 compile.log > $R/G4_compile/$2/compile_tail.log )
done
echo "OVERNIGHT_WSL DONE $(date +%T)"
