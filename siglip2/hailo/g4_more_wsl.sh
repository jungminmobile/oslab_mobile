#!/usr/bin/env bash
# g4_more_wsl.sh — G4 추가 실험 (WSL Ubuntu-22.04, 홈에서 실행, 단계마다 공용 잠금)
#  1) 보정 사진 수 64 → 256 (ROADMAP G4 "보정 사진 수 64/256/1024"): 구성 D와 같은 .alls에 calibset_size=256
#     (g4_D256_official_b9fc2.alls) → optimize → 스크리닝(dev 6×200, ORT와 같은 형식)
#  2) 64장 비교에서 열세였던 A·B도 같은 형식으로 스크리닝 (overnight_wsl2.sh에서 지연 측정 때문에 멈춘 것)
#  3) 구성 E·C 컴파일 (E: overnight_wsl2.sh 마지막 단계, C: D-032 판단 자료)
# 시작 조건: 지연 측정(scripts/speed_when_idle.sh)이 끝난 뒤. 보정 256장 최적화 HAR이 이미 있으면 건너뛴다 (10/14 14:31 완료)
set -uo pipefail
source ~/hailo10_venv/bin/activate
cd ~
LOCK="python3 /mnt/d/mokerton/locks/heavy_lock.py run --owner siglip2 --job"
REPO=/mnt/d/mokerton/oslab_mobile
R=$REPO/results/siglip2
ART=/mnt/d/mokerton/artifacts/siglip2
W0=~/siglip2_dfc/20261013_124213        # G1 HAR
W1=~/siglip2_dfc/g4_20261013_165132     # A·B·C
W2=~/siglip2_dfc/g4_20261013_221931     # D·E
until grep -q "SPEED_IDLE_DONE" $ART/logs/speed_when_idle_v2_main.log 2>/dev/null; do sleep 60; done   # 지연 측정이 끝난 뒤 (10/14 재개)
tr -d '\r' < $REPO/siglip2/hailo/s2_dfc_screen.py > ~/s2_dfc_screen.py
echo "$(date +%T) start commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -c core.autocrlf=true -C $REPO status --porcelain --untracked-files=no -- . ':(exclude)results' | wc -l)"

screen() {   # $1 har, $2 tag
  $LOCK "G4 screen $2" -- python ~/s2_dfc_screen.py --har $1 --tag $2 --k 200 --bs 8 --out $R/G4_dfc_screen > $R/G4_dfc_screen/log_$2.txt 2>&1
  echo "$(date +%T) screen $2 rc=$? $(grep '"hr_cos"' $R/G4_dfc_screen/log_$2.txt | tail -1)"
}

# 1) 보정 256장
W3=~/siglip2_dfc/g4_calib256_20261014; mkdir -p $W3   # 14:31에 최적화한 폴더 (날짜가 바뀌어도 같은 폴더)
cp $W0/siglip2_b16_224.har $W3/
cp $ART/dfc/calib_highres256_s0_nhwc.npy $W3/
tr -d '\r' < $REPO/siglip2/hailo/g4_D256_official_b9fc2.alls > $W3/g4_D256_official_b9fc2.alls
if [ ! -f $W3/opt_D256_official_b9fc2.har ]; then
t0=$(date +%s)
  ( cd $W3 && $LOCK "DFC optimize D256" -- hailo optimize siglip2_b16_224.har --hw-arch hailo10h --model-script g4_D256_official_b9fc2.alls \
      --calib-set-path calib_highres256_s0_nhwc.npy --output-har-path opt_D256_official_b9fc2.har > optimize_D256.log 2>&1 )
  echo "$(date +%T) optimize D256 rc=$? sec=$(( $(date +%s) - t0 ))"
else echo "$(date +%T) optimize D256 skipped (HAR exists)"; fi
[ -f $R/G4_dfc_screen/calib256/sha256.txt ] || { mkdir -p $R/G4_dfc_screen/calib256; cp $W3/optimize_D256.log $W3/g4_D256_official_b9fc2.alls $R/G4_dfc_screen/calib256/ 2>/dev/null
sha256sum $W3/opt_D256_official_b9fc2.har $W3/calib_highres256_s0_nhwc.npy > $R/G4_dfc_screen/calib256/sha256.txt 2>/dev/null; }
[ -f $W3/opt_D256_official_b9fc2.har ] && screen $W3/opt_D256_official_b9fc2.har D256_official_b9fc2

# 2) A·B
screen $W1/opt_B_latefc2.har B_latefc2
screen $W1/opt_A_ewadd_patch16.har A_ewadd_patch16

# 3) 컴파일: E (overnight_wsl2.sh 마지막 단계), C (D-032 판단 자료: C의 HEF 크기·컨텍스트)
compile() {  # $1 har, $2 tag
  C=~/siglip2_dfc/compile_$(date +%Y%m%d); mkdir -p $C/$2 $R/G4_compile/$2
  t0=$(date +%s)
  ( cd $C/$2 && $LOCK "DFC compile $2" -- hailo compiler $1 --hw-arch hailo10h --output-dir . --output-har-path compiled_$2.har > compile.log 2>&1 )
  echo "$(date +%T) compile $2 rc=$? sec=$(( $(date +%s) - t0 ))"
  ( cd $C/$2 && $LOCK "DFC profiler $2" -- hailo profiler compiled_$2.har --hef *.hef --out-path profiler.html --csv profiler.csv > prof.log 2>&1 )
  echo "$(date +%T) profiler $2 rc=$?"
  ( cd $C/$2 && ls -la > $R/G4_compile/$2/files.txt; sha256sum *.hef >> $R/G4_compile/$2/files.txt 2>/dev/null
    grep -a -E "contexts|Number of Contexts|Successful|FPS|rror" compile.log prof.log | tail -40 > $R/G4_compile/$2/summary.txt
    cp profiler.csv $R/G4_compile/$2/ 2>/dev/null; tail -c 200000 compile.log > $R/G4_compile/$2/compile_tail.log )
}
compile $W2/opt_E_fc2_b0b5b9.har E_fc2_b0b5b9
compile $W1/opt_C_official.har C_official
echo "G4_MORE DONE $(date +%T)"
