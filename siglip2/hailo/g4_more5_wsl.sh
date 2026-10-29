#!/usr/bin/env bash
# g4_more5_wsl.sh — (10/15 낮) C + 보정 256장 + 최적화 수준 1(C256L1) 컴파일: HEF 크기·컨텍스트 확인 (D-032 판단 자료)
#  C256L1은 D256L1과 top-1 동률(results/siglip2/G4_dfc_screen 6-4절). C(97.1 MB·16)와 같은 크기인지 실제로 확인한다.
#  g4_more_wsl.sh의 compile()과 같은 명령·같은 기록 방식.
set -uo pipefail
source ~/hailo10_venv/bin/activate
cd ~
LOCK="python3 /mnt/d/mokerton/locks/heavy_lock.py run --owner siglip2 --job"
REPO=/mnt/d/mokerton/oslab_mobile
R=$REPO/results/siglip2
W4=~/siglip2_dfc/g4_lvl1_20261015
echo "$(date +%T) start commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -c core.autocrlf=true -C $REPO status --porcelain --untracked-files=no -- . ':(exclude)results' | wc -l)"
compile() {  # $1 har, $2 tag
  C=~/siglip2_dfc/compile_$(date +%Y%m%d); mkdir -p $C/$2 $R/G4_compile/$2
  t0=$(date +%s)
  ( cd $C/$2 && $LOCK "DFC compile $2" -- hailo compiler $1 --hw-arch hailo10h --output-dir . --output-har-path compiled_$2.har > compile.log 2>&1 )
  rc=$?; echo "$(date +%T) compile $2 rc=$rc sec=$(( $(date +%s) - t0 ))"
  ( cd $C/$2 && $LOCK "DFC profiler $2" -- hailo profiler compiled_$2.har --hef *.hef --out-path profiler.html --csv profiler.csv > prof.log 2>&1 )
  rc=$?; echo "$(date +%T) profiler $2 rc=$rc"
  ( cd $C/$2 && ls -la > $R/G4_compile/$2/files.txt; sha256sum *.hef >> $R/G4_compile/$2/files.txt 2>/dev/null
    grep -a -E "contexts|Number of Contexts|Successful|FPS|rror" compile.log prof.log | tail -40 > $R/G4_compile/$2/summary.txt
    cp profiler.csv $R/G4_compile/$2/ 2>/dev/null; tail -c 200000 compile.log > $R/G4_compile/$2/compile_tail.log )
}
compile $W4/opt_C256L1_official.har C256L1_official
echo "G4_MORE5 DONE $(date +%T)"
