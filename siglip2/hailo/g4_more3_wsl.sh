#!/usr/bin/env bash
# g4_more3_wsl.sh — 10/15 새벽: ① 구성 C 컴파일 결과 정리(프로파일러, 10/15 02:49 컴파일 완료, 정리 단계는 WSL 종료로 취소됐음)
#                  ② DFC 최적화 수준 1 (구성 D, 보정 256장) — GPU 없이 되는지, 되면 스크리닝 (g4_more2_wsl.sh와 같은 내용)
set -uo pipefail
source ~/hailo10_venv/bin/activate
cd ~
LOCK="python3 /mnt/d/mokerton/locks/heavy_lock.py run --owner siglip2 --job"
REPO=/mnt/d/mokerton/oslab_mobile
R=$REPO/results/siglip2
echo "$(date +%T) start commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -c core.autocrlf=true -C $REPO status --porcelain --untracked-files=no -- . ':(exclude)results' | wc -l)"
C=~/siglip2_dfc/compile_20261014/C_official
mkdir -p $R/G4_compile/C_official
( cd $C && $LOCK "DFC profiler C_official" -- hailo profiler compiled_C_official.har --hef siglip2_b16_224.hef --out-path profiler.html --csv profiler.csv > prof.log 2>&1 )
rc=$?; echo "$(date +%T) profiler C rc=$rc"
( cd $C && ls -la > $R/G4_compile/C_official/files.txt; sha256sum *.hef >> $R/G4_compile/C_official/files.txt 2>/dev/null
  grep -a -E "contexts|Number of Contexts|Successful|FPS|rror" compile.log prof.log | tail -40 > $R/G4_compile/C_official/summary.txt
  cp profiler.csv $R/G4_compile/C_official/ 2>/dev/null; tail -c 200000 compile.log > $R/G4_compile/C_official/compile_tail.log )
tr -d '\r' < $REPO/siglip2/hailo/g4_more2_wsl.sh > ~/g4_more2_wsl.sh
bash ~/g4_more2_wsl.sh
rc=$?; echo "$(date +%T) level1 chain rc=$rc"
echo "G4_MORE3 DONE $(date +%T)"
