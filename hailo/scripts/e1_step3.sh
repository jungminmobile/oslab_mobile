#!/bin/bash
# E1 3단계: 컴파일(.hef) + 예상 성능 보고서(추정치). 사용법: bash ~/e1_step3.sh [rn50|rn50x4|all]
set -u
W=~/hailo10_work; mkdir -p $W/hef $W/logs; cd $W/hef
source ~/hailo10_venv/bin/activate
WHICH=${1:-all}
for M in rn50 rn50x4; do
  [ "$WHICH" != all ] && [ "$WHICH" != "$M" ] && continue
  IN=$(ls $W/opt/${M}*optimized*.har 2>/dev/null | head -1)
  [ -z "$IN" ] && { echo "[$(date +%T)] $M: 2단계 결과가 없어요 → 먼저 bash ~/e1_step2.sh $M"; continue; }
  [ -f $W/hef/$M.hef ] && { echo "[$(date +%T)] $M 이미 완료 → 건너뜀"; continue; }
  echo "[$(date +%T)] ===== $M 컴파일 시작 (예상 RN50 20~60분 / RN50x4 1~2시간)"
  /usr/bin/time -v hailo compiler "$IN" --hw-arch hailo10h 2>&1 | tee $W/logs/${M}_compile.log
  H=$(ls -t $W/hef/*.hef 2>/dev/null | head -1)
  if [ -n "$H" ]; then
    [ "$H" != "$W/hef/$M.hef" ] && mv "$H" $W/hef/$M.hef
    echo "[$(date +%T)] ===== $M 컴파일 완료 ✅  $W/hef/$M.hef ($(du -h $W/hef/$M.hef | cut -f1))"
    C=$(ls -t $W/hef/*compiled*.har 2>/dev/null | head -1)
    [ -n "$C" ] && hailo profiler "$C" > $W/logs/${M}_profiler.log 2>&1 && echo "  예상 성능 보고서(추정치): $W/logs/${M}_profiler.log"
  else
    echo "[$(date +%T)] ===== $M 컴파일 실패 ❌  로그: $W/logs/${M}_compile.log"
  fi
done
echo "[$(date +%T)] ===== 3단계 끝. 이제 노트북을 덮어도 됩니다."
