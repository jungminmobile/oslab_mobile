#!/bin/bash
# 진행 상황 한눈에 보기: bash ~/e1_status.sh
W=~/hailo10_work
echo "== 지금 시각 $(date +%T)"
echo "== 돌고 있는 hailo 작업"; ps -eo pid,etime,pcpu,rss,args | grep -E "hailo (optimize|compiler|parser)" | grep -v grep | awk '{printf "  PID %s  경과 %s  CPU %s%%  메모리 %.1fGB  %s %s %s\n",$1,$2,$3,$4/1048576,$6,$7,$8}' || true
echo "== 만들어진 결과"; ls -la $W/opt/*.har $W/hef/*.hef 2>/dev/null | awk '{print "  "$6" "$7" "$8"  "$9}'
for f in $(ls -t $W/logs/*_optimize.log $W/logs/*_compile.log 2>/dev/null | head -1); do
  echo "== 최근 로그: $f (마지막 8줄)"; tail -8 "$f" | cut -c1-160
done
echo "== 메모리"; free -g | awk 'NR<=2'
