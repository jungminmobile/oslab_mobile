# b32t_hotwatch.sh — (AGX) README B-4: eval 중 15초마다 칩 온도를 읽어 max(ts0, ts1) > HOT(95) 이면 즉시 eval을 죽이고 HOT_STOP 파일을 남긴다.
# 읽기 오류도 기록한다. STOP_WATCH 파일이 생기면 끝난다. 사용: setsid nohup bash b32t_hotwatch.sh > /dev/null 2>&1 < /dev/null &
cd ~/ssd/junyong/hailo_b32t; mkdir -p out; W=out/hotwatch.jsonl; HOT=${HOT:-95}
while [ ! -f STOP_WATCH ]; do
  j=$(timeout 30 python3 s2_hailo_temp.py hotwatch 2>/dev/null | grep '^{')
  echo "$j" >> $W
  m=$(echo "$j" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(max(d['ts0_c'], d['ts1_c']))" 2>/dev/null || echo err)
  if [ "$m" != err ] && python3 -c "import sys; sys.exit(0 if float('$m') > $HOT else 1)"; then
    touch HOT_STOP; pkill -f "g6_hailo_chip.py eval"; echo "{\"time\": \"$(date +%T)\", \"label\": \"HOT_STOP_KILL\", \"max\": $m}" >> $W; break
  fi
  sleep 15
done
