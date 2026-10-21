# g6f_watch.sh — (AGX) 30초마다 Hailo 관련 프로세스(hailortcli, run.sh, 홈 아래 python) 중 작업 폴더가 hailo_g6f가 아닌 것을 적는다 (README 2-3).
# 패턴은 [x] 꼴로 써서 run.sh의 others()(ps | grep hailortcli…)에 이 감시 자체가 잡히지 않게 한다.
# 줄 형식: <시각> foreign=<개수> [pid cwd cmd] ... STOP_WATCH 파일이 생기면 멈춘다. 사용: nohup bash g6f_watch.sh > /dev/null 2>&1 < /dev/null &
cd ~/ssd/junyong/hailo_g6f; mkdir -p out; W=out/watch.log
while [ ! -f STOP_WATCH ]; do
  f=""; n=0
  for p in $(pgrep -f "hailo[r]tcli|[r]un\.sh|[p]ython"); do
    c=$(readlink /proc/$p/cwd 2>/dev/null) || continue
    case "$c" in "$HOME"/ssd/junyong/hailo_g6f) continue;; "$HOME"*) ;; *) continue;; esac
    cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-120); [ -z "$cmd" ] && continue
    n=$((n+1)); f="$f [$p $c $cmd]"
  done
  echo "$(date +%T) foreign=$n$f" >> $W; sleep 30
done
