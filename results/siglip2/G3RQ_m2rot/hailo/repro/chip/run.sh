# G3RQ 빌드 재현성 실칩: 2차 빌드 dev 전체 eval (rothad2, M2rot2) → 지연 3세션 (A B 번갈아) → run2 10초씩. 세션마다 칩 온도 < 64.6 °C 대기 (최대 90분). nohup으로 실행
export OPENBLAS_NUM_THREADS=1
cd ~/ssd/junyong/hailo_repro
T=out/temps.jsonl; L=out/run.log; mkdir -p out
TH=64.6
log() { echo "$(date +%T) $*" | tee -a $L; }
temp() { python3 s2_hailo_temp.py "$1" 2>/dev/null | grep '^{' | tee -a $T; }
wait_cool() {   # $1 = 라벨
  local t0=$(date +%s) j m
  while true; do
    j=$(temp "wait_$1"); m=$(echo "$j" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(max(d['ts0_c'], d['ts1_c']))" 2>/dev/null || echo 999)
    if python3 -c "import sys; sys.exit(0 if float('$m') < $TH else 1)"; then log "cool $1 max=$m"; return 0; fi
    if [ $(( $(date +%s) - t0 )) -gt 5400 ]; then log "HOT $1 max=$m (90분 넘김, 진행)"; return 1; fi
    sleep 30
  done
}
others() { ps aux | grep -iE "hailortcli|s2_hailo_chip|hailo_platform" | grep -v grep | grep -v "$$" | wc -l; }
# 연구실 공용 AGX: 다른 사람의 Hailo 작업이 있으면 끝날 때까지 기다린다 (최대 3시간, 넘으면 멈춤)
t0w=$(date +%s); while [ "$(others)" -gt 0 ]; do log "다른 Hailo 작업 대기 $(others)개"; [ $(( $(date +%s) - t0w )) -gt 10800 ] && { log "BUSY 3시간 넘김 — 멈춤"; exit 1; }; sleep 120; done
log "start load=$(cut -d' ' -f1-3 /proc/loadavg) users=$(who | wc -l) other_hailo=$(others) sha_rothad=$(sha256sum rothad2_D256L1.hef | cut -c1-16) sha_m2rot=$(sha256sum M2rot2_D256L1.hef | cut -c1-16)"
for m in rothad2_D256L1 M2rot2_D256L1; do
  wait_cool "eval_$m"
  log "eval $m 시작"
  python3 -u s2_hailo_chip.py eval --x-npy --data devfull --out out/eval_$m --models $m=$m.hef >> $L 2>&1 || log "EVAL FAIL $m"
  temp "after_eval_$m" > /dev/null; log "eval $m 끝"
done
for rd in 1 2 3; do
  wait_cool "bench_s$rd"
  python3 -u s2_hailo_chip.py bench --data ../siglip2_xplat/dev200 --out out/bench_s$rd --models rothad2_D256L1=rothad2_D256L1.hef M2rot2_D256L1=M2rot2_D256L1.hef --warmup 20 --n 200 --rounds 1 >> $L 2>&1 || log "BENCH FAIL s$rd"
  temp "after_bench_s$rd" > /dev/null; log "bench s$rd 끝"
done
for h in rothad2_D256L1 M2rot2_D256L1; do
  wait_cool "run2_$h"
  log "run2 $h $(hailortcli run2 -t 10 set-net $h.hef 2>&1 | tr '\r' '\n' | grep fps | tail -1)"
  temp "after_run2_$h" > /dev/null
done
log "REPRO CHIP DONE load=$(cut -d' ' -f1-3 /proc/loadavg)"
