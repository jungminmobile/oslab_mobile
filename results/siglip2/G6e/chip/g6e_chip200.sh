# g6e_chip200.sh — G6e 실칩 dev 6×200 (HEF마다): 다른 Hailo 작업이 0개일 때까지 기다림(최대 3시간) → 64.6 °C 미만 냉각(최대 90분) → eval --k 200 → 95 °C 넘으면 멈춤
# G6 run.sh의 others·temp·wait_cool 규칙을 그대로 쓴다 (AGX ~/ssd/junyong/hailo_g6). 다른 세션(G6f)이 칩을 같이 쓰므로 HEF마다 다시 확인한다.
# 사용: HEFS="b16_g6e_E3" nohup bash g6e_chip200.sh > g6e_chip200.out 2>&1 < /dev/null &
export OPENBLAS_NUM_THREADS=1
cd ~/ssd/junyong/hailo_g6
O=out; T=$O/temps.jsonl; L=$O/g6e.log; mkdir -p $O
HEFS=${HEFS:?}; TH=64.6; HOT=95
log() { echo "$(date '+%F %T') $*" | tee -a $L; }
temp() { timeout 60 python3 s2_hailo_temp.py "$1" 2>/dev/null | grep '^{' | tee -a $T; }
tmax() { python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(max(d['ts0_c'], d['ts1_c']))" 2>/dev/null || echo 999; }
others() { ps aux | grep -iE "hailortcli|hailo_chip|hailo_bench|hailo_platform" | grep -v grep | grep -v "$$" | wc -l; }
wait_free() {
  local t0=$(date +%s)
  while [ "$(others)" -gt 0 ]; do log "다른 Hailo 작업 대기 $(others)개 ($1)"; [ $(( $(date +%s) - t0 )) -gt 10800 ] && { log "BUSY 3시간 넘김 — 멈춤"; exit 1; }; sleep 120; done
}
wait_cool() {
  local t0=$(date +%s) m
  while true; do
    m=$(temp "wait_$1" | tmax)
    if python3 -c "import sys; sys.exit(0 if float('$m') < $TH else 1)"; then log "cool $1 max=$m"; return 0; fi
    [ $(( $(date +%s) - t0 )) -gt 5400 ] && { log "HOT $1 max=$m (90분 넘김 — 멈춤)"; exit 1; }
    sleep 30
  done
}
log "g6e start commit=$(cat COMMIT_G6E 2>/dev/null) hailort=$(hailortcli --version | tail -1) script_sha=$(sha256sum g6_hailo_chip.py g6e_chip200.sh | cut -c1-16 | tr '\n' ' ')HEFS=$HEFS"
for m in $HEFS; do
  log "hef $m sha256=$(sha256sum $m.hef | cut -c1-64) bytes=$(stat -c %s $m.hef)"
  timeout 120 hailortcli parse-hef $m.hef > $O/parsehef_$m.txt 2>&1
  wait_free "eval200_$m"; wait_cool "eval200_$m"; wait_free "eval200_$m (냉각 뒤)"
  log "other_hailo=$(others) eval200 $m 시작"
  timeout 3600 python3 -u g6_hailo_chip.py eval --data . --out $O/eval200_$m --k 200 --models $m=$m.hef > $O/eval200_$m.log 2>&1 || { log "EVAL FAIL $m"; tail -5 $O/eval200_$m.log | tee -a $L; exit 1; }
  m2=$(temp "after_eval200_$m" | tmax); log "eval200 $m 끝 max=$m2"
  python3 -c "import sys; sys.exit(0 if float('$m2') > $HOT else 1)" && { log "HOT_STOP after_eval200_$m max=$m2 > $HOT — 멈춤"; exit 2; }
done
log "G6E CHIP200 DONE"
