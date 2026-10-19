# G5b 작업 B 실칩 (AGX Hailo-10H): dev 전체 eval — M2rot2 → 냉각 → 벤더 (같은 실행). 입력은 각 HEF 형식 (벤더만 --pixel-models)
# 세션마다 칩 온도 max(ts0, ts1) < 64.6 °C 대기 (최대 90분). nohup bash run_devfull.sh > run_devfull.out 2>&1 < /dev/null &
export OPENBLAS_NUM_THREADS=1
cd ~/ssd/junyong/hailo_g5b
O=out_devfull; T=$O/temps.jsonl; L=$O/run.log; mkdir -p $O
TH=64.6
log() { echo "$(date +%T) $*" | tee -a $L; }
temp() { python3 s2_hailo_temp.py "$1" 2>/dev/null | grep '^{' | tee -a $T; }
wait_cool() {
  local t0=$(date +%s) j m
  while true; do
    j=$(temp "wait_$1"); m=$(echo "$j" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(max(d['ts0_c'], d['ts1_c']))" 2>/dev/null || echo 999)
    if python3 -c "import sys; sys.exit(0 if float('$m') < $TH else 1)"; then log "cool $1 max=$m"; return 0; fi
    if [ $(( $(date +%s) - t0 )) -gt 5400 ]; then log "HOT $1 max=$m (90분 넘김, 진행)"; return 1; fi
    sleep 30
  done
}
others() { ps aux | grep -iE "hailortcli|hailo_chip|hailo_bench|hailo_platform" | grep -v grep | grep -v "$$" | wc -l; }
t0w=$(date +%s); while [ "$(others)" -gt 0 ]; do log "다른 Hailo 작업 대기 $(others)개"; [ $(( $(date +%s) - t0w )) -gt 10800 ] && { log "BUSY 3시간 넘김 — 멈춤"; exit 1; }; sleep 120; done
H="M2rot2_D256L1 vendor_v500"
log "start commit=$(cat COMMIT 2>/dev/null) hailort=$(hailortcli --version | tail -1) script_sha=$(sha256sum g5b_hailo_chip.py run_devfull.sh | cut -c1-16 | tr '\n' ' ')load=$(cut -d' ' -f1-3 /proc/loadavg) users=$(who | wc -l) other_hailo=$(others) data_sha=$(cat ../hailo_repro/devfull/SHA256SUMS | cut -c1-12 | tr '\n' ' ')$(for h in $H; do echo -n "$h=$(sha256sum $h.hef | cut -c1-16) "; done)"
for m in $H; do
  wait_cool "eval_$m"
  log "other_hailo=$(others) eval $m 시작"
  PIX=""; [ "$m" = vendor_v500 ] && PIX="--pixel-models vendor_v500"
  python3 -u g5b_hailo_chip.py eval --x-npy $PIX --data ../hailo_repro/devfull --out $O/eval_$m --models $m=$m.hef >> $L 2>&1 || log "EVAL FAIL $m"
  temp "after_eval_$m" > /dev/null; log "eval $m 끝"
done
log "G5B DEVFULL DONE load=$(cut -d' ' -f1-3 /proc/loadavg)"
