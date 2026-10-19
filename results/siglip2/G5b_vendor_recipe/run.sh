# G5b 실칩 (AGX Hailo-10H): dev 스크리닝 6×200 eval (M2rot2, 벤더) → 지연 배치 1·8 각 3세션 (두 HEF 번갈아) → run2 -t 10 (배치 1·8)
# 세션마다 칩 온도 max(ts0, ts1) < 64.6 °C 대기 (최대 90분, G3RQ repro/chip/run.sh·G5 run.sh와 같음). nohup bash run.sh > run.out 2>&1 < /dev/null &
export OPENBLAS_NUM_THREADS=1
cd ~/ssd/junyong/hailo_g5b
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
others() { ps aux | grep -iE "hailortcli|hailo_chip|hailo_bench|hailo_platform" | grep -v grep | grep -v "$$" | wc -l; }
# 연구실 공용 AGX + G5 세션: 다른 Hailo 작업이 0개가 될 때까지 기다린다 (최대 3시간, 넘으면 멈춤)
t0w=$(date +%s); while [ "$(others)" -gt 0 ]; do log "다른 Hailo 작업 대기 $(others)개"; [ $(( $(date +%s) - t0w )) -gt 10800 ] && { log "BUSY 3시간 넘김 — 멈춤"; exit 1; }; sleep 120; done
H="M2rot2_D256L1 vendor_v500"
MODELS=""; for h in $H; do MODELS="$MODELS $h=$h.hef"; done
log "start commit=$(cat COMMIT 2>/dev/null) hailort=$(hailortcli --version | tail -1) script_sha=$(sha256sum g5b_hailo_chip.py g5_hailo_bench.py run.sh | cut -c1-16 | tr '\n' ' ')load=$(cut -d' ' -f1-3 /proc/loadavg) users=$(who | wc -l) other_hailo=$(others) $(for h in $H; do echo -n "$h=$(sha256sum -L $h.hef | cut -c1-16) "; done)"
# 1) 정확도: dev 스크리닝 6×200 (devfull x.npy 앞 200장), 같은 실행에서 두 HEF
for m in $H; do
  wait_cool "eval_$m"
  log "other_hailo=$(others) eval $m 시작"
  python3 -u g5b_hailo_chip.py eval --x-npy --k 200 --pixel-models vendor_v500 --data ../hailo_repro/devfull --out out/eval_$m --models $m=$m.hef >> $L 2>&1 || log "EVAL FAIL $m"
  temp "after_eval_$m" > /dev/null; log "eval $m 끝"
done
# 2) 지연: 세션 = 두 HEF를 M2rot2 → 벤더 순서로 한 번씩. 배치 1 세션 1~3, 배치 8 세션 1~3 (세션마다 냉각)
for b in 1 8; do
  for s in 1 2 3; do
    wait_cool "bench_b${b}_s$s"
    log "other_hailo=$(others) bench b$b s$s 시작"
    python3 -u g5_hailo_bench.py --data ../siglip2_xplat/dev200 --out out/bench_b${b}_s$s --batch $b --warmup 20 --n 200 --label "b${b}_s$s" --models $MODELS >> $L 2>&1 || log "BENCH FAIL b$b s$s"
    temp "after_bench_b${b}_s$s" > /dev/null; log "bench b$b s$s 끝"
  done
done
# 3) run2 -t 10 (보조, FPS)
for b in 1 8; do
  for h in $H; do
    wait_cool "run2_b${b}_$h"
    log "run2 b$b $h $(hailortcli run2 -t 10 -j out/run2_b${b}_$h.json set-net $h.hef --batch-size $b 2>&1 | tr '\r' '\n' | grep fps | tail -1)"
    temp "after_run2_b${b}_$h" > /dev/null
  done
done
log "G5B CHIP DONE load=$(cut -d' ' -f1-3 /proc/loadavg)"
