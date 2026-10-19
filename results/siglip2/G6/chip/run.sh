# G6 실칩 (AGX Hailo-10H): parse-hef → dev 전체 eval (HEF마다, 세션마다 냉각) → 지연 배치 1·8 각 3세션 (모든 HEF 번갈아) → run2 -t 10
# G5c run.sh와 같은 규칙: 다른 Hailo 작업 0개일 때 시작(최대 3시간), 세션마다 칩 온도 max(ts0, ts1) < TH (최대 90분).
# TH = 쉬는 온도 62.6 °C(G3RQ) + 2 = 64.6 고정 (README 8절 사후 변경 — 시작 온도로 문턱을 올리던 규칙은 폐기).
# 사용: HEFS="b32_Z b32_V ..." nohup bash run.sh > run.out 2>&1 < /dev/null &   (모든 대기에는 상한이 있다)
export OPENBLAS_NUM_THREADS=1
cd ~/ssd/junyong/hailo_g6
O=out; T=$O/temps.jsonl; L=$O/run.log; mkdir -p $O
HEFS=${HEFS:?}; SKIP_EVAL=${SKIP_EVAL:-0}; SKIP_BENCH=${SKIP_BENCH:-0}
log() { echo "$(date +%T) $*" | tee -a $L; }
temp() { timeout 60 python3 s2_hailo_temp.py "$1" 2>/dev/null | grep '^{' | tee -a $T; }
others() { ps aux | grep -iE "hailortcli|hailo_chip|hailo_bench|hailo_platform" | grep -v grep | grep -v "$$" | wc -l; }
t0w=$(date +%s); while [ "$(others)" -gt 0 ]; do log "다른 Hailo 작업 대기 $(others)개"; [ $(( $(date +%s) - t0w )) -gt 10800 ] && { log "BUSY 3시간 넘김 — 멈춤"; exit 1; }; sleep 120; done
j=$(temp "idle_start"); IDLE=$(echo "$j" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(max(d['ts0_c'], d['ts1_c']))" 2>/dev/null || echo 62.6)
TH=64.6   # 고정 (G5와 같음). 10/20 첫 속도 실행에서 'max(62.6, 시작 온도)+2'가 eval 직후의 뜨거운 시작 온도(약 83 °C)를 쉬는 온도로 잡아 냉각이 꺼졌다 → 칩 103 °C·드라이버 오류. 시작 온도는 기록만 한다
log "start commit=$(cat COMMIT 2>/dev/null) hailort=$(hailortcli --version | tail -1) script_sha=$(sha256sum g6_hailo_chip.py run.sh | cut -c1-16 | tr '\n' ' ')load=$(cut -d' ' -f1-3 /proc/loadavg) users=$(who | wc -l) other_hailo=$(others) idle=$IDLE TH=$TH data_sha=$(sha256sum dev_zoo*/*.npy | cut -c1-12 | tr '\n' ' ')"
wait_cool() {
  local t0=$(date +%s) j m
  while true; do
    j=$(temp "wait_$1"); m=$(echo "$j" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(max(d['ts0_c'], d['ts1_c']))" 2>/dev/null || echo 999)
    if python3 -c "import sys; sys.exit(0 if float('$m') < $TH else 1)"; then log "cool $1 max=$m"; return 0; fi
    if [ $(( $(date +%s) - t0 )) -gt 5400 ]; then log "HOT $1 max=$m (90분 넘김 — 과열 위험이라 멈춤)"; exit 1; fi
    sleep 30
  done
}
MODELS=""; for h in $HEFS; do MODELS="$MODELS $h=$h.hef"; log "hef $h sha256=$(sha256sum $h.hef | cut -c1-64) bytes=$(stat -c %s $h.hef)"; done
for h in $HEFS; do timeout 120 hailortcli parse-hef $h.hef > $O/parsehef_$h.txt 2>&1; done
if [ "$SKIP_EVAL" != 1 ]; then
  for m in $HEFS; do
    wait_cool "eval_$m"
    log "other_hailo=$(others) eval $m 시작"
    timeout 7200 python3 -u g6_hailo_chip.py eval --data . --out $O/eval_$m --models $m=$m.hef >> $L 2>&1 || log "EVAL FAIL $m"
    temp "after_eval_$m" > /dev/null; log "eval $m 끝"
  done
fi
if [ "$SKIP_BENCH" != 1 ]; then
  for b in 1 8; do
    for s in 1 2 3; do
      wait_cool "bench_b${b}_s$s"
      log "other_hailo=$(others) bench b$b s$s 시작"
      timeout 3600 python3 -u g6_hailo_chip.py bench --data . --out $O/bench_b${b}_s$s --batch $b --warmup 20 --n 200 --label "b${b}_s$s" --models $MODELS >> $L 2>&1 || log "BENCH FAIL b$b s$s"
      temp "after_bench_b${b}_s$s" > /dev/null; log "bench b$b s$s 끝"
    done
  done
  for b in 1 8; do
    for h in $HEFS; do
      wait_cool "run2_b${b}_$h"
      log "run2 b$b $h $(timeout 120 hailortcli run2 -t 10 -j $O/run2_b${b}_$h.json set-net $h.hef --batch-size $b 2>&1 | tr '\r' '\n' | grep fps | tail -1)"
      temp "after_run2_b${b}_$h" > /dev/null
    done
  done
fi
log "G6 CHIP DONE load=$(cut -d' ' -f1-3 /proc/loadavg)"
