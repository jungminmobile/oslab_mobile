# G5 Zoo 비교 실칩 (AGX Hailo-10H): 지연 배치 1·배치 8 각 3세션 (HEF 3개 번갈아) → run2 -t 10 (배치 1·8) → hailortcli benchmark --batch-size 1 (HEF 4개) → 런타임 데이터 수집 (기록용)
# 세션마다 칩 온도 max(ts0, ts1) < 64.6 °C 대기 (최대 90분, G3RQ repro/chip/run.sh와 같음). nohup으로 실행: nohup bash run.sh > run.out 2>&1 &
export OPENBLAS_NUM_THREADS=1
cd ~/ssd/junyong/hailo_g5
T=out/temps.jsonl; L=out/run.log; mkdir -p out
TH=64.6
TEMP=../hailo_repro/s2_hailo_temp.py
log() { echo "$(date +%T) $*" | tee -a $L; }
temp() { python3 $TEMP "$1" 2>/dev/null | grep '^{' | tee -a $T; }
wait_cool() {   # $1 = 라벨
  local t0=$(date +%s) j m
  while true; do
    j=$(temp "wait_$1"); m=$(echo "$j" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(max(d['ts0_c'], d['ts1_c']))" 2>/dev/null || echo 999)
    if python3 -c "import sys; sys.exit(0 if float('$m') < $TH else 1)"; then log "cool $1 max=$m"; return 0; fi
    if [ $(( $(date +%s) - t0 )) -gt 5400 ]; then log "HOT $1 max=$m (90분 넘김, 진행)"; return 1; fi
    sleep 30
  done
}
others() { ps aux | grep -iE "hailortcli|s2_hailo_chip|g5_hailo_bench|hailo_platform" | grep -v grep | grep -v "$$" | wc -l; }
t0w=$(date +%s); while [ "$(others)" -gt 0 ]; do log "다른 Hailo 작업 대기 $(others)개"; [ $(( $(date +%s) - t0w )) -gt 10800 ] && { log "BUSY 3시간 넘김 — 멈춤"; exit 1; }; sleep 120; done
H="M2rot2_D256L1 rothad2_D256L1 siglip_b_16_image_encoder"
MODELS=""; for h in $H; do MODELS="$MODELS $h=$h.hef"; done
log "start commit=$(cat COMMIT 2>/dev/null) script_sha=$(sha256sum g5_hailo_bench.py run.sh | cut -c1-16 | tr '\n' ' ')load=$(cut -d' ' -f1-3 /proc/loadavg) users=$(who | wc -l) other_hailo=$(others) $(for h in $H siglip2_b_32_256_image_encoder; do echo -n "$h=$(sha256sum -L $h.hef | cut -c1-16) "; done)"
# 1) 지연: 세션 = HEF 3개를 정해진 순서로 한 번씩. 배치 1 세션 1~3, 배치 8 세션 1~3 (세션마다 냉각)
for b in 1 8; do
  for s in 1 2 3; do
    wait_cool "bench_b${b}_s$s"
    log "other_hailo=$(others) bench b$b s$s 시작"
    python3 -u g5_hailo_bench.py --data ../siglip2_xplat/dev200 --out out/bench_b${b}_s$s --batch $b --warmup 20 --n 200 --label "b${b}_s$s" --models $MODELS >> $L 2>&1 || log "BENCH FAIL b$b s$s"
    temp "after_bench_b${b}_s$s" > /dev/null; log "bench b$b s$s 끝"
  done
done
# 2) run2 -t 10 (보조, FPS)
for b in 1 8; do
  for h in $H; do
    wait_cool "run2_b${b}_$h"
    log "run2 b$b $h $(hailortcli run2 -t 10 -j out/run2_b${b}_$h.json set-net $h.hef --batch-size $b 2>&1 | tr '\r' '\n' | grep fps | tail -1)"
    temp "after_run2_b${b}_$h" > /dev/null
  done
done
# 3) hailortcli benchmark --batch-size 1 (보조, 1회씩, 로그만)
for h in $H siglip2_b_32_256_image_encoder; do
  wait_cool "benchmark_$h"
  log "benchmark $h 시작"
  hailortcli benchmark $h.hef --batch-size 1 --csv out/benchmark_$h.csv > out/benchmark_$h.log 2>&1; log "benchmark $h rc=$?"
  temp "after_benchmark_$h" > /dev/null
done
# 4) 런타임 데이터 (기록용, 배치 1): 데스크탑 프로파일러에서 컨텍스트별 시간을 보기 위해
for h in $H; do
  wait_cool "rtdata_$h"
  hailortcli run2 -t 5 measure-fw-actions $h.hef --batch-size 1 --output-path out/runtime_data_$h.json > out/rtdata_$h.log 2>&1; log "rtdata $h rc=$?"
  temp "after_rtdata_$h" > /dev/null
done
log "G5 CHIP DONE load=$(cut -d' ' -f1-3 /proc/loadavg)"
