# G3Q-H 8절: dev 전체 실칩 eval + 온도 + 지연 3회 (회차별 프로세스, 회차 전후 온도). nohup으로 실행
export OPENBLAS_NUM_THREADS=1
cd ~/ssd/junyong/hailo_g3q
T=out_full/temps.jsonl; mkdir -p out_full
echo "start $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg)"
python3 s2_hailo_temp.py before_eval 2>/dev/null | grep '^{' >> $T
for m in D_D256L1 rothad_D256L1; do
  python3 -u s2_hailo_chip.py eval --x-npy --data devfull --out out_full/eval_$m --models $m=$m.hef || echo "EVAL FAIL $m"
  python3 s2_hailo_temp.py after_eval_$m 2>/dev/null | grep '^{' >> $T
done
sleep 5
for rd in 1 2 3; do
  python3 s2_hailo_temp.py before_bench_r$rd 2>/dev/null | grep '^{' >> $T
  python3 -u s2_hailo_chip.py bench --data ../siglip2_xplat/dev200 --out out_full/bench_r$rd --models D_D256L1=D_D256L1.hef rothad_D256L1=rothad_D256L1.hef --warmup 20 --n 200 --rounds 1 || echo "BENCH FAIL $rd"
  python3 s2_hailo_temp.py after_bench_r$rd 2>/dev/null | grep '^{' >> $T
done
cat $T
echo "done $(date +%T) load=$(cut -d' ' -f1-3 /proc/loadavg)"
