export OPENBLAS_NUM_THREADS=1
echo "start $(date +%T) load=$(cut -d" " -f1-3 /proc/loadavg)"
timeout 900 python3 -u s2_hailo_chip.py bench --data ../siglip2_xplat/dev200 --out out_chip --models D_D256L1=D_D256L1.hef rothad_D256L1=rothad_D256L1.hef --warmup 20 --n 200 --rounds 3
for r in 1 2; do for h in D_D256L1 rothad_D256L1; do echo "run2 $h round $r $(date +%T)"; hailortcli run2 -t 10 set-net $h.hef 2>&1 | tr "\r" "\n" | grep fps | tail -1; sleep 2; done; done
echo "done $(date +%T) load=$(cut -d" " -f1-3 /proc/loadavg)"
