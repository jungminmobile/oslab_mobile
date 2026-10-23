# b32t_run.sh — (AGX) README B-4: test 7종 × GF·Z 실칩 eval (G6 g6_hailo_chip.py eval --subdir test_zoo256), 데이터셋마다 냉각·다른 Hailo 작업 확인,
# eval 중 온도 감시(b32t_hotwatch.sh, 95 °C 즉시 중단). HEF·데이터셋마다 done 파일(1회).
# SMOKE=1: test 대신 dev_zoo256을 GF로 (감시 점검, out/smoke<TAG>). SMOKE_K(기본 200)·SMOKE_SETS(기본 pets_dev)·SMOKE_TAG
# 사용: setsid nohup bash b32t_run.sh > out/run.out 2>&1 < /dev/null &
export OPENBLAS_NUM_THREADS=1
cd ~/ssd/junyong/hailo_b32t
O=out; T=$O/temps.jsonl; L=$O/run.log; mkdir -p $O
SMOKE=${SMOKE:-0}
log() { echo "$(date +%T) $*" | tee -a $L; }
temp() { timeout 60 python3 s2_hailo_temp.py "$1" 2>/dev/null | grep '^{' | tee -a $T; }
tmax() { echo "$1" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(max(d['ts0_c'], d['ts1_c']))" 2>/dev/null || echo 999; }
others() { ps aux | grep -iE "hailortcli|hailo_chip|hailo_bench|hailo_platform" | grep -v grep | grep -v "$$" | wc -l; }
wait_others() { local t0=$(date +%s); while [ "$(others)" -gt 0 ]; do log "다른 Hailo 작업 대기 $(others)개 ($1)"; [ $(( $(date +%s) - t0 )) -gt 10800 ] && { log "BUSY 3시간 넘김 — 멈춤"; stop 1; }; sleep 120; done; }
TH=64.6
wait_cool() {
  local t0=$(date +%s) j m
  while true; do
    [ -f HOT_STOP ] && { log "HOT_STOP 파일 — 멈춤"; stop 2; }
    j=$(temp "wait_$1"); m=$(tmax "$j")
    if python3 -c "import sys; sys.exit(0 if float('$m') < $TH else 1)"; then log "cool $1 max=$m"; return 0; fi
    [ $(( $(date +%s) - t0 )) -gt 5400 ] && { log "HOT $1 max=$m (90분 넘김 — 멈춤)"; stop 1; }
    sleep 30
  done
}
stop() { touch STOP_WATCH; log "끝 rc=$1"; exit $1; }
declare -A WANT=([GF]=0c2e04a6ef9bfb6055c249ad969f07546711d6130ce401800fcc7fdc61c8b29a [Z]=c16123297d83a36cc862839ba1b9984d1ab539abe3773f99a332acae39e318cc)
[ -f HOT_STOP ] && { log "HOT_STOP 남아 있음 — 사람이 확인 후 지울 것"; exit 2; }
wait_others start
for h in GF Z; do s=$(sha256sum b32_$h.hef | cut -c1-64); [ "$s" = "${WANT[$h]}" ] || { log "SHA256 MISMATCH $h $s"; exit 2; }; log "hef $h sha256=$s"; done
log "start commit=$(cat COMMIT) hailort=$(hailortcli --version | tail -1) script_sha=$(sha256sum g6_hailo_chip.py b32t_run.sh b32t_hotwatch.sh s2_hailo_temp.py | cut -c1-16 | tr '\n' ' ')load=$(cut -d' ' -f1-3 /proc/loadavg) users=$(who | wc -l) SMOKE=$SMOKE"
rm -f STOP_WATCH; setsid nohup bash b32t_hotwatch.sh > /dev/null 2>&1 < /dev/null &
if [ "$SMOKE" = 1 ]; then
  wait_cool smoke
  timeout 1800 python3 -u g6_hailo_chip.py eval --data . --k ${SMOKE_K:-200} --sets ${SMOKE_SETS:-pets_dev} --out $O/smoke${SMOKE_TAG:-} --models b32_GF=b32_GF.hef >> $L 2>&1; rc=$?
  temp after_smoke > /dev/null; log "smoke rc=$rc"; sleep 20; stop $rc
fi
log "test data sha=$(sha256sum test_zoo256/*.npy | cut -c1-12 | tr '\n' ' ')"
CH=${CH:-2000}   # (사후 변경, README B-4) 덩어리 장수 — 덩어리마다 냉각
for h in GF Z; do
  for d in pets caltech flowers cars food cifar_test imagenetv2; do
    D=$O/done_${h}_$d.json
    [ -f $D ] && { log "skip $h $d (done)"; continue; }
    N=$(python3 -c "import numpy as np; print(len(np.load('test_zoo256/${d}_y.npy')))")
    P=$O/eval_$h/part_$d
    for ((i=0; i<N; i+=CH)); do
      C=$P/c_$i; [ -f $C/done.json ] && { log "skip $h $d chunk $i (done)"; continue; }
      wait_others "eval_${h}_${d}_$i"; wait_cool "eval_${h}_${d}_$i"
      log "other_hailo=$(others) eval $h $d chunk $i/$N 시작"
      timeout 3600 python3 -u g6_hailo_chip.py eval --data . --subdir test_zoo256 --sets $d --start $i --k $CH --out $C --models b32_$h=b32_$h.hef >> $L 2>&1; rc=$?
      j=$(temp "after_eval_${h}_${d}_$i"); log "eval $h $d chunk $i rc=$rc after_max=$(tmax "$j")"
      [ -f HOT_STOP ] && { log "HOT_STOP (eval 중 95 °C 넘음) — 멈춤"; stop 2; }
      [ $rc = 0 ] || { log "EVAL FAIL $h $d $i"; stop 1; }
      echo "{\"date\": \"$(date +%FT%T)\"}" > $C/done.json
    done
    python3 - "$P" "$h" "$d" "$N" "$CH" <<'PY' >> $L 2>&1 || { log "MERGE FAIL $h $d"; stop 1; }
import json, sys, numpy as np
from pathlib import Path
P, h, d, N, CH = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
parts, infos = [], []
for i in range(0, N, CH):
    parts.append(np.load(P / f"c_{i}" / f"emb_b32_{h}.npz")[f"{d}_emb"]); infos.append(json.loads((P / f"c_{i}" / "eval_info.json").read_text()))
e = np.concatenate(parts); assert e.shape[0] == N, (e.shape, N)
assert len({i["models"][f"b32_{h}"]["hef_sha256"] for i in infos}) == 1
np.savez_compressed(P / f"emb_b32_{h}.npz", **{f"{d}_emb": e})
info = infos[0]; info["models"][f"b32_{h}"]["sec"] = sum(i["models"][f"b32_{h}"]["sec"] for i in infos)
info["chunks"] = {str(i * CH): x["date"] for i, x in enumerate(infos)}; info["n"] = N
(P / "eval_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1)); print("merged", h, d, e.shape)
PY
    echo "{\"date\": \"$(date +%FT%T)\", \"hef\": \"$h\", \"set\": \"$d\", \"n\": $N}" > $D; log "done $h $d n=$N"
  done
done
log "ALL DONE"; stop 0
