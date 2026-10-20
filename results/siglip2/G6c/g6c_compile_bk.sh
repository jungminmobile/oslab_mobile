#!/usr/bin/env bash
# g6c_compile_bk.sh — 이미 optimize가 끝난 HAR(B16_rb, B16_both)을 다시 optimize하지 않고, bucket 3줄을 더한 model script로 compile만 다시 (용 지시)
# compile 상한 3시간 (timeout 10800). 하나 끝날 때마다 build.jsonl에 기록. 실패해도 다음으로 넘어간다 (둘은 서로 독립).
# 사용: nohup bash g6c_compile_bk.sh > ~/siglip2_work/g6/g6c_compile_bk.log 2>&1 < /dev/null &
set -uo pipefail
source ~/hailo10_venv/bin/activate
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1
H=$(cd "$(dirname "$0")" && pwd); REPO=$(git -C $H rev-parse --show-toplevel); W=$HOME/siglip2_work/g6; LOCK=$HOME/siglip2_work/dfc.lock
B=$H/build.jsonl
one() {   # tag src_tag alls
  local tag=$1 src=$W/dfc/$2 alls=$3 D=$W/dfc/$1
  mkdir -p $D/comp; cd $D; L=$D/run_info.txt
  log() { echo "$(date +%T) $*" | tee -a $L; }
  log "tag=$tag commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -C $REPO status --porcelain --untracked-files=no | wc -l) src_har=$src/siglip2_m2rot_optimized.har src_har_sha256=$(sha256sum $src/siglip2_m2rot_optimized.har | cut -c1-64) alls=$alls alls_sha256=$(sha256sum $alls | cut -c1-64) (optimize 안 함, compile만)"
  cp $alls model.alls
  ( flock 9
    while [ $(free -g | awk '/Mem:/{print $7}') -lt 20 ]; do log "compile 대기: free=$(free -g | awk '/Mem:/{print $7}')G"; sleep 120; done
    log "compile 잠금 잡음 free=$(free -g | awk '/Mem:/{print $7}')G load=$(cut -d' ' -f1-3 /proc/loadavg)"
    t0=$(date +%s); /usr/bin/time -v -o time_compile.txt timeout 10800 bash -c "cd comp && hailo compiler $src/siglip2_m2rot_optimized.har --hw-arch hailo10h --model-script ../model.alls --output-dir . --output-har-path siglip2_m2rot_compiled.har" > compile.log 2>&1; rc=$?
    log "compile rc=$rc sec=$(( $(date +%s) - t0 )) maxrss_kb=$(awk -F': ' '/Maximum resident/{print $2}' time_compile.txt) $( [ $rc -eq 124 ] && echo 'TIMEOUT 3h')"; exit $rc ) 9>$LOCK
  local rc=$?
  mkdir -p $H/logs/$tag; cp $L model.alls $H/logs/$tag/; grep -a -E "\[error\]|Applying selected partition|Partitioner finished|Successful" compile.log | sort -u > $H/logs/$tag/compile_errors.txt; tail -c 150000 compile.log > $H/logs/$tag/compile_tail.log
  local hef=$(ls comp/*.hef 2>/dev/null | head -1)
  python3 - "$tag" "$rc" "$hef" "$D" >> $B <<'PY'
import sys, json, hashlib, re, os, datetime
tag, rc, hef, d = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
info = open(f"{d}/run_info.txt").read()
row = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "what": "compile_only", "tag": tag, "rc": rc, "timeout_3h": rc == 124,
       "commit_dirty": re.findall(r"commit=(\S+) dirty=(\S+)", info)[:1], "src_har_sha256": re.findall(r"src_har_sha256=(\w+)", info)[0],
       "alls_sha256": re.findall(r"alls_sha256=(\w+)", info)[0],
       "compile": [{"rc": int(m[0]), "sec": int(m[1])} for m in re.findall(r"compile rc=(\d+) sec=(\d+)", info)]}
if hef:
    row.update(hef=hef.replace(os.path.expanduser("~"), "~"), bytes=os.path.getsize(hef), sha256=hashlib.sha256(open(hef, "rb").read()).hexdigest(),
               contexts=len(set(re.findall(r"_context_(\d+)", open(f"{d}/compile.log").read()))))
print(json.dumps(row, ensure_ascii=False))
PY
  tail -1 $B
}
one b16_rb_bk b16_rb $H/alls/g6c_b16_rb_bk.alls
one b16_both_bk b16_both $H/alls/g6c_b16_both_bk.alls
echo "G6C COMPILE BK DONE $(date +%T)"
