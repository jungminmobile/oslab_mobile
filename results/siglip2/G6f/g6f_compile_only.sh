#!/usr/bin/env bash
# g6f_compile_only.sh — G6d g6d_compile_only.sh 그대로 (build.jsonl만 G6f 폴더). 이미 optimize가 끝난 HAR에서 compile만: <tag> <optimized HAR> <alls> <상한 초>
# 하나 끝날 때마다 build.jsonl에 기록. 실패해도 다음으로 넘어간다. 사용: nohup bash g6d_compile_only.sh <tag> <har> <alls> <초> >> log 2>&1 &
set -uo pipefail
source ~/hailo10_venv/bin/activate
export CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1
H=$(cd "$(dirname "$0")" && pwd); REPO=$(git -C $H rev-parse --show-toplevel); W=$HOME/siglip2_work/g6; LOCK=$HOME/siglip2_work/dfc.lock
B=$H/build.jsonl
one() {   # tag src_har alls timeout_sec
  local tag=$1 srchar=$2 alls=$3 tsec=$4 D=$W/dfc/$1
  mkdir -p $D/comp; cd $D; L=$D/run_info.txt
  log() { echo "$(date +%T) $*" | tee -a $L; }
  log "tag=$tag commit=$(git -C $REPO rev-parse --short HEAD) dirty=$(git -C $REPO status --porcelain --untracked-files=no | wc -l) src_har=$srchar src_har_sha256=$(sha256sum $srchar | cut -c1-64) alls=$alls alls_sha256=$(sha256sum $alls | cut -c1-64) (optimize 안 함, compile만)"
  cp $alls model.alls
  ( flock 9
    while [ $(free -g | awk '/Mem:/{print $7}') -lt 20 ]; do log "compile 대기: free=$(free -g | awk '/Mem:/{print $7}')G"; sleep 120; done
    log "compile 잠금 잡음 free=$(free -g | awk '/Mem:/{print $7}')G load=$(cut -d' ' -f1-3 /proc/loadavg)"
    t0=$(date +%s); /usr/bin/time -v -o time_compile.txt timeout $tsec bash -c "cd comp && hailo compiler $srchar --hw-arch hailo10h --model-script ../model.alls --output-dir . --output-har-path compiled.har" > compile.log 2>&1; rc=$?
    log "compile rc=$rc sec=$(( $(date +%s) - t0 )) maxrss_kb=$(awk -F': ' '/Maximum resident/{print $2}' time_compile.txt) $( [ $rc -eq 124 ] && echo "TIMEOUT ${tsec}s")"; exit $rc ) 9>$LOCK
  local rc=$?
  mkdir -p $H/logs/$tag; cp $L model.alls $H/logs/$tag/; grep -a -E "\[error\]|Applying selected partition|Partitioner finished|Successful" compile.log | sort -u > $H/logs/$tag/compile_errors.txt; tail -c 150000 compile.log > $H/logs/$tag/compile_tail.log
  local hef=$(ls comp/*.hef 2>/dev/null | head -1)
  python3 - "$tag" "$rc" "$hef" "$D" >> $B <<'PY'
import sys, json, hashlib, re, os, datetime
tag, rc, hef, d = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
info = open(f"{d}/run_info.txt").read()
row = {"date": datetime.datetime.now().isoformat(timespec="seconds"), "what": "compile_only", "tag": tag, "rc": rc, "timeout": rc == 124,
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
one "$1" "$2" "$3" "$4"
echo "G6F COMPILE ONLY DONE $1 $(date +%T)"
