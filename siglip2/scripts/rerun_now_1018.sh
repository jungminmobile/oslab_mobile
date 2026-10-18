#!/usr/bin/env bash
# rerun_now_1018.sh — (10/18 02:35 쭈리: "폰 측정 지금 안 하고 있으니까 진행하자") 여유 메모리 9 GB 조건 없이 지금 바로 DFC 재실행
#  개인연구 폰 러너는 멈춰 있고 이 작업이 끝날 때까지 재시작을 미루기로 함. 시스템 보호용 감시(mem_guard, 여유 0.5 GB 미만 20초면 WSL 종료)는 유지.
#  끝나면(성공·실패 모두) WSL을 내리고 "RERUN_NOW DONE"을 찍는다 → 개인연구 세션이 그 뒤 폰 러너를 띄움
set -uo pipefail
PY="$SIGLIP2_PYTHON"; L=/d/mokerton/artifacts/siglip2/logs/rerun_now_1018.log
finish() { wsl.exe -t Ubuntu-22.04 >/dev/null 2>&1; echo "$(date +%T) WSL terminated" >> $L; echo "RERUN_NOW DONE $(date +%T)" >> $L; }
trap finish EXIT
[ -f /d/mokerton/locks/siglip2.STOP ] && { echo "$(date +%T) STOP 파일 — 시작 안 함" >> $L; exit 3; }
echo "$(date +%T) start avail=$("$PY" -c 'import psutil; print(round(psutil.virtual_memory().available/2**30,1))')GB" >> $L
"$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "SigLIP2 DFC optimize rerun D256L1 (WSL ~1h)" -- \
  wsl.exe -d Ubuntu-22.04 -e bash -lc "tr -d '\r' < /mnt/d/mokerton/oslab_mobile/siglip2/hailo/crosscheck/self_rerun_wsl.sh > ~/self_rerun_wsl.sh && bash ~/self_rerun_wsl.sh" >> $L 2>&1
echo "$(date +%T) self rerun rc=$?" >> $L
