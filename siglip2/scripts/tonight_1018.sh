#!/usr/bin/env bash
# tonight_1018.sh — (10/17 쭈리: "오늘 밤 PC 안 쓸 때 돌리자") 01:00~07:00 사이, 전체 CPU가 60초 평균 20% 미만일 때만
#  공용 잠금 안에서 hailo/crosscheck/self_rerun_wsl.sh 실행 (WSL 약 1시간, 메모리 많이 씀) → 끝나면 WSL 종료. STOP 파일이면 멈춤.
set -uo pipefail
PY="$SIGLIP2_PYTHON"; STOP=/d/mokerton/locks/siglip2.STOP
while true; do
  [ -f $STOP ] && { echo "$(date +%T) STOP 파일 — 멈춤"; exit 3; }
  h=$(date +%H)
  if [ "$h" -ge 1 ] && [ "$h" -lt 7 ]; then
    c=$("$PY" -c 'import psutil; print(round(psutil.cpu_percent(interval=60)))')
    echo "$(date +%T) cpu60s=$c%"
    [ "$c" -lt 20 ] && break
  fi
  sleep 300
done
"$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "SigLIP2 DFC optimize rerun D256L1 (WSL ~1h, 10 GB)" -- \
  wsl.exe -d Ubuntu-22.04 -e bash -lc "tr -d '\r' < /mnt/d/mokerton/oslab_mobile/siglip2/hailo/crosscheck/self_rerun_wsl.sh > ~/self_rerun_wsl.sh && bash ~/self_rerun_wsl.sh"
echo "$(date +%T) self rerun rc=$?"
wsl.exe -t Ubuntu-22.04; echo "$(date +%T) WSL terminated"
echo "TONIGHT_1018 DONE $(date +%T)"
