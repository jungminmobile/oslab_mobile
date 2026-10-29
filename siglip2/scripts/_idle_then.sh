#!/usr/bin/env bash
# _idle_then.sh — 잠금 안에서 전체 CPU 30초 평균이 25% 미만일 때만 뒤 명령을 실행, 아니면 75로 끝냄 (지연 측정용)
c=$("$SIGLIP2_PYTHON" -c 'import psutil; print(round(psutil.cpu_percent(interval=30)))')
echo "in-lock cpu30s=$c%"
[ "$c" -ge 25 ] && { echo BUSY_RETRY; exit 75; }
"$@"
