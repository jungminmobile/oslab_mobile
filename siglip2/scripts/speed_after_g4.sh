#!/usr/bin/env bash
# speed_after_g4.sh — G3 후보 지연 측정 (PC가 한가할 때 실행)
# 공용 잠금을 잡고 잰다(개인연구 세션·밤샘 작업과 겹치지 않게).
# 주의: 잠금 도구(Windows Python)가 그냥 "bash"를 실행하면 Windows는 C:\Windows\System32\bash.exe(WSL bash)를 먼저 찾아,
#       Windows 환경 변수(SIGLIP2_PYTHON 등)가 전달되지 않는다 (10/14 측정 2회 실패 원인) → Git Bash 전체 경로를 쓴다.
GITBASH="C:/Program Files/Git/bin/bash.exe"
cd /d/mokerton/oslab_mobile/siglip2 && "$SIGLIP2_PYTHON" D:/mokerton/locks/heavy_lock.py run --owner siglip2 \
    --job "SigLIP2 latency benchmark (needs idle PC)" -- "$GITBASH" scripts/g3_cumulative.sh speed
