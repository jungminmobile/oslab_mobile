#!/usr/bin/env bash
# speed_after_g4.sh — G3 후보 지연 측정 (run_when_idle.ps1이 PC가 한가할 때 실행)
# 공용 잠금을 잡고 잰다(개인연구 세션·밤샘 작업과 겹치지 않게). WSL은 내리지 않는다(쉬는 WSL은 CPU를 거의 안 씀).
cd /d/mokerton/oslab_mobile/siglip2 && "$SIGLIP2_PYTHON" D:/mokerton/locks/heavy_lock.py run --owner siglip2 \
    --job "SigLIP2 latency benchmark (needs idle PC)" -- bash scripts/g3_cumulative.sh speed
