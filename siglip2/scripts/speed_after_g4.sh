#!/usr/bin/env bash
# speed_after_g4.sh — G3 후보 지연 측정 (run_when_idle.ps1이 PC가 한가할 때 실행)
# WSL은 내리지 않는다: WSL 안에서 G4 스크리닝 예약(g4_screen_all.sh)이 이 측정이 끝나기를 기다리고 있고,
# 쉬고 있는 WSL은 CPU를 거의 쓰지 않는다 (측정 직전 CPU 한가함은 run_when_idle.ps1이 확인).
cd /d/mokerton/oslab_mobile/siglip2 && bash scripts/g3_cumulative.sh speed
