#!/usr/bin/env bash
# speed_after_g4.sh — G4 DFC 작업이 끝난 뒤 WSL을 내리고 G3 후보 지연을 잰다 (run_when_idle.ps1이 PC가 한가할 때 실행)
wsl.exe --terminate Ubuntu-22.04
cd /d/mokerton/oslab_mobile/siglip2 && bash scripts/g3_cumulative.sh speed
