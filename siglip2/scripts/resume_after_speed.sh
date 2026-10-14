#!/usr/bin/env bash
# resume_after_speed.sh — 10/14 재개: 지연 측정(speed_when_idle.sh)이 끝나면
#   ① WSL G4 추가 실험(hailo/g4_more_wsl.sh: D256 스크리닝 → B·A 스크리닝 → E·C 컴파일)을 시작하고
#   ② Windows에서 새 venv 재현 테스트(scripts/g3_repro_venv.sh)를 돌린다. 둘 다 공용 잠금 순서로 번갈아 실행된다.
# WSL 쪽은 wsl.exe 클라이언트를 이 스크립트가 끝까지 붙잡고 있는다 (클라이언트가 먼저 끝나면 WSL 작업이 끊긴 일이 있었음).
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
ART="$SIGLIP2_ARTIFACTS"
until grep -q "SPEED_IDLE_DONE" $ART/logs/speed_when_idle_v2_main.log 2>/dev/null; do sleep 60; done
echo "$(date +%T) speed done -> start WSL chain and repro"
wsl.exe -d Ubuntu-22.04 -e bash -lc "tr -d '\r' < /mnt/d/mokerton/oslab_mobile/siglip2/hailo/g4_more_wsl.sh > ~/g4_more_wsl.sh && cd ~ && bash ~/g4_more_wsl.sh > /mnt/d/mokerton/artifacts/siglip2/logs/g4_more_wsl.log 2>&1" &
WPID=$!
bash scripts/g3_repro_venv.sh > $ART/logs/repro_venv.log 2>&1
echo "$(date +%T) repro rc=$? $(tail -1 $ART/logs/repro_venv.log)"
wait $WPID
echo "$(date +%T) WSL chain rc=$? $(tail -1 $ART/logs/g4_more_wsl.log)"
echo "RESUME DONE $(date +%T)"
