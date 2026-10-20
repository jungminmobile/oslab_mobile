#!/usr/bin/env bash
# launch_g4_more.sh — 지연 재측정이 끝나고 보정 256장 파일이 생기면 WSL에서 hailo/g4_more_wsl.sh를 분리 실행
# (지연 측정 중에는 WSL을 켜지 않는다: 쉬는 VM도 측정 잡음이 될 수 있음)
ART="$SIGLIP2_ARTIFACTS"
until grep -q "dfc calib256 rc=0" $ART/logs/after_speed_win.log 2>/dev/null; do sleep 60; done
wsl -d Ubuntu-22.04 -e bash -lc "tr -d '\r' < /mnt/d/mokerton/oslab_mobile/siglip2/hailo/g4_more_wsl.sh > ~/g4_more_wsl.sh && cd ~ && setsid nohup bash ~/g4_more_wsl.sh > /mnt/d/mokerton/artifacts/siglip2/logs/g4_more_wsl.log 2>&1 < /dev/null & disown"
echo "$(date +%T) launched g4_more_wsl.sh"
