#!/usr/bin/env bash
# yield_wsl_1015.sh — (10/15 21:40 개인연구 세션 요청) 개인연구 작업이 잠금을 잡으면 WSL Ubuntu-22.04를 내려 메모리를 돌려주고,
#  그 작업이 잠금을 놓으면 C256L1 dev 전체 평가를 이어 한다 (hailo/g4_more7b_wsl.sh).
#  WSL은 우리 체인이 잠금을 "기다리는" 동안에만 내린다 (잠금을 잡은 채 죽으면 상대가 막힘). 기다리던 대기표는 5분 뒤 자동 삭제된다.
set -uo pipefail
L=/d/mokerton/locks/lock_log.txt
LOG=/d/mokerton/artifacts/siglip2/logs/g4_more7_wsl.log
n0=$(wc -l < $L)
until tail -n +$((n0 + 1)) $L | grep -q "acquire owner=personal"; do sleep 10; done
echo "$(date +%T) personal acquired: $(tail -n +$((n0 + 1)) $L | grep 'acquire owner=personal' | head -1)"
wsl.exe -t Ubuntu-22.04; echo "$(date +%T) WSL Ubuntu-22.04 terminated rc=$?" | tee -a $LOG
n1=$(wc -l < $L)
until tail -n +$((n1 + 1)) $L | grep -q "release owner=personal"; do sleep 10; done
sleep 30
echo "$(date +%T) personal released, resuming" | tee -a $LOG
wsl.exe -d Ubuntu-22.04 -e bash -lc "tr -d '\r' < /mnt/d/mokerton/oslab_mobile/siglip2/hailo/g4_more7b_wsl.sh > ~/g4_more7b_wsl.sh && cd ~ && bash ~/g4_more7b_wsl.sh >> /mnt/d/mokerton/artifacts/siglip2/logs/g4_more7_wsl.log 2>&1"
echo "$(date +%T) g4_more7b rc=$?"
