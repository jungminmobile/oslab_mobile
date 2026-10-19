#!/usr/bin/env bash
# G8: 다른 세션이 지연 측정 중인지 (s2_bench_speed·s2_model_stats·s2_profile_ort·hailortcli 프로세스 수). 0이면 내 양자화·정확도 평가를 시작해도 된다.
timeout 30 ps -eo pid,cmd | grep -E "s2_bench_speed|s2_model_stats|s2_profile_ort|hailortcli" | grep -v grep | wc -l
