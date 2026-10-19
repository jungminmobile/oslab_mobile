#!/bin/bash
date +%T; cut -c1-230 ~/hailo10_work/logs/step6_progress.txt
echo "--- 레벨0 로그에서 실행된 알고리즘/경고"
grep -ohE "Starting [A-Z][A-Za-z ]+|Reducing optimization level[^.]*" ~/hailo10_work/logs/rn50_c3eq_optimize.log | sort | uniq -c
