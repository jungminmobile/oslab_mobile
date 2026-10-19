#!/bin/bash
nohup bash "$(pwd)/e1_step6_prep.sh" > /dev/null 2>&1 &
sleep 30; cat ~/hailo10_work/logs/step6_progress.txt
