#!/bin/bash
nohup bash "$(pwd)/e1_step7_add16.sh" > /dev/null 2>&1 &
sleep 20; cat ~/hailo10_work/logs/step7_progress.txt
