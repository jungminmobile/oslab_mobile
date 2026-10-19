#!/bin/bash
nohup bash "$(pwd)/e1_step5_emu990.sh" > /dev/null 2>&1 &
sleep 45; cat ~/hailo10_work/logs/step5_progress.txt; tail -n3 ~/hailo10_work/logs/step5_fp.log
