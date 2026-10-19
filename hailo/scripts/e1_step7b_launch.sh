#!/bin/bash
RUN_LIST="t1b_off8L4add15only16 t4_off8L1 t4b_off8L1add1only16" nohup bash "$(pwd)/e1_step7_add16.sh" > /dev/null 2>&1 &
sleep 15; tail -n3 ~/hailo10_work/logs/step7_progress.txt
