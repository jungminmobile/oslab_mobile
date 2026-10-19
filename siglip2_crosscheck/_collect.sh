#!/bin/bash
D="/mnt/c/study/3/2/2026 모바일 학술대회/oslab_mobile/siglip2_crosscheck/result_jungmin"
mkdir -p "$D"
cd ~/siglip2_crosscheck
cp crosscheck_steps.txt crosscheck_env.txt crosscheck_result.json *.log "$D"/
cd "$D"/..
rm -f result_jungmin.zip
python3 -c "import shutil; shutil.make_archive('result_jungmin', 'zip', 'result_jungmin')"
ls -la result_jungmin result_jungmin.zip
