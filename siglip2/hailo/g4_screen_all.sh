#!/usr/bin/env bash
# g4_screen_all.sh — G4 구성 A~E를 ORT 스크리닝과 같은 표본(dev 6종 × 앞 200장)으로 DFC 양자화 에뮬레이션 (WSL)
# 지연 측정(Windows, run_when_idle.ps1)이 끝날 때까지 기다린다. 단 START_BY(기본 03:30)가 지나면 먼저 시작한다
# (그 경우 지연 측정은 이 작업이 끝나 PC가 한가해진 뒤로 밀린다).
# 사용: bash g4_screen_all.sh [START_BY HH:MM]
set -uo pipefail
START_BY=${1:-03:30}
SPEED_LOG=/mnt/d/mokerton/artifacts/siglip2/logs/speed_when_idle.log
OUT=/mnt/d/mokerton/oslab_mobile/results/siglip2/G4_dfc_screen
W1=~/siglip2_dfc/g4_20261013_165132   # A·B·C
W2=~/siglip2_dfc/g4_20261013_221931   # D·E
until grep -q "finished exit" $SPEED_LOG 2>/dev/null || [ "$(date +%H:%M)" \> "$START_BY" -a "$(date +%H)" -lt 12 ]; do sleep 60; done
echo "$(date +%T) start (speed finished: $(grep -c 'finished exit' $SPEED_LOG 2>/dev/null))"
source ~/hailo10_venv/bin/activate
mkdir -p $OUT
tr -d '\r' < /mnt/d/mokerton/oslab_mobile/siglip2/hailo/s2_dfc_screen.py > ~/s2_dfc_screen.py
for item in "$W2/opt_D_official_b9fc2.har D_official_b9fc2" "$W2/opt_E_fc2_b0b5b9.har E_fc2_b0b5b9" \
            "$W1/opt_C_official.har C_official" "$W1/opt_B_latefc2.har B_latefc2" "$W1/opt_A_ewadd_patch16.har A_ewadd_patch16"; do
  set -- $item
  python ~/s2_dfc_screen.py --har $1 --tag $2 --k 200 --bs 8 --out $OUT > $OUT/log_$2.txt 2>&1
  echo "$2 rc=$? $(grep '"hr_cos"' $OUT/log_$2.txt | tail -1)"
done
echo G4SCREEN DONE
