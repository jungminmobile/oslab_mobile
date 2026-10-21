#!/usr/bin/env bash
# g6e_chain.sh — 2단계 compile을 순서대로 (E3 → E1 → E2). 모두 B16_both의 optimized HAR(7d22df79…)에서 compile만 (상한 3시간).
# 각 compile 뒤 compiled HAR로 g6e_diag.py를 돌려 ew_add38의 컨텍스트·경계 홉을 기록한다 (diag_<tag>.json).
# 사용: nohup bash g6e_chain.sh > ~/siglip2_work/g6/g6e_chain.log 2>&1 < /dev/null &
set -uo pipefail
H=$(cd "$(dirname "$0")" && pwd); W=$HOME/siglip2_work/g6
HAR=$W/dfc/b16_both/siglip2_m2rot_optimized.har
for e in ${ORDER:-E3 E1 E2}; do
  t=g6e_both_$e
  bash $H/g6e_compile_only.sh $t $HAR $H/alls/$t.alls 10800
  c=$(ls $W/dfc/$t/comp/*compiled.har 2>/dev/null | head -1)
  if [ -n "$c" ]; then
    ~/venvs/siglip2/bin/python $H/g6e_diag.py --har $t=$c --focus $t=ew_add38 --hops 1 --out $H/diag_$t.json 2>&1 | grep -E "^==|ew_add38 " 
  fi
done
echo "G6E CHAIN DONE $(date +%T)"
