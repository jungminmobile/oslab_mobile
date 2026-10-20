#!/usr/bin/env bash
# night_1016.sh — 10/16 새벽 순서 실행 (쭈리 "PC 작업 다시 시작", 개인연구 세션과 조율)
#  ① 4비트 sweep (0단계 기본 경로 sha256 확인 포함, Windows, 잠금 하나, WSL 내려간 상태)
#  ② C256L1 dev 전체 남은 조각 (WSL, hailo/g4_more7b_wsl.sh, 끝난 조각 건너뜀)
#  ③ WSL 종료 → portable_eval x86 검증 → ④ 지연 2세션(speed_1016.sh와 같은 내용) → "SPEED_1016 DONE"
#  멈추기: D:\mokerton\locks\siglip2.STOP 파일을 만들면 다음 단계로 넘어가기 전에 멈춘다 (10/15 밤: 백그라운드 작업을 끝내도
#  Windows에서 자식 bash가 살아남아 WSL을 다시 띄운 일이 있어, 프로세스 종료 대신 파일로 멈춘다).
set -uo pipefail
cd /d/mokerton/oslab_mobile/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
PY="$SIGLIP2_PYTHON"; ART="$SIGLIP2_ARTIFACTS"; L=$ART/logs
STOP=/d/mokerton/locks/siglip2.STOP
chk() { [ -f $STOP ] && { echo "$(date +%T) STOP 파일 발견 — 여기서 멈춤 ($1 전)"; exit 3; }; return 0; }
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) status=[$(git status --porcelain --untracked-files=no -- . | tr '\n' ' ')]"
chk "w4"
W4_ONE_LOCK=1 "$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "SigLIP2 w4 sweep (quantize x5 + screen, ~50 min, 3 GB)" -- \
  "C:/Program Files/Git/bin/bash.exe" scripts/w4_sweep.sh > $L/w4_sweep.log 2>&1
echo "$(date +%T) w4 sweep rc=$? $(tail -1 $L/w4_sweep.log)"
chk "C256L1"
wsl.exe -d Ubuntu-22.04 -e bash -lc "tr -d '\r' < /mnt/d/mokerton/oslab_mobile/siglip2/hailo/g4_more7b_wsl.sh > ~/g4_more7b_wsl.sh && cd ~ && bash ~/g4_more7b_wsl.sh >> /mnt/d/mokerton/artifacts/siglip2/logs/g4_more7_wsl.log 2>&1"
echo "$(date +%T) C256L1 rc=$? $(tail -1 $L/g4_more7_wsl.log)"
wsl.exe -t Ubuntu-22.04; echo "$(date +%T) WSL terminated rc=$?"
chk "portable_eval"
B="$ART/share/20261015_xplat"; O=../results/siglip2/G3_xplat/x86_i7-1165G7; mkdir -p $O
echo "{\"commit\": \"$(git rev-parse --short HEAD)\", \"bundle\": \"share/20261015_xplat\"}" > $O/run_info.json
"$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "SigLIP2 portable_eval x86 check" -- "$PY" -u "$B/portable_eval.py" --data "$B/dev200" --out $O --threads 4 \
  --models fp32="$B/models/siglip2_image_fp32.onnx" s8="$B/models/siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16_fc2L2.onnx" \
  u8rr="$B/models/siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx" > $L/portable_eval_x86.log 2>&1
echo "$(date +%T) portable_eval rc=$?"
chk "speed"
# 지연: speed_1016.sh의 측정 부분과 같다 (대기 단계 없이 바로). 잠금 안에서 CPU 30초 25% 미만일 때만
R=../results/siglip2/G3_cumulative
P=siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16
U=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16
timed() {
  local job=$1; shift
  while true; do
    chk "$job"
    "$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job "$job" -- \
      "C:/Program Files/Git/bin/bash.exe" -c "c=\$('$PY' -c 'import psutil; print(round(psutil.cpu_percent(interval=30)))'); echo \"in-lock cpu30s=\$c%\"; \
        if [ \"\$c\" -ge 25 ]; then echo BUSY_RETRY; exit 75; fi; $*"
    local rc=$?; echo "$(date +%T) $job rc=$rc"
    [ $rc -eq 75 ] && { sleep 180; continue; }
    return $rc
  done
}
M="fp32=siglip2_image_fp32.onnx fc2L2=${P}_fc2L2.onnx fc2L2_u8=${U}_fc2L2.onnx fc2L2_u8rr=${U}_rr_fc2L2.onnx"
for s in a b; do for t in 1 2 4; do
  timed "SigLIP2 latency 1016$s threads=$t (needs idle PC)" "RN50_RESULTS=$R/speed_1016$s '$PY' -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t --models $M"
done; done
echo "SPEED_1016 DONE $(date +%T)" | tee -a $L/speed_1016.log
