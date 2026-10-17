#!/usr/bin/env bash
# run_crosscheck.sh — 정민 노트북 WSL(DFC 5.4.0)에서 실행. SigLIP2 Hailo 최종(D256L1, D-032)을 같은 입력으로 다시 만들어 쭈리 PC 결과와 비교 (D-011·D-027)
# 준비: GitHub pre-release siglip2-hailo-crosscheck-20261017의 파일을 한 폴더에 받고, DFC 가상환경을 켠 뒤 그 폴더에서 실행:
#   bash run_crosscheck.sh            # ① parse ② optimize ③ 에뮬레이터 비교 (약 1시간 반, GPU 없어도 됨)
#   COMPILE=1 bash run_crosscheck.sh  # ④ HEF 컴파일·프로파일러까지 (추가 약 1시간 반)
set -uo pipefail
NET=siglip2_b16_224
log() { echo "$(date +%T) $*" | tee -a crosscheck_steps.txt; }
{ hailo --version 2>&1 | tail -1; python --version; uname -r; nproc; free -g | head -2; } > crosscheck_env.txt 2>&1
sha256sum -c SHA256SUMS.txt | tee -a crosscheck_steps.txt
t0=$(date +%s)
[ -f $NET.har ] || hailo parser onnx siglip2_image_fp32_b1.onnx --hw-arch hailo10h --net-name $NET --har-path $NET.har -y > parse.log 2>&1
log "parse rc=$? sec=$(( $(date +%s) - t0 ))"
t0=$(date +%s)
hailo optimize $NET.har --hw-arch hailo10h --model-script g4_D256L1_official_b9fc2.alls \
  --calib-set-path calib_highres256_s0_nhwc.npy --output-har-path opt_D256L1.har > optimize.log 2>&1
log "optimize rc=$? sec=$(( $(date +%s) - t0 ))"
sha256sum $NET.har opt_D256L1.har | tee -a crosscheck_steps.txt
python crosscheck_emulate.py --har opt_D256L1.har --data pets_dev.npz --ref ref_D256L1_pets200.npz > emulate.log 2>&1
log "emulate rc=$? $(tr -d '\n' < crosscheck_result.json 2>/dev/null)"
if [ "${COMPILE:-0}" = 1 ]; then
  t0=$(date +%s)
  hailo compiler opt_D256L1.har --hw-arch hailo10h --output-dir . --output-har-path compiled_D256L1.har > compile.log 2>&1
  log "compile rc=$? sec=$(( $(date +%s) - t0 ))"
  hailo profiler compiled_D256L1.har --hef *.hef --out-path profiler.html --csv profiler.csv > prof.log 2>&1
  log "profiler rc=$? $(grep -a 'Number of Contexts' prof.log | tail -1)"
  ls -la *.hef | tee -a crosscheck_steps.txt; sha256sum *.hef | tee -a crosscheck_steps.txt
fi
log "CROSSCHECK DONE — crosscheck_steps.txt, crosscheck_env.txt, crosscheck_result.json, *.log 를 쭈리에게 보내 주세요"
