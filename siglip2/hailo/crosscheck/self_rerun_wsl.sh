#!/usr/bin/env bash
# self_rerun_wsl.sh — (10/18 새벽) 쭈리 PC에서 D256L1 optimize를 같은 입력으로 다시 돌려, 10/15 결과와 비트 단위로 같은지 본다
#  같으면 → 정민 노트북과의 차이(사진별 cos 평균 0.005)는 CPU에 따른 계산 차이 (가설 1). 다르면 → DFC 안의 무작위성 (가설 2). results/siglip2/G4_crosscheck 참고
set -uo pipefail
source ~/hailo10_venv/bin/activate
B=/mnt/d/mokerton/artifacts/siglip2/share/20261017_hailo_crosscheck
R=/mnt/d/mokerton/oslab_mobile/results/siglip2/G4_crosscheck/self_rerun
W=~/siglip2_dfc/rerun_20261018
mkdir -p $W $R && cd $W
cp $B/siglip2_image_fp32_b1.onnx $B/calib_highres256_s0_nhwc.npy $B/g4_D256L1_official_b9fc2.alls $B/pets_dev.npz $B/ref_D256L1_pets200.npz $B/crosscheck_emulate.py $B/SHA256SUMS.txt .
sha256sum -c SHA256SUMS.txt | tee $R/steps.txt
{ hailo --version 2>&1 | tail -1; python --version; uname -r; nproc; free -g | head -2; } > $R/env.txt 2>&1
t0=$(date +%s); hailo parser onnx siglip2_image_fp32_b1.onnx --hw-arch hailo10h --net-name siglip2_b16_224 --har-path siglip2_b16_224.har -y > parse.log 2>&1
echo "$(date +%T) parse rc=$? sec=$(( $(date +%s) - t0 ))" | tee -a $R/steps.txt
t0=$(date +%s); hailo optimize siglip2_b16_224.har --hw-arch hailo10h --model-script g4_D256L1_official_b9fc2.alls \
  --calib-set-path calib_highres256_s0_nhwc.npy --output-har-path opt_D256L1_rerun.har > optimize.log 2>&1
echo "$(date +%T) optimize rc=$? sec=$(( $(date +%s) - t0 ))" | tee -a $R/steps.txt
python crosscheck_emulate.py --har opt_D256L1_rerun.har --data pets_dev.npz --ref ref_D256L1_pets200.npz > emulate.log 2>&1
echo "$(date +%T) emulate rc=$? $(tr -d '\n' < crosscheck_result.json 2>/dev/null)" | tee -a $R/steps.txt
cp crosscheck_result.json emulate.log $R/ 2>/dev/null; tail -c 60000 optimize.log > $R/optimize_tail.log
echo "SELF_RERUN DONE $(date +%T)" | tee -a $R/steps.txt
