#!/bin/bash
# E1 6단계 준비 (노트북 WSL에서 실행)
#  (a) 새 구성 2개(c3eq, off8L2L3)를 레벨 0으로 양자화 + 990장 평가 → 레벨 2와 비교할 기준값
#  (b) Kaggle 업로드 폴더 만들기: <대회폴더>/kaggle_upload
set -u
S="$(cd "$(dirname "$0")" && pwd)"; W=~/hailo10_work; P=$W/logs/step6_progress.txt
U="$(cd "$S/../../.." && pwd)/kaggle_upload"
source ~/hailo10_venv/bin/activate
mkdir -p $W/logs "$U/alls"; : > $P
echo "$(date +%T) (b) 업로드 폴더 준비: $U" >> $P
WHL=$(ls ~/hailo_dataflow_compiler-5.4.0*.whl ~/*/hailo_dataflow_compiler-5.4.0*.whl /mnt/c/Users/*/Downloads/hailo_dataflow_compiler-5.4.0*.whl 2>/dev/null | head -1)
echo "whl=$WHL" >> $P
[ -n "$WHL" ] && cp -u "$WHL" "$U/"
cp -u $W/parse/rn50.har "$U/rn50_parsed.har"
cp -u $W/calib/calib_rn50_highres64_s0_nhwc_u8.npy "$U/"
for k in x y clf ref_fp32; do cp -u "$S/../../rn50/cache/pets_dev_$k.npy" "$U/"; done
cp "$S/emu_eval.py" "$U/"
pip freeze > "$U/laptop_hailo10_venv_freeze.txt"; python3 --version >> "$U/laptop_hailo10_venv_freeze.txt.pyver" 2>&1
for c in official:rn50_official base8:e1_base off8L4:rn50_off8L4 off8L2L3:rn50_off8L2L3 c3eq:rn50_c3eq; do
  n=${c%%:*}; f=${c##*:}
  cp "$S/$f.alls" "$U/alls/${n}_lv0.alls"
  sed 's/optimization_level=0/optimization_level=2/' "$S/$f.alls" > "$U/alls/${n}_lv2.alls"
done
(cd "$U" && sha256sum *.whl *.har *.npy > SHA256SUMS.txt)   # 경로 없이 파일 이름만 기록 (Kaggle에서 sha256sum -c 가능)
echo "$(date +%T) (b) 끝" >> $P
for n in c3eq off8L2L3; do
  echo "$(date +%T) (a) $n 레벨0 양자화 시작" >> $P
  mkdir -p $W/opt_$n; cd $W/opt_$n
  hailo optimize $W/parse/rn50.har --hw-arch hailo10h --calib-set-path $W/calib/calib_rn50_highres64_s0_nhwc_u8.npy \
     --model-script "$U/alls/${n}_lv0.alls" > $W/logs/rn50_${n}_optimize.log 2>&1
  echo "$(date +%T) (a) $n 양자화 exit=$? → 평가" >> $P
  python3 "$S/emu_eval.py" --har $(ls $W/opt_$n/*_optimized.har | head -1) --ctx quantized --data pets_dev --n 990 --tag lv0_990_$n > $W/logs/step6_$n.log 2>&1
  echo "$(date +%T) (a) $n 끝: $(tail -n1 $W/results/emu_eval.jsonl | cut -c1-260)" >> $P
done
echo "$(date +%T) ALL DONE" >> $P
