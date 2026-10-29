# E3: 보고서 9.2 (가) 빠른 확인 절차를 깨끗한 폴더에서 그대로 (2026-10-21 18:36~18:50)

- 목적: 학술부가 보고서 9장 명령대로 따라 했을 때 실제로 도는지, 기대값이 나오는지.
- 방법 (`siglip2/scripts/g8_repro_procedure_check.sh`): 빈 폴더 `D:\mokerton\repro_check_1021_1836`에서 저장소 코드·데이터 없이
  1. `gh release download siglip2-xplat-20261015 … -p "*.npz" -p "portable_eval.py" -p "siglip2_image_fp32.onnx" -p "SHA256SUMS.txt"` (약 1.1 GB, 약 3분)
  2. `gh release download siglip2-m2rot-ort-20261019 … -p "siglip2_image_int8_*"`
  3. `python -I release/portable_eval.py --data release --out out_quick --threads 4 --models fp32=… int8=…` (이 PC 약 8분)
- 결과 (`result.json`, `files.txt`)
  - 받은 파일 sha256: FP32 `333cad08e68dc85a`, INT8 `e84c3727f51e3b97` — 배포 기록과 같음
  - **INT8 고화질 cos 0.9756, Δtop-1 +0.3%p** (각 200장) — 보고서 9.3 기대값과 같음. 데이터셋별 값도 A3(`../A3_bundle/out_this_pc`)과 같음
- 발견해서 고친 것: 처음 초안(v4)의 명령은 `--data release/dev200`이었는데 Release 자산은 폴더 없이 들어 있어 `--data release`가 맞다 → 보고서 v5에서 고침.
- 남은 것: Windows PowerShell에서도 같은 순서로 한 번 (이번 실행은 Git Bash). `gh` CLI 없이 웹에서 받는 방법도 README에 적기.
