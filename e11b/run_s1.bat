@echo off
rem E11b S1: SigLIP2 R0 보정 데이터 출처·장수 4개 → dev 전체 정확도
chcp 65001 >nul
cd /d "%~dp0..\..\oslab_s2\siglip2"
set "PY=%~dp0..\..\s2_venv\Scripts\python.exe"
set "SIGLIP2_DATA=%~dp0..\..\s2data"
set "SIGLIP2_ARTIFACTS=C:\s2art"
set "RN50_RESULTS=C:\s2art\E11b"
set OPENBLAS_NUM_THREADS=1
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
set "LOG=C:\s2art\E11b\run_log.txt"
if not exist C:\s2art\E11b mkdir C:\s2art\E11b
set Q=--op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/" --act-type u8 --reduce-range --tag fc2L2 --dump-ops
echo %date% %time% S1 start >> "%LOG%"
"%PY%" -u s2_quantize.py --calib-source cifar --calib-n 256 %Q% >> "%LOG%" 2>&1
"%PY%" -u s2_quantize.py --calib-source mix --calib-n 256 %Q% >> "%LOG%" 2>&1
"%PY%" -u s2_quantize.py --calib-source highres --calib-n 64 %Q% >> "%LOG%" 2>&1
"%PY%" -u s2_quantize.py --calib-source highres --calib-n 311 %Q% >> "%LOG%" 2>&1
echo %date% %time% S1 eval >> "%LOG%"
set P=siglip2_image_int8_percentile99999
set T=s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
"%PY%" -u s2_paired_ci.py --threads 6 --models %P%_cifar256_%T% %P%_mix256_%T% %P%_highres64_%T% %P%_highres311_%T% >> "%LOG%" 2>&1
echo %date% %time% S1 DONE >> "%LOG%"
