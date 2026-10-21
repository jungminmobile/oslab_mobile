@echo off
rem E11b S3 1단계: Linear 단위 51개 비용표 (s3_cost.py)
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
set R0=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2
echo %date% %time% S3 cost start >> "%LOG%"
"%PY%" -u "%~dp0s3_cost.py" --group 6 >> "%LOG%" 2>&1
echo %date% %time% S3 COST DONE >> "%LOG%"
