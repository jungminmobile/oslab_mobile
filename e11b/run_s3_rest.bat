@echo off
rem E11b S3 2~3단계: 계획 → 구성 생성·dev 정확도 → R0 대비 짝 비교 → 속도·메모리
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
echo %date% %time% S3 plan >> "%LOG%"
"%PY%" -u "%~dp0s3_plan.py" C:\s2art\E11b\s3_cost.jsonl C:\s2art\E11b\s3_plan.json >> "%LOG%" 2>&1
if errorlevel 1 goto :eof
echo %date% %time% S3 build eval >> "%LOG%"
"%PY%" -u "%~dp0s3_build_eval.py" C:\s2art\E11b\s3_plan.json >> "%LOG%" 2>&1
for %%f in (C:\s2art\E11b\pred_*S3P*.npz) do "%PY%" -u "%~dp0pairs.py" --base C:\s2art\E5a\pred_%R0%.npz --pred "%%f" --out C:\s2art\E11b\pairs.jsonl >> "%LOG%" 2>&1
echo %date% %time% S3 speed >> "%LOG%"
"%PY%" -u "%~dp0s3_speed.py" C:\s2art\E11b\s3_plan.json >> "%LOG%" 2>&1
echo %date% %time% S3 ALL DONE >> "%LOG%"
