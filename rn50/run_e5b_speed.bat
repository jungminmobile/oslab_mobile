@echo off
rem E5b 2단계 속도·메모리. 사전 등록 results\E5b_bias_corr\PREREG.md. 정확도(run_e5b.bat) 끝난 뒤, 다른 작업 없이 실행
chcp 65001 >nul
cd /d "%~dp0"
set "RN50_RESULTS=%~dp0..\results\E5b_bias_corr"
set OPENBLAS_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
set "LOG=%~dp0..\results\E5b_bias_corr\run_log.txt"
echo %date% %time% speed start >> "%LOG%"
powercfg /getactivescheme >> "%LOG%"
.venv\Scripts\python.exe -u e5b_run.py speed >> "%LOG%" 2>&1
echo %date% %time% mem >> "%LOG%"
.venv\Scripts\python.exe -u e5b_run.py mem >> "%LOG%" 2>&1
echo %date% %time% SPEED MEM DONE rc=%errorlevel% >> "%LOG%"
