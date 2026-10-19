@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "RN50_RESULTS=%~dp0..\results\E10_ilp"
set OPENBLAS_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
set "LOG=%~dp0..\results\E10_ilp\run_log.txt"
echo %date% %time% stage3 start >> "%LOG%"
.venv\Scripts\python.exe -u e10_eval.py all >> "%LOG%" 2>&1
echo %date% %time% STAGE3 DONE rc=%errorlevel% >> "%LOG%"
