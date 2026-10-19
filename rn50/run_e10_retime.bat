@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "RN50_RESULTS=%~dp0..\results\E10_ilp"
set OPENBLAS_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -u e10_cost.py retime >> "%~dp0..\results\E10_ilp\run_log.txt" 2>&1
echo %date% %time% RETIME DONE rc=%errorlevel% >> "%~dp0..\results\E10_ilp\run_log.txt"
