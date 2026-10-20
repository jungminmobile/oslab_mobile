@echo off
rem E5b 0~2단계(정확도). 사전 등록 results\E5b_bias_corr\PREREG.md (b5c4b7f)
chcp 65001 >nul
cd /d "%~dp0"
set "RN50_RESULTS=%~dp0..\results\E5b_bias_corr"
set OPENBLAS_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
set "LOG=%~dp0..\results\E5b_bias_corr\run_log.txt"
echo %date% %time% start >> "%LOG%"
powercfg /getactivescheme >> "%LOG%"
.venv\Scripts\python.exe -u e5b_run.py check >> "%LOG%" 2>&1
if errorlevel 1 (
  echo %date% %time% STOP: stage0 gate failed >> "%LOG%"
  exit /b 1
)
echo %date% %time% build >> "%LOG%"
.venv\Scripts\python.exe -u e5b_run.py build >> "%LOG%" 2>&1
echo %date% %time% acc >> "%LOG%"
.venv\Scripts\python.exe -u e5b_run.py acc >> "%LOG%" 2>&1
echo %date% %time% ACC DONE rc=%errorlevel% >> "%LOG%"
