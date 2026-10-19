@echo off
rem E10 0~1단계: B2 동일성 확인 -> 단위별 비용표. 사전 등록 results\E10_ilp\PREREG.md (0e99334)
chcp 65001 >nul
cd /d "%~dp0"
set "RN50_RESULTS=%~dp0..\results\E10_ilp"
set OPENBLAS_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
set "LOG=%~dp0..\results\E10_ilp\run_log.txt"
echo %date% %time% start > "%LOG%"
powercfg /getactivescheme >> "%LOG%"
.venv\Scripts\python.exe -u e10_cost.py check >> "%LOG%" 2>&1
if errorlevel 1 (
  echo %date% %time% STOP: B2 sha mismatch >> "%LOG%"
  exit /b 1
)
echo %date% %time% cost start >> "%LOG%"
.venv\Scripts\python.exe -u e10_cost.py cost >> "%LOG%" 2>&1
echo %date% %time% COST DONE rc=%errorlevel% >> "%LOG%"
