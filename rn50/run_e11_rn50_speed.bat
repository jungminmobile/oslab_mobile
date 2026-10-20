@echo off
rem E11 RN50 속도(1·2·4T × 3세션, 세션 사이 2분)·메모리. 다른 작업 없이 실행
chcp 65001 >nul
cd /d "%~dp0"
set "RN50_RESULTS=%~dp0..\results\E11_cross\rn50"
set OPENBLAS_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
set "LOG=%~dp0..\results\E11_cross\rn50\run_log.txt"
set PY=.venv\Scripts\python.exe
set P=rn50_image_int8_q1
set S=percentile99999_highres256_s0
set M=fp32=rn50_image_fp32_q1.onnx B2=%P%_%S%_B2.onnx B2rr=%P%_%S%_B2_rr.onnx Ddef=%P%_dynamic_Ddef.onnx Dpc=%P%_dynamic_Dpc_rr.onnx B2w4=%P%_%S%_B2w4.onnx rotvc=%P%rotvc_%S%_B2.onnx rotqkvc=%P%rotqkvc_%S%_B2.onnx
echo %date% %time% speed start >> "%LOG%"
powercfg /getactivescheme >> "%LOG%"
for %%s in (1 2 3) do (
  for %%t in (1 2 4) do (
    %PY% -u bench_speed.py --threads %%t --n 200 --repeats 3 --warmup 20 --models %M% >> "%LOG%" 2>&1
    echo %date% %time% speed s%%s t%%t done >> "%LOG%"
  )
  if not %%s==3 timeout /t 120 /nobreak >nul
)
echo %date% %time% mem >> "%LOG%"
%PY% -u bench_mem.py --threads 4 --models %M% >> "%LOG%" 2>&1
echo %date% %time% SPEED MEM DONE >> "%LOG%"
