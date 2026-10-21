@echo off
chcp 65001 >nul
rem E12-1 ORT session memory options - run one stage at a time (plan 12-3)
rem usage: e12\run_e12_1.bat smoke / g1 / mem_m2 / mem_fp / speed1 / speed2 / speed3
cd /d "%~dp0.."
set "PY=%~dp0..\..\s2_venv\Scripts\python.exe"
set "OUT=results\E12-1_ORT세션메모리설정\results"
set "ONNX=C:\s2art\onnx"
set "M2=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx"
set "FP=siglip2_image_fp32.onnx"
set "SIGLIP2_DATA=%~dp0..\..\s2data"
set "SIGLIP2_ARTIFACTS=C:\s2art"
set OPENBLAS_NUM_THREADS=1
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
set "LOG=%OUT%\run_log.txt"
if not exist "%OUT%" mkdir "%OUT%"
echo %date% %time% START %1 >> "%LOG%"
goto %1

:smoke
"%PY%" -u e12\check_equal.py --model "%ONNX%\%M2%" --combos C0 C4 --n 2 --threads 4 --out "%OUT%\smoke_equal.json" >> "%LOG%" 2>&1 || goto fail
"%PY%" -u e12\bench_mem_session.py --models "m2rot=%ONNX%\%M2%" --combos C0 C4 --threads 4 --n 2 --repeats 1 --interval-ms 10 --out "%OUT%\smoke_memory.jsonl" >> "%LOG%" 2>&1 || goto fail
"%PY%" -u e12\bench_speed_session.py --session 0 --models "m2rot=%ONNX%\%M2%" --combos C0 C4 --threads 4 --warmup 2 --n 2 --out "%OUT%\smoke_speed.jsonl" >> "%LOG%" 2>&1 || goto fail
goto done
:g1
"%PY%" -u e12\check_equal.py --model "%ONNX%\%M2%" --combos C0 C1 C2 C3 C4 --n 200 --threads 4 --out "%OUT%\equal.json" >> "%LOG%" 2>&1 || goto fail
goto done
:mem_m2
"%PY%" -u e12\bench_mem_session.py --models "m2rot=%ONNX%\%M2%" --combos C0 C1 C2 C3 C4 --threads 4 --n 50 --repeats 3 --interval-ms 10 --out "%OUT%\memory.jsonl" >> "%LOG%" 2>&1 || goto fail
goto done
:mem_fp
"%PY%" -u e12\bench_mem_session.py --models "fp32=%ONNX%\%FP%" --combos C0 C1 C2 C3 C4 --threads 4 --n 50 --repeats 3 --interval-ms 10 --out "%OUT%\memory.jsonl" >> "%LOG%" 2>&1 || goto fail
goto done
:speed1
"%PY%" -u e12\bench_speed_session.py --session 1 --models "m2rot=%ONNX%\%M2%" "fp32=%ONNX%\%FP%" --combos C0 C1 C2 C3 C4 --threads 1 2 4 --warmup 20 --n 200 --out "%OUT%\speed.jsonl" >> "%LOG%" 2>&1 || goto fail
goto done
:speed2
"%PY%" -u e12\bench_speed_session.py --session 2 --models "m2rot=%ONNX%\%M2%" "fp32=%ONNX%\%FP%" --combos C2 C1 C4 C3 C0 --threads 1 2 4 --warmup 20 --n 200 --out "%OUT%\speed.jsonl" >> "%LOG%" 2>&1 || goto fail
goto done
:speed3
"%PY%" -u e12\bench_speed_session.py --session 3 --models "m2rot=%ONNX%\%M2%" "fp32=%ONNX%\%FP%" --combos C4 C3 C2 C1 C0 --threads 1 2 4 --warmup 20 --n 200 --out "%OUT%\speed.jsonl" >> "%LOG%" 2>&1 || goto fail
goto done

:fail
echo %date% %time% FAIL %1 >> "%LOG%"
exit /b 1
:done
echo %date% %time% DONE %1 >> "%LOG%"
