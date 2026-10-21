@echo off
chcp 65001 >nul
rem E12-2 pre-quantization fusion - run one stage at a time (plan 10-3)
rem usage: e12\run_e12_2.bat env / diag / fuse / equal / quant / smoke / acc / mem / speed1 / speed2 / speed3
cd /d "%~dp0.."
set "PY=%~dp0..\..\s2_venv\Scripts\python.exe"
set "G8=%~dp0..\..\oslab_g8\siglip2"
set "OUT=%~dp0..\results\E12-2_양자화전연산합치기\results"
set "ONNX=C:\s2art\onnx"
set "M2FP=siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx"
set "M2FUSE=siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad_E12-2fused.onnx"
set "O=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx"
set "B0=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_E12-2_B0.onnx"
set "B1=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_E12-2fused_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_E12-2_B1.onnx"
set "SIGLIP2_DATA=%~dp0..\..\s2data"
set "SIGLIP2_ARTIFACTS=C:\s2art"
set OPENBLAS_NUM_THREADS=1
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
set "QOPT=--calib-source highres --calib-n 256 --calib-bs 8 --method percentile --percentile 99.999 --act-type u8 --reduce-range --weight-type s8 --op-types MatMul --hist-chunk 16 --per-channel 1 --seed 0 --dump-ops --threads 4"
set "LOG=%OUT%\run_log.txt"
if not exist "%OUT%" mkdir "%OUT%"
echo %date% %time% START %1 >> "%LOG%"
goto %1

:env
"%PY%" -u e12\e12_2_env.py --out "%OUT%" >> "%LOG%" 2>&1 || goto fail
goto done
:diag
"%PY%" -u e12\diag_ln.py --model "%ONNX%\%O%" --also-raw "%ONNX%\%M2FP%" --out "%OUT%\diag_ln.json" >> "%LOG%" 2>&1 || goto fail
goto done
:fuse
"%PY%" -u e12\fuse_fp32.py --src "%ONNX%\%M2FP%" --dst "%ONNX%\%M2FUSE%" --log "%OUT%\fuse_log.json" >> "%LOG%" 2>&1 || goto fail
goto done
:equal
"%PY%" -u e12\check_fp32_equal.py --a "%ONNX%\%M2FP%" --b "%ONNX%\%M2FUSE%" --n 200 --threads 4 --out "%OUT%\fp32_equal.json" >> "%LOG%" 2>&1 || goto fail
goto done
:quant
pushd "%G8%"
"%PY%" -u s2_quantize.py --src "%M2FP%" %QOPT% --log "%OUT%\quantize_log.jsonl" --tag E12-2_B0 >> "%LOG%" 2>&1 || (popd & goto fail)
"%PY%" -u s2_quantize.py --src "%M2FUSE%" %QOPT% --log "%OUT%\quantize_log.jsonl" --tag E12-2_B1 >> "%LOG%" 2>&1 || (popd & goto fail)
popd
goto done
:smoke
"%PY%" -u e12\bench_mem_session.py --models "B1=%ONNX%\%B1%" --combos C0 --threads 4 --n 2 --repeats 1 --interval-ms 10 --out "%OUT%\smoke_memory.jsonl" >> "%LOG%" 2>&1 || goto fail
"%PY%" -u e12\bench_speed_session.py --session 0 --models "B1=%ONNX%\%B1%" --combos C0 --threads 4 --warmup 2 --n 2 --out "%OUT%\smoke_speed.jsonl" >> "%LOG%" 2>&1 || goto fail
goto done
:acc
rem s2common requires RN50_RESULTS under oslab_g8\results\siglip2 or SIGLIP2_ARTIFACTS -> write to C:\s2art\results_E12-2 then copy to OUT
set "RN50_RESULTS=C:\s2art\results_E12-2"
pushd "%G8%"
"%PY%" -u s2_paired_ci.py --models "%B0%" "%B1%" "%O%" --threads 4 --n-boot 1000 >> "%LOG%" 2>&1 || (popd & goto fail)
popd
copy /y "C:\s2art\results_E12-2\paired_ci.jsonl" "%OUT%\" >nul
copy /y "C:\s2art\results_E12-2\pred_*.npz" "%OUT%\" >nul
goto done
:mem
"%PY%" -u e12\bench_mem_session.py --models "B0=%ONNX%\%B0%" "B1=%ONNX%\%B1%" --combos C0 --threads 4 --n 50 --repeats 3 --interval-ms 10 --out "%OUT%\memory.jsonl" >> "%LOG%" 2>&1 || goto fail
goto done
:speed1
"%PY%" -u e12\bench_speed_session.py --session 1 --models "O=%ONNX%\%O%" "B0=%ONNX%\%B0%" "B1=%ONNX%\%B1%" --combos C0 --threads 1 2 4 --warmup 20 --n 200 --out "%OUT%\speed.jsonl" >> "%LOG%" 2>&1 || goto fail
goto done
:speed2
"%PY%" -u e12\bench_speed_session.py --session 2 --models "B0=%ONNX%\%B0%" "B1=%ONNX%\%B1%" "O=%ONNX%\%O%" --combos C0 --threads 1 2 4 --warmup 20 --n 200 --out "%OUT%\speed.jsonl" >> "%LOG%" 2>&1 || goto fail
goto done
:speed3
"%PY%" -u e12\bench_speed_session.py --session 3 --models "B1=%ONNX%\%B1%" "O=%ONNX%\%O%" "B0=%ONNX%\%B0%" --combos C0 --threads 1 2 4 --warmup 20 --n 200 --out "%OUT%\speed.jsonl" >> "%LOG%" 2>&1 || goto fail
goto done

:fail
echo %date% %time% FAIL %1 >> "%LOG%"
exit /b 1
:done
echo %date% %time% DONE %1 >> "%LOG%"
