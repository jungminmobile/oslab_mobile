@echo off
rem 준혁 요청(10/18): release 파일 fp32·final 지연, RN50과 같은 조건(충전, 조용할 때), 스레드 1·2·4
chcp 65001 >nul
cd /d "%~dp0\.."
set PY=rn50\.venv\Scripts\python.exe
set D=siglip2_onnx
set RN50_RESULTS=results\final_compare\siglip2_release_1018
set LOG=results\final_compare\siglip2_release_1018_log.txt
echo %date% %time% start > %LOG%
powercfg /getactivescheme >> %LOG%
for %%t in (1 2 4) do %PY% rn50/bench_speed.py --warmup 20 --n 200 --repeats 3 --threads %%t --models fp32=%D%/siglip2_image_fp32.onnx final=%D%/siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx >> %LOG% 2>&1
echo %date% %time% ALL DONE >> %LOG%
