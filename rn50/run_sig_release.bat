@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
set PY=rn50\.venv\Scripts\python.exe
set RN50_RESULTS=results\final_compare\siglip2_release
set D=siglip2_onnx
echo %date% %time% start > results\final_compare\siglip2_release_log.txt
%PY% rn50/bench_speed.py --warmup 20 --n 200 --repeats 3 --threads 1 2 4 --models fp32=%D%/siglip2_image_fp32.onnx final=%D%/siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx s8=%D%/siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16_fc2L2.onnx >> results\final_compare\siglip2_release_log.txt 2>&1
echo %date% %time% ALL DONE >> results\final_compare\siglip2_release_log.txt
