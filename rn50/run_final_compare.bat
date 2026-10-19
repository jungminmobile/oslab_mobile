@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set R=..\results\final_compare
set RN=fp32=rn50_image_fp32_q1.onnx B0=rn50_image_int8_q1_minmax_cifar64_s0_spdB0.onnx C1=rn50_image_int8_q1_minmax_cifar64_s0_spdC1.onnx H1=rn50_image_int8_q1_minmax_cifar64_s0_spdH1.onnx B2=rn50_image_int8_q1_minmax_cifar64_s0_spdB2.onnx H2=rn50_image_int8_q1_minmax_cifar64_s0_spdH2.onnx B4=rn50_image_int8_q1_minmax_cifar64_s0_spdB4.onnx B6=rn50_image_int8_q1_minmax_cifar64_s0_spdB6.onnx
set SG=fp32=siglip2_image_fp32.onnx S8=siglip2_image_int8_percentile99999_highres256_s0_s8_ops-MatMul_hc16_fc2L2.onnx U8rr=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
echo %date% %time% start > %R%_log.txt
set RN50_RESULTS=%R%\rn50
%PY% bench_speed.py --models %RN% --threads 1 2 4 >> %R%_log.txt 2>&1
echo %date% %time% rn50 speed done >> %R%_log.txt
set RN50_RESULTS=%R%\siglip2
%PY% bench_speed.py --models %SG% --threads 1 2 4 >> %R%_log.txt 2>&1
echo %date% %time% siglip2 speed done >> %R%_log.txt
set RN50_RESULTS=%R%\rn50
%PY% bench_mem.py --models %RN% --threads 4 >> %R%_log.txt 2>&1
set RN50_RESULTS=%R%\siglip2
%PY% bench_mem.py --models %SG% --threads 4 >> %R%_log.txt 2>&1
echo %date% %time% ALL DONE >> %R%_log.txt
