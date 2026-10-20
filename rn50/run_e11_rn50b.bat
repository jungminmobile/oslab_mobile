@echo off
rem E11 RN50 R1~R3: 양자화 + dev 정확도. 사전 등록 results\E11_cross\PREREG.md
chcp 65001 >nul
cd /d "%~dp0"
set "RN50_RESULTS=%~dp0..\results\E11_cross\rn50"
set OPENBLAS_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
set "LOG=%~dp0..\results\E11_cross\rn50\run_log.txt"
set PY=.venv\Scripts\python.exe
set AP=/attnpool/MatMul,/attnpool/MatMul_1,/attnpool/MatMul_2,/attnpool/Gemm
set REST=/layer1/layer1.1/,/layer1/layer1.2/,/layer2/layer2.0/,/layer2/layer2.1/,/layer2/layer2.2/,/layer2/layer2.3/,/layer3/layer3.0/,/layer3/layer3.1/,/layer3/layer3.2/,/layer3/layer3.3/,/layer3/layer3.4/,/layer3/layer3.5/,/layer4/layer4.0/,/layer4/layer4.1/,/layer4/layer4.2/
set Q=--enc image --variant _q1 --calib-source highres --calib-n 256 --seed 0 --method percentile --only %REST% --only-exact %AP%
set HR=pets_dev flowers_dev cars_dev food_dev caltech_dev
set B2=rn50_image_int8_q1_percentile99999_highres256_s0_B2
echo %date% %time% retry R2 >> "%LOG%"
%PY% -u quantize_onnx.py --enc image --variant _q1 --mode dynamic --per-channel 0 --tag Ddef --overwrite >> "%LOG%" 2>&1
%PY% -u quantize_onnx.py --enc image --variant _q1 --mode dynamic --per-channel 1 --reduce-range --tag Dpc --overwrite >> "%LOG%" 2>&1
echo %date% %time% retry R3 >> "%LOG%"
set "PYTHONPATH=C:\e11_deps"
%PY% -u quantize_onnx.py %Q% --tag B2w4 --w4nbits /attnpool/MatMul,/attnpool/MatMul_1,/attnpool/MatMul_2 --overwrite >> "%LOG%" 2>&1
set "PYTHONPATH="
echo %date% %time% acc >> "%LOG%"
%PY% -u evaluate.py --exec int --threads 4 --save-pred --data cifar_A %HR% --models fp32 %B2%.onnx %B2%_rr.onnx rn50_image_int8_q1_dynamic_Ddef.onnx rn50_image_int8_q1_dynamic_Dpc_rr.onnx rn50_image_int8_q1_percentile99999_highres256_s0_B2w4.onnx >> "%LOG%" 2>&1
for %%m in (%B2%_rr rn50_image_int8_q1_dynamic_Ddef rn50_image_int8_q1_dynamic_Dpc_rr rn50_image_int8_q1_percentile99999_highres256_s0_B2w4) do (
  %PY% -u bootstrap.py --a %B2% --b %%m --data %HR% >> "%LOG%" 2>&1
  %PY% -u bootstrap.py --a fp32 --b %%m --data %HR% >> "%LOG%" 2>&1
)
%PY% -u bootstrap.py --a fp32 --b %B2% --data %HR% >> "%LOG%" 2>&1
echo %date% %time% R123 ACC DONE2 >> "%LOG%"
call "%~dp0run_e11_r4.bat"
