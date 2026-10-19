@echo off
rem E5a: CLE / SmoothQuant식 채널 균형 후 B0·B2 양자화 → 같은 노트북에서 만든 원본 B0·B2와 짝 비교
chcp 65001 >nul
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe
set RN50_RESULTS=..\results\E5a_cle
set LOG=..\results\E5a_cle_log.txt
set AP=/attnpool/MatMul,/attnpool/MatMul_1,/attnpool/MatMul_2,/attnpool/Gemm
set STEM=/conv1,/conv2,/conv3,/act1,/act2,/act3,/avgpool
set REST=/layer1/layer1.1/,/layer1/layer1.2/,/layer2/layer2.0/,/layer2/layer2.1/,/layer2/layer2.2/,/layer2/layer2.3/,/layer3/layer3.0/,/layer3/layer3.1/,/layer3/layer3.2/,/layer3/layer3.3/,/layer3/layer3.4/,/layer3/layer3.5/,/layer4/layer4.0/,/layer4/layer4.1/,/layer4/layer4.2/
set ONLY_B0=%STEM%,/layer1/layer1.0/,%REST%
set ONLY_B2=%REST%
set HR=pets_dev flowers_dev cars_dev food_dev caltech_dev
echo %date% %time% start > %LOG%
%PY% fetch_highres.py >> %LOG% 2>&1
echo %date% %time% build dev caches >> %LOG%
for %%n in (pets flowers cars food caltech) do %PY% datasets.py dev %%n >> %LOG% 2>&1
echo %date% %time% equalize >> %LOG%
%PY% equalize.py --method cle >> %LOG% 2>&1
%PY% equalize.py --method sq --alpha 0.5 >> %LOG% 2>&1
for %%v in (_q1 _q1cle _q1sq05) do (
  for %%b in (B0 B2) do (
    echo %date% %time% quantize %%v %%b >> %LOG%
    if "%%b"=="B0" (%PY% quantize_onnx.py --enc image --variant %%v --calib-source highres --calib-n 256 --seed 0 --method percentile --only %ONLY_B0% --only-exact %AP% --tag %%b --overwrite >> %LOG% 2>&1)
    if "%%b"=="B2" (%PY% quantize_onnx.py --enc image --variant %%v --calib-source highres --calib-n 256 --seed 0 --method percentile --only %ONLY_B2% --only-exact %AP% --tag %%b --overwrite >> %LOG% 2>&1)
    %PY% evaluate.py --exec int --threads 4 --save-pred --data cifar_A %HR% --models rn50_image_int8%%v_percentile99999_highres256_s0_%%b.onnx >> %LOG% 2>&1
    %PY% bootstrap.py --a fp32 --b rn50_image_int8%%v_percentile99999_highres256_s0_%%b --data %HR% >> %LOG% 2>&1
    %PY% bootstrap.py --a rn50_image_int8_q1_percentile99999_highres256_s0_%%b --b rn50_image_int8%%v_percentile99999_highres256_s0_%%b --data %HR% >> %LOG% 2>&1
  )
)
echo %date% %time% ALL DONE >> %LOG%
