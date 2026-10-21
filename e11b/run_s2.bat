@echo off
rem E11b S2: SigLIP2 R0 Bias Correction (zero-delta 관문 → BC(256장) → BC'(55장) → dev 전체 정확도)
chcp 65001 >nul
cd /d "%~dp0..\..\oslab_s2\siglip2"
set "PY=%~dp0..\..\s2_venv\Scripts\python.exe"
set "SIGLIP2_DATA=%~dp0..\..\s2data"
set "SIGLIP2_ARTIFACTS=C:\s2art"
set "RN50_RESULTS=C:\s2art\E11b"
set OPENBLAS_NUM_THREADS=1
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8
set "LOG=C:\s2art\E11b\run_log.txt"
set R0=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2
echo %date% %time% S2 zero >> "%LOG%"
"%PY%" -u "%~dp0s2_bias_correct.py" --src %R0%.onnx --tag BCzero --zero-delta >> "%LOG%" 2>&1
"%PY%" -c "import hashlib;h=hashlib.sha256(open(r'C:\s2art\onnx\%R0%_BCzero.onnx','rb').read()).hexdigest();print('ZEROSHA',h,h.startswith('cb710af4'))" >> "%LOG%" 2>&1
echo %date% %time% S2 BC >> "%LOG%"
"%PY%" -u "%~dp0s2_bias_correct.py" --src %R0%.onnx --tag BC --calib-seed 0 >> "%LOG%" 2>&1
"%PY%" -u "%~dp0s2_bias_correct.py" --src %R0%.onnx --tag BCp --calib-seed 1 >> "%LOG%" 2>&1
echo %date% %time% S2 eval >> "%LOG%"
"%PY%" -u s2_paired_ci.py --threads 6 --models %R0%_BC.onnx %R0%_BCp.onnx >> "%LOG%" 2>&1
echo %date% %time% S2 DONE >> "%LOG%"
