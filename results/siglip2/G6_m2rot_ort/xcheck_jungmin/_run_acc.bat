@echo off
rem 정확도: portable_eval.py, dev 6x200, 4스레드 (rn50\.venv 읽기 전용 사용)
cd /d D:\siglip2_xcheck
set OPENBLAS_NUM_THREADS=1
for /d %%d in ("C:\study\3\2\2026*") do set "PY=%%~d\oslab_mobile\rn50\.venv\Scripts\python.exe"
echo %date% %time% start > _acc.log
echo PY=%PY% >> _acc.log
"%PY%" -I rel_xplat\portable_eval.py --data rel_xplat --out out_jungmin --threads 4 --models fp32=rel_xplat\siglip2_image_fp32.onnx R0=rel_xplat\siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx M2rot_ort=rel_m2rot\siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx >> _acc.log 2>&1
echo %date% %time% DONE rc=%errorlevel% >> _acc.log
