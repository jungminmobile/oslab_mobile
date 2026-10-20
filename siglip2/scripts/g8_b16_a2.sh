#!/usr/bin/env bash
# g8_b16_a2.sh — G8 A2: 새 가상환경 재현 (보고서 9장). 깨끗한 venv에 requirements.txt만 설치 → README 3절 순서대로 FP32 export → 회전 → M2 → INT8
#  비교: ① 같은 PC 기존 venv 빌드(A1, onnx_g8)와 sha256 ② 준용 빌드·배포 파일과 sha256 ③ dev 6×200 정확도 (배포 파일과 같은 실행에서)
#  산출물: $ART/onnx_a2 (기존 파일과 섞지 않음). 설치는 잠금 없이(가벼움), 빌드·평가는 잠금 안에서
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
unset S2_MODEL_ID
VENV=D:/mokerton/venvs/siglip2_repro_g8; BASE="C:/Users/qaz43/AppData/Local/Programs/Python/Python311/python.exe"
export SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2 RN50_ONNX=D:/mokerton/artifacts/siglip2/onnx_a2
ART=$SIGLIP2_ARTIFACTS; L=$ART/logs; STOP=/d/mokerton/locks/siglip2.STOP
R=../results/siglip2/G8_candidates/b16/A2_fresh_env; mkdir -p $R $RN50_ONNX
LOCKPY="$SIGLIP2_PYTHON"
chk() { [ -f $STOP ] && { echo "$(date +%T) STOP 파일 — $1 전에 멈춤"; exit 3; }; return 0; }
echo "$(date +%T) start commit=$(git rev-parse --short HEAD)"
if [ ! -x $VENV/Scripts/python.exe ]; then
  "$BASE" -m venv $VENV || exit 1
  $VENV/Scripts/python.exe -m pip install --upgrade pip > $L/a2_pip.log 2>&1
  PYTHONUTF8=1 $VENV/Scripts/python.exe -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu >> $L/a2_pip.log 2>&1 || { echo "pip 실패"; tail -5 $L/a2_pip.log; exit 1; }
fi
PY=$VENV/Scripts/python.exe
$PY -m pip freeze > $R/pip_freeze.txt; $PY --version | tee $R/python_version.txt
diff <("$SIGLIP2_PYTHON" -m pip freeze | sort) <(sort $R/pip_freeze.txt) > $R/pip_freeze_diff_vs_main_venv.txt; echo "freeze diff lines: $(wc -l < $R/pip_freeze_diff_vs_main_venv.txt)"
LOCK=("$LOCKPY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
step() { local name=$1; shift; chk "$name"; echo "$(date +%T) [$name] 시작"; "${LOCK[@]}" "SigLIP2 G8 A2 $name" -- "$@" > "$L/a2_$name.log" 2>&1; local rc=$?; echo "$(date +%T) [$name] rc=$rc"; tail -2 "$L/a2_$name.log" | cut -c1-250; return $rc; }
sha() { "$LOCKPY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }
cmp3() { local f=$1 w=$2; local h=$(sha $RN50_ONNX/$f) a1=$( [ -f $ART/onnx_g8/$f ] && sha $ART/onnx_g8/$f || echo none)
  echo "$f  new=${h:0:16}  A1(같은 PC 기존 venv)=${a1:0:16} $([ "$h" = "$a1" ] && echo SAME || echo DIFF)  준용·배포=${w:0:16} $([ "$h" = "$w" ] && echo SAME || echo DIFF)" | tee -a $R/sha256_compare.txt; }
# README 3절 순서 그대로 (새 venv의 python)
step export env RN50_RESULTS=$R/export $PY -u s2_export_onnx.py --verify 64 --threads 4 --out $R/export || exit 1
cmp3 siglip2_image_fp32.onnx 333cad08e68dc85a5036f8f7396bc61d5548eb69ea365a0c5d83ff15c48840e0
step rot_build env RN50_RESULTS=$R/rotate $PY -u s2_rotate.py build || exit 1
cmp3 rotq_rothad.npy 3a6391a2091db60df7820b2a3598b3ec083682efd482c5b3d6a409b51eeea8eb
step m2_cache env RN50_RESULTS=$R/regcache $PY -u s2_regcache.py cache --rot rothad || exit 1
step m2_ucache env RN50_RESULTS=$R/regcache $PY -u s2_regcache.py ucache --rot rothad || exit 1
step m2_export env RN50_RESULTS=$R/regcache $PY -u s2_regcache.py export --graph mask --mask-form bcast --mask-site mlp --rot rothad || exit 1
cmp3 regcache_55at5-13at9_rothad_highres256_s0.npz c205e3a0e6670837cc2ac18fc5d4d7e4cd4ed5c482e574151d0ea5aa739e7f3e
cmp3 regcache_uc_55at5-13at9_rothad_highres256_s0.npz 00d8ee3d18e81c2269b5a4d1a32b4a18cffcc57e064a8388842094da169a515b
cmp3 siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx 16ef696e8016f4d1f85f1cafc1d98d32882a813d83e75c179f2cfffa55c36401
F=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
step quantize env RN50_RESULTS=$R/int8 $PY -u s2_quantize.py --src siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx --calib-source highres --calib-n 256 --op-types MatMul \
  --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --tag all --dump-ops --threads 4 || exit 1
cmp3 $F e84c3727f51e3b97cd59d9ab9604e01acc78f81493926a0696fbf96dc9c60f9e
# dev 6×200: 새 환경 빌드 vs 배포 파일 (같은 실행, 새 venv의 ORT)
[ -f $RN50_ONNX/siglip2_image_int8_release_m2rot_ort.onnx ] || cp $ART/onnx/$F $RN50_ONNX/siglip2_image_int8_release_m2rot_ort.onnx
cmp3 siglip2_image_int8_release_m2rot_ort.onnx e84c3727f51e3b97cd59d9ab9604e01acc78f81493926a0696fbf96dc9c60f9e
step screen env RN50_RESULTS=$R/screen $PY -u s2_screen.py --k 200 --tag a2 --models $F siglip2_image_int8_release_m2rot_ort.onnx
echo "$(date +%T) A2 DONE"
