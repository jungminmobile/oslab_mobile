#!/usr/bin/env bash
# g8_b16_a1.sh — G8 A1: B/16 기법별 단독 효과 2×2 (원래 / 회전만 / M2만 / 회전+M2), 모두 MatMul 전부 INT8·같은 양자화 옵션, dev 전체
#  이 PC에서 회전·M2·M2rot FP32를 다시 만들고(준용 빌드 sha256과 비교 = 재현 확인), 별도 폴더 onnx_g8에 둔다 (배포 M2rot_ort 파일 덮어쓰기 방지)
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
unset S2_MODEL_ID
export SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2 RN50_ONNX=D:/mokerton/artifacts/siglip2/onnx_g8
PY="$SIGLIP2_PYTHON"; ART=$SIGLIP2_ARTIFACTS; L=$ART/logs; STOP=/d/mokerton/locks/siglip2.STOP
R=../results/siglip2/G8_candidates/b16/A1_ablation; mkdir -p $R $RN50_ONNX
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
chk() { [ -f $STOP ] && { echo "$(date +%T) STOP 파일 — $1 전에 멈춤"; exit 3; }; return 0; }
step() { local name=$1; shift; chk "$name"; echo "$(date +%T) [$name] 시작"; "${LOCK[@]}" "SigLIP2 G8 A1 $name" -- "$@" > "$L/a1_$name.log" 2>&1; local rc=$?; echo "$(date +%T) [$name] rc=$rc"; tail -2 "$L/a1_$name.log"; return $rc; }
sha() { "$PY" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$1"; }
want() { local f=$1 w=$2 h; h=$(sha "$f"); echo "sha256 $(basename $f) $h want ${w:0:16} $([ "$h" = "$w" ] && echo SAME || echo DIFFERENT)" | tee -a $R/sha256_check.txt; }
echo "$(date +%T) start commit=$(git rev-parse --short HEAD)"
F0=siglip2_image_fp32.onnx; FR=siglip2_image_fp32_rothad.onnx; FM=siglip2_image_fp32_rcmaskbcmlp55at5-13at9.onnx; FMR=siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx
[ -f $RN50_ONNX/$F0 ] || cp $ART/onnx/$F0 $RN50_ONNX/$F0
want $RN50_ONNX/$F0 $(sha $ART/onnx/$F0)   # 원본 복사 확인 (원본 앞 16자 333cad08e68dc85a)
step rot_build env RN50_RESULTS=$R/rotate "$PY" -u s2_rotate.py build || exit 1
want $RN50_ONNX/rotq_rothad.npy 3a6391a2091db60df7820b2a3598b3ec083682efd482c5b3d6a409b51eeea8eb
step rot_export env RN50_RESULTS=$R/rotate "$PY" -u s2_rotate.py export --q rothad || exit 1
for rot in none rothad; do
  step m2_cache_$rot env RN50_RESULTS=$R/regcache "$PY" -u s2_regcache.py cache --rot $rot || exit 1
  step m2_ucache_$rot env RN50_RESULTS=$R/regcache "$PY" -u s2_regcache.py ucache --rot $rot || exit 1
  step m2_export_$rot env RN50_RESULTS=$R/regcache "$PY" -u s2_regcache.py export --graph mask --mask-form bcast --mask-site mlp --rot $rot || exit 1
done
want $RN50_ONNX/regcache_55at5-13at9_rothad_highres256_s0.npz c205e3a0e6670837cc2ac18fc5d4d7e4cd4ed5c482e574151d0ea5aa739e7f3e
want $RN50_ONNX/regcache_uc_55at5-13at9_rothad_highres256_s0.npz 00d8ee3d18e81c2269b5a4d1a32b4a18cffcc57e064a8388842094da169a515b
want $RN50_ONNX/$FMR 16ef696e8016f4d1f85f1cafc1d98d32882a813d83e75c179f2cfffa55c36401
QOPT=(--calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --tag all --dump-ops --threads 4)
for f in $F0 $FR $FM $FMR; do
  step q_${f%.onnx} env RN50_RESULTS=$R/int8 "$PY" -u s2_quantize.py --src $f "${QOPT[@]}" || exit 1
done
ls $RN50_ONNX/*int8* | tee $R/int8_files.txt
MR=$(ls $RN50_ONNX | grep -E "int8_rcmaskbcmlp55at5-13at9_rothad_.*_rr_all.onnx$")
want $RN50_ONNX/$MR e84c3727f51e3b97cd59d9ab9604e01acc78f81493926a0696fbf96dc9c60f9e
MODELS=$(ls $RN50_ONNX | grep -E "int8_.*_u8_ops-MatMul_hc16_rr_all.onnx$" | tr '\n' ' ')
step devfull env RN50_RESULTS=$R/dev_full "$PY" -u s2_paired_ci.py --models $MODELS
echo "$(date +%T) A1 DONE"
