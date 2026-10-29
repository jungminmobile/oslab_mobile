#!/usr/bin/env bash
# g8_b32_stage1.sh — G8 B1~B3: SigLIP2 B/32-256을 B/16 G1·G2와 같은 순서로 (export·검증 → dev 캐시 → 단순 INT8 붕괴 → 프롬프트 → FP32 dev → 모델 통계)
#  B/32 산출물은 D:/mokerton/artifacts/siglip2_b32 (model_id.txt로 확인), 결과는 results/siglip2/G8_candidates/b32/<B/16 폴더 이름>
#  단계마다 공용 잠금. STOP 파일이면 단계 사이에서 멈춤. 속도 측정은 여기서 하지 않음 (B11, 조용할 때)
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2_b32
PY="$SIGLIP2_PYTHON"; ART=$SIGLIP2_ARTIFACTS; L=$ART/logs; STOP=/d/mokerton/locks/siglip2.STOP
R=../results/siglip2/G8_candidates/b32; mkdir -p "$L" $R
Q=siglip2_image_int8_minmax_highres256_s0_s8_all.onnx
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
chk() { [ -f $STOP ] && { echo "$(date +%T) STOP 파일 — $1 전에 멈춤"; exit 3; }; return 0; }
step() { local name=$1; shift; chk "$name"; echo "$(date +%T) [$name] 시작"; "${LOCK[@]}" "SigLIP2 G8 B/32 $name" -- "$@" > "$L/stage1_$name.log" 2>&1; local rc=$?; echo "$(date +%T) [$name] rc=$rc"; tail -3 "$L/stage1_$name.log"; return $rc; }
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -- . | wc -l)"
step export   "$PY" -u s2_export_onnx.py --verify 64 --ops --threads 4 --out $R/G1_onnx_export || exit 1
step reexport env RN50_ONNX=$ART/onnx_tmp_reexport "$PY" -u s2_export_onnx.py --overwrite --threads 4 --out $ART/logs/reexport_check
"$PY" -c "import hashlib,sys; h=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest(); a,b=sys.argv[1:3]; print(h(a)); print(h(b)); print('same', h(a)==h(b))" $ART/onnx/siglip2_image_fp32.onnx $ART/onnx_tmp_reexport/siglip2_image_fp32.onnx | tee $R/G1_onnx_export/reexport_sha256.txt
rm -rf $ART/onnx_tmp_reexport
step build_dev env RN50_RESULTS=$R/G2_fp32_dev "$PY" -u s2_build_dev.py all --threads 4 || exit 1
step int8_all env RN50_RESULTS=$R/G1_int8_minmax_all "$PY" -u s2_quantize.py --calib-source highres --calib-n 256 --tag all --dump-ops --overwrite --threads 4
step int8_all_eval env RN50_RESULTS=$R/G1_int8_minmax_all "$PY" -u s2_evaluate.py --data cifar_A pets_dev flowers_dev cars_dev food_dev caltech_dev --models fp32 $Q --exec int --threads 4
step prompt env RN50_RESULTS=$R/G2_prompt_compare "$PY" -u s2_prompt_compare.py
step fp32_dev env RN50_RESULTS=$R/G2_fp32_dev "$PY" -u s2_evaluate.py --data cifar_A cifar_B pets_dev flowers_dev cars_dev food_dev caltech_dev --models fp32 --exec int --threads 4
step stats env RN50_RESULTS=$R/G2_fp32_baseline "$PY" -u s2_model_stats.py --threads 4 --models fp32=siglip2_image_fp32.onnx int8_all=$Q
echo "$(date +%T) STAGE1 DONE"
