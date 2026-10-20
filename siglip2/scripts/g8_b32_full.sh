#!/usr/bin/env bash
# g8_b32_full.sh — G8 B3~B11: B/16에서 한 실험을 B/32-256에 똑같이 (쭈리 10/20 "B/32도 동일한 실험 모두"). test는 하지 않음 (팀이 B/32를 고른 뒤 B13)
#  B/16 스크립트(g3_sensitivity·g3_calib·g3_cumulative)는 결과 폴더만 바꾼 사본으로 그대로 실행 (사본은 $ART/scripts_b32, 바꾼 줄은 로그에 남김)
#  단계마다 공용 잠금, 끝난 단계는 $R/.done_<단계>로 건너뜀 (다시 실행하면 이어서). STOP 파일이면 단계 사이에서 멈춤
#  사용: bash scripts/g8_b32_full.sh [noSpeed]   — 속도(B11)는 PC가 조용할 때만 (needs idle PC)
set -uo pipefail
cd /d/mokerton/wt_g8/siglip2
export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' HF_HUB_DISABLE_SYMLINKS_WARNING=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2_b32
unset RN50_ONNX RN50_CACHE
PY="$SIGLIP2_PYTHON"; ART=$SIGLIP2_ARTIFACTS; L=$ART/logs; STOP=/d/mokerton/locks/siglip2.STOP
R=../results/siglip2/G8_candidates/b32; mkdir -p $R $L $ART/scripts_b32
LOCK=("$PY" D:/mokerton/locks/heavy_lock.py run --owner siglip2 --job)
GB="C:/Program Files/Git/bin/bash.exe"   # 잠금(Windows python)이 부르는 "bash"는 WSL bash일 수 있음 (10/20 rc=127) → Git Bash를 직접 지정
chk() { [ -f $STOP ] && { echo "$(date +%T) STOP 파일 — $1 전에 멈춤"; exit 3; }; return 0; }
step() { local name=$1; shift; [ -f $R/.done_$name ] && { echo "$(date +%T) [$name] 이미 끝남 — 건너뜀"; return 0; }
  chk "$name"; echo "$(date +%T) [$name] 시작"; "${LOCK[@]}" "SigLIP2 G8 B/32 $name" -- "$@" > "$L/full_$name.log" 2>&1; local rc=$?
  echo "$(date +%T) [$name] rc=$rc"; tail -2 "$L/full_$name.log" | cut -c1-300; [ $rc -eq 0 ] && touch $R/.done_$name; return $rc; }
timed() { local name=$1; shift; [ -f $R/.done_$name ] && return 0; while true; do chk "$name"
  "${LOCK[@]}" "SigLIP2 G8 B/32 $name (needs idle PC)" -- "C:/Program Files/Git/bin/bash.exe" scripts/_idle_then.sh "$@" > "$L/full_$name.log" 2>&1; local rc=$?
  echo "$(date +%T) [$name] rc=$rc"; [ $rc -eq 75 ] && { sleep 180; continue; }; [ $rc -eq 0 ] && touch $R/.done_$name; return $rc; done; }
B32S() { local src=$1 dst=$ART/scripts_b32/$1; sed -e 's#^R=\.\./results/siglip2/G3_#R=../results/siglip2/G8_candidates/b32/G3_#' scripts/$src > $dst
  diff scripts/$src $dst | tee -a $L/scripts_b32_diff.txt >/dev/null; echo $dst; }
echo "$(date +%T) start commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no -- . | wc -l)"
F0=siglip2_image_fp32.onnx; Q0=siglip2_image_int8_minmax_highres256_s0_s8_all.onnx
DEV6=(cifar_A pets_dev flowers_dev cars_dev food_dev caltech_dev)
# ---- B3·B2 나머지: 단순 INT8 붕괴, 프롬프트 비교, FP32 dev(7종), 모델 통계 (G1·G2와 같음)
step int8_all env RN50_RESULTS=$R/G1_int8_minmax_all "$PY" -u s2_quantize.py --calib-source highres --calib-n 256 --tag all --dump-ops --overwrite --threads 4 || exit 1
step int8_all_eval env RN50_RESULTS=$R/G1_int8_minmax_all "$PY" -u s2_evaluate.py --data ${DEV6[@]} --models fp32 $Q0 --exec int --threads 4
step prompt env RN50_RESULTS=$R/G2_prompt_compare "$PY" -u s2_prompt_compare.py
step fp32_dev env RN50_RESULTS=$R/G2_fp32_dev "$PY" -u s2_evaluate.py --data cifar_A cifar_B pets_dev flowers_dev cars_dev food_dev caltech_dev --models fp32 --exec int --threads 4
step stats0 env RN50_RESULTS=$R/G2_fp32_baseline "$PY" -u s2_model_stats.py --threads 4 --models fp32=$F0 int8_all=$Q0
# ---- B4 민감도 지도 (블록 14 + 연산 유형 6), B6 보정 방법, B7 혼합 정밀도 — B/16 스크립트 사본
step sensitivity "$GB" $(B32S g3_sensitivity.sh) all || exit 1
step calib "$GB" $(B32S g3_calib.sh) || exit 1
CUM=$(B32S g3_cumulative.sh)
step cum_screen "$GB" $CUM screen || exit 1
step cum_screen2 "$GB" $CUM screen2 || exit 1
step cum_full "$GB" $CUM full
# R0와 같은 구성 (fc2L2, U8 + reduce_range) — B/16 이전 최종
R0=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
step r0 env RN50_RESULTS=$R/G3_cumulative/r0 "$PY" -u s2_quantize.py --calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 \
  --act-type u8 --reduce-range --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/" --tag fc2L2 --dump-ops --overwrite --threads 4 || exit 1
# ---- B5 원인 분석 (거대 활성값·쓰는 MLP·개입)
step why_massive env RN50_RESULTS=$R/G3_why "$PY" -u s2_why_massive.py --n 32
step why_writers env RN50_RESULTS=$R/G3_why "$PY" -u s2_why_writers.py --n 32
step why_fix env RN50_RESULTS=$R/G3_why "$PY" -u s2_why_fix.py --n 32
# ---- B8 회전·M2·M2rot (G6 통일 규칙 15@6:5, 7@10:9) + 2×2 (B/16 A1과 같은 양자화)
SINKS="15@6:5,7@10:9"
step rot_build env RN50_RESULTS=$R/G3RQ/rotate "$PY" -u s2_rotate.py build || exit 1
step rot_export env RN50_RESULTS=$R/G3RQ/rotate "$PY" -u s2_rotate.py export --q rothad || exit 1
for rot in none rothad; do
  step m2_cache_$rot env RN50_RESULTS=$R/G3RQ/regcache "$PY" -u s2_regcache.py cache --sinks $SINKS --rot $rot || exit 1
  step m2_ucache_$rot env RN50_RESULTS=$R/G3RQ/regcache "$PY" -u s2_regcache.py ucache --sinks $SINKS --rot $rot || exit 1
  step m2_export_$rot env RN50_RESULTS=$R/G3RQ/regcache "$PY" -u s2_regcache.py export --sinks $SINKS --graph mask --mask-form bcast --mask-site mlp --rot $rot || exit 1
done
FR=siglip2_image_fp32_rothad.onnx; FM=siglip2_image_fp32_rcmaskbcmlp15at6u5-7at10u9.onnx; FMR=siglip2_image_fp32_rcmaskbcmlp15at6u5-7at10u9_rothad.onnx
step m2approx env RN50_RESULTS=$R/G3RQ "$PY" -u s2_screen.py --k 200 --tag m2fp32 --models $FM $FMR
QOPT=(--calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --tag all --dump-ops --threads 4 --overwrite)
for f in $F0 $FR $FM $FMR; do step q_${f%.onnx} env RN50_RESULTS=$R/G3RQ/int8 "$PY" -u s2_quantize.py --src $f "${QOPT[@]}" || exit 1; done
MODELS=$(ls $ART/onnx | grep -E "int8_.*_u8_ops-MatMul_hc16_rr_all.onnx$" | tr '\n' ' ')
MAIN=siglip2_image_int8_rcmaskbcmlp15at6u5-7at10u9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
step devfull env RN50_RESULTS=$R/G3RQ/dev_full "$PY" -u s2_paired_ci.py --models $MODELS $R0
# ---- B9 4비트 (B/16 G3_w4의 A: 모든 MLP fc1, 주 후보 위에서), B10 오류 클래스
#  B/16 G3_w4와 같은 4구성(A fc1 / B fc1·fc2 / C +qkv·proj / D +풀링 헤드), R0 구성 위에서, dev 6×200 스크리닝
declare -A RX=([A]='/blocks\.\d+/mlp/fc1/MatMul$' [B]='/blocks\.\d+/mlp/fc[12]/MatMul$' [C]='/blocks\.\d+/(mlp/fc[12]|attn/(qkv|proj))/MatMul$' [D]='(/blocks\.\d+/(mlp/fc[12]|attn/(qkv|proj))|/attn_pool/(q|kv|proj|mlp/fc[12]))/MatMul$')
W4Q=(--calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/" --threads 4 --overwrite)
W4M=()
for k in A B C D; do step w4_$k env RN50_RESULTS=$R/G3_w4 "$PY" -u s2_quantize.py "${W4Q[@]}" --w4-regex "${RX[$k]}" --tag w4${k}_fc2L2
  f=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_w4b32a4_w4${k}_fc2L2.onnx; [ -f $ART/onnx/$f ] && W4M+=($f); done
[ ${#W4M[@]} -gt 0 ] && step w4_screen env RN50_RESULTS=$R/G3_w4 "$PY" -u s2_screen.py --k 200 --tag w4 --models $R0 "${W4M[@]}"
step error_classes env RN50_RESULTS=$R/G3_error_classes "$PY" -u s2_error_classes.py --preds main=$R/G3RQ/dev_full/pred_${MAIN%.onnx}.npz r0=$R/G3RQ/dev_full/pred_${R0%.onnx}.npz
# ---- B11 속도·메모리 (이 PC, PC 조용할 때)
step stats_final env RN50_RESULTS=$R/G6_speed/model_stats "$PY" -u s2_model_stats.py --threads 4 --models fp32=$F0 r0=$R0 m2rot=$MAIN
[ "${1:-}" = noSpeed ] && { echo "$(date +%T) B/32 (속도 제외) DONE"; exit 0; }
for s in a b c; do for t in 1 2 4; do
  timed speed_${s}_t$t env RN50_RESULTS=$R/G6_speed/speed_$s "$PY" -u s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t --models fp32=$F0 r0=$R0 m2rot=$MAIN
done; done
echo "$(date +%T) B/32 FULL DONE"
