# SigLIP2 ViT-B/16 이미지 인코더 양자화 — 재현 방법

대회 안내 4.2절 "재현용 코드 및 실행 환경"과 예선 결과보고서 9장에 해당하는 문서다 (v3, 2026-10-21 — 제출 모델 M2rot_ort, D-046).
실험 기록·결정은 `ROADMAP.md`, `DECISIONS.md`, `PROGRESS.md`, 결과는 루트 `results/siglip2/`에 있다.

> **제출 모델 = M2rot_ort** (D-044·D-046): 레지스터 캐시 M2 + 회전, 이미지 인코더 MatMul 79개 전부 INT8. test는 모델당 1회 썼다 (`results/siglip2/G6_m2rot_ort`).
> **본선 Hailo HEF = B16_rb_bk** (D-047): 같은 M2rot 그래프 + 컴파일 자원 설정 (`results/siglip2/G6c`, 준용).
> **채점·재현은 배포한 ONNX 파일로 한다.** 다른 PC에서 다시 만들면 파일이 달라진다 (아래 3절, 7절).

## 0. 무엇을 재현하나

| 모델 | 파일 | 크기 | sha256 | Release |
|---|---|---|---|---|
| 베이스라인 FP32 | `siglip2_image_fp32.onnx` | 371.67 MB (354.45 MiB) | `333cad08e68dc85a5036f8f7396bc61d5548eb69ea365a0c5d83ff15c48840e0` | `siglip2-xplat-20261015` |
| **최적화 (제출)** | `siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx` | 96.18 MB (91.72 MiB) | `e84c3727f51e3b97cd59d9ab9604e01acc78f81493926a0696fbf96dc9c60f9e` | `siglip2-m2rot-ort-20261019` |
| (참고) M2rot FP32 | `siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx` | 371.61 MB | `16ef696e8016f4d1…` | `siglip2-m2rot-ort-20261019` |

- 포맷: ONNX opset 17, 입력 `image` float32 N×3×224×224 (SigLIP2 전처리 후: bicubic으로 224×224, 평균·표준편차 0.5 정규화), 출력 768차원 이미지 임베딩.
- 실행: onnxruntime 1.25.0 CPU, 기본 그래프 최적화(`ORT_ENABLE_ALL`, 정수 커널).
- 같은 제출 파일이 x86 세 CPU(AVX-512 VNNI i7-1165G7, AVX-VNNI Ultra 9 285K·Ultra 7 258V)에서 dev 결과가 같고, i7과 258V는 사진별 출력까지 비트 단위로 같다 (`results/siglip2/G6_m2rot_ort`).

## 1. 빠른 확인 (저장소 코드·데이터 없이, 내려받기 약 3분 + 실행 약 8분)

numpy와 onnxruntime만 있으면 된다. dev 6종 × 200장 묶음으로 FP32와 제출 모델을 비교한다.

```bash
python -m pip install numpy==2.4.4 onnxruntime==1.25.0
gh release download siglip2-xplat-20261015 --repo jungminmobile/oslab_mobile -D release -p "*.npz" -p "portable_eval.py" -p "siglip2_image_fp32.onnx" -p "SHA256SUMS.txt"
gh release download siglip2-m2rot-ort-20261019 --repo jungminmobile/oslab_mobile -D release -p "siglip2_image_int8_*"
python release/portable_eval.py --data release --out out_quick --threads 4 \
  --models fp32=release/siglip2_image_fp32.onnx \
           int8=release/siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
```
- 기대 출력 (`out_quick/result.json`): INT8 고화질 5종 cos **0.9756**, Δtop-1 **+0.3%p** (각 200장). x86 VNNI PC에서는 같은 값이 나온다.
- 확인 기록: 깨끗한 폴더에서 이 순서 그대로 실행해 같은 값 (`results/siglip2/G8_candidates/b16/repro_procedure`).
- `gh`가 없으면 GitHub Release 페이지에서 같은 파일을 받아 `release/` 폴더 하나에 둔다 (dev 묶음 6개는 폴더 없이 들어 있다).

## 2. 환경 (전체 재현)

- Python 3.11, x86-64 또는 ARM64 CPU. GPU 불필요. 디스크 약 20 GB (데이터·캐시·ONNX).

```bash
git clone https://github.com/jungminmobile/oslab_mobile.git && cd oslab_mobile
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r siglip2/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
export SIGLIP2_DATA=/path/to/data SIGLIP2_ARTIFACTS=/path/to/artifacts HF_HOME=/path/to/hf_cache OPENBLAS_NUM_THREADS=1
mkdir -p $SIGLIP2_ARTIFACTS/onnx && cp release/*.onnx $SIGLIP2_ARTIFACTS/onnx/       # 1절에서 받은 모델
```
- Windows PowerShell: `$env:SIGLIP2_DATA="D:\data"` 처럼 쓴다.
- `RN50_*` 경로는 `s2common.py`가 위 값으로 자동 설정한다 (RN50 파일을 덮어쓰지 않게 막음, D-013).

## 3. 모델 만들기 (레시피 — 재현은 배포 파일로)

1. FP32 ONNX export: `python s2_export_onnx.py --out ../results/siglip2/G1_onnx_export --verify 1000`
2. 회전 행렬: `RN50_RESULTS=… python s2_rotate.py build` (rothad = 무작위 부호 Hadamard × Householder, 직교성 오차 ~1e-15)
3. 레지스터 캐시 M2 + 회전 (보정 train 256장, 시드 0):
   `python s2_regcache.py cache --rot rothad` → `python s2_regcache.py ucache --rot rothad` → `python s2_regcache.py export --graph mask --mask-form bcast --mask-site mlp --rot rothad`
4. INT8: `python s2_quantize.py --src siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx --calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --tag all --dump-ops` (MatMul 79개 전부 INT8, 보정에 실제로 쓰인 장수를 assert)

**같은 PC에서는 다시 만들어도 비트 단위로 같은 파일이 나오지만, 다른 PC에서는 어느 단계에서 시작해도 다른 파일이 나온다.**
- float64 행렬 곱의 반올림이 CPU·OpenBLAS 커널에 따라 달라 회전 행렬부터 달라진다 (`results/siglip2/G8_candidates/repro_fingerprint`).
- 다른 PC가 만든 M2rot FP32에서 양자화만 해도 INT8이 다르다 — 보정 때 CPU로 FP32 추론해 범위를 잰다. 같은 회전 행렬·캐시로 FP32를 만들어도 다르다 (`…/b16/final_checks` E2).
- 정확도 차이는 0.1~0.3%p다 (보정 시드·빌드 PC를 바꾼 5개 빌드에서 dev −0.23 ~ −0.49, `…/b16/E5_calib_seed`). 그래서 **채점·재현은 배포 파일**로 한다.

## 4. 정확도 — dev (구성 비교용)

```bash
cd siglip2
python s2_build_dev.py all --threads 4        # dev 캐시·텍스트 분류기·FP32 기준 (데이터 자동 다운로드, 처음 1~2시간)
RN50_RESULTS=../results/siglip2/repro_dev python s2_paired_ci.py --models siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
```
- 기대값: FP32 고화질 91.66%, 제출 모델 91.44% (Δ **−0.23 [−0.64, +0.18]**), cos 0.976, CIFAR A −4.3. 출력: `paired_ci.jsonl`, 사진별 예측 `pred_<모델>.npz`.
- `paired_ci.jsonl`은 이어 쓰기라 재현할 때는 새 결과 폴더를 지정한다.

## 5. 정확도 — test (최종 1회)

```bash
for s in pets caltech flowers cars cifar_test imagenetv2 food; do
  RN50_RESULTS=../results/siglip2/repro_test python s2_test_eval.py $s --threads 4 \
    --models siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
done
RN50_RESULTS=../results/siglip2/repro_test python s2_test_summary.py
```
- test 7종: CIFAR-100 test(1만), ImageNetV2 matched-frequency(1만, `scripts/test_g6.sh`가 받음), Oxford Pets·Flowers-102·Stanford Cars·Food-101·Caltech-101 공식 test 전체.
- 기대값: 고화질 5종 평균 FP32 90.34% → 90.00%, Δ **−0.34 [−0.50, −0.18]%p**. CIFAR-100 −4.09, ImageNetV2 −0.72%p.
- 시간: 4코어 노트북에서 약 9~10시간 (FP32와 INT8을 함께 잼). 데이터셋마다 `done_<set>.json`이 있으면 다시 돌지 않는다.

## 6. 속도·메모리·크기

```bash
python scripts/cross_latency.py --out out_speed --mem \
  --models b16_fp32=$SIGLIP2_ARTIFACTS/onnx/siglip2_image_fp32.onnx \
           b16_int8=$SIGLIP2_ARTIFACTS/onnx/siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
```
- 배치 1, 워밍업 20, 200장 × 3회 모델 번갈아, 스레드 1·2·4 × 세션 3 → 중앙값. 전원 구성표·충전 상태가 결과에 기록된다.
- 기대값: FP32 대비 4스레드 약 −50~56%, 1스레드 약 −52~63% (기기마다 절대 ms는 다름), 피크 메모리 약 −52%, 크기 371.67 → 96.18 MB (−74.1%).
- **지연은 같은 실행 안의 FP32 대비 비율로 본다.** 노트북은 고성능 모드·충전 상태, 다른 프로그램 없이.

## 7. 실행 환경 기록

- 결과 jsonl 한 줄마다 커밋 해시, 커밋 안 된 변경 여부(`dirty`), CPU 모델·명령어 플래그(AVX2·AVX-512·VNNI), onnxruntime 버전을 남긴다 (`scripts/env_info.py`, 결과 폴더의 `env.json`).
- PC마다 빌드가 갈리는 지점을 찾는 지문: `scripts/repro_fingerprint.py` (버전·BLAS·회전 행렬 재계산·보정 사진·캐시·ONNX sha256).
