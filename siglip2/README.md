# SigLIP2 ViT-B/16 이미지 인코더 양자화 — 재현 방법

대회 안내 4.2절 "재현용 코드 및 실행 환경"에 해당하는 문서다 (초안 v1, 2026-10-15).
실험 기록·결정은 `ROADMAP.md`, `DECISIONS.md`, `PROGRESS.md`, 결과는 루트 `results/siglip2/`에 있다.

> 상태: **초안.** 최종 모델은 ORT fc2L2 U8 + reduce_range(D-030)이고, 4비트 실험(D-037) 결과에 따라 바뀔 수 있다.
> test 평가 명령(5절)은 최종 동결 뒤 한 번만 실행한다. 제출 전에 새 환경에서 이 문서대로 처음부터 실행해 확인한다 (ROADMAP G6).

## 0. 무엇을 재현하나

| 모델 | 파일 | 크기 | sha256 (앞 16자) |
|---|---|---|---|
| 베이스라인 FP32 | `siglip2_image_fp32.onnx` | 354.45 MiB | `333cad08e68dc85a` |
| 최적화 (최종 후보) | `siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx` | 105.31 MiB | `5eee783955591d1d` |

- 전체 sha256: `results/siglip2/candidates/onnx_sha256.txt`. ONNX 파일은 GitHub Release로 받거나 아래 1~3절로 직접 만든다 (D-016).
- 포맷: ONNX opset 17, 입력 `image` float32 N×3×224×224 (SigLIP2 전처리 후), 출력 768차원 이미지 임베딩. 실행: onnxruntime 1.25.0 CPU, 기본 그래프 최적화(ORT_ENABLE_ALL).

## 1. 환경

- Python 3.11, x86-64 또는 ARM64 CPU. GPU 불필요. 디스크 약 20 GB (데이터·캐시·ONNX).
- 저장소를 받은 뒤 `siglip2/` 폴더에서 실행한다. 공통 코드(데이터 선택 규칙, 평가, 속도 측정)는 `rn50/`에서 import한다.

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
PYTHONUTF8=1 python -m pip install -r siglip2/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

환경 변수 (경로는 예시, 모두 저장소 밖):

```bash
export SIGLIP2_DATA=/path/to/data            # 데이터셋 원본 (CIFAR-100, 고화질 parquet)
export SIGLIP2_ARTIFACTS=/path/to/artifacts  # 전처리 캐시, ONNX, 로그
export HF_HOME=/path/to/hf_cache             # 모델 가중치 (timm/ViT-B-16-SigLIP2)
export OPENBLAS_NUM_THREADS=1
```

`RN50_*` 경로는 `s2common.py`가 위 값으로 자동 설정한다 (RN50 파일을 덮어쓰지 않게 막음, D-013).

## 2. 데이터 (dev 캐시·보정 후보)

```bash
cd siglip2
RN50_RESULTS=../results/siglip2/G2_fp32_dev python s2_build_dev.py all --threads 4
```

- CIFAR-100은 torchvision으로, 고화질 5종(Oxford Pets, Flowers-102, Stanford Cars, Food-101, Caltech-101)은 Hugging Face Hub parquet로 자동으로 받는다.
- 사진 선택(dev·보정 후보·test 구간)은 RN50 트랙과 같은 규칙(`rn50/datasets.py`, SHA-1 해시 구간)이다. 보정은 train에서 dev와 겹치지 않게 뽑는다.
- 만들어지는 것: `$SIGLIP2_ARTIFACTS/cache/<set>_{x,y,clf,ref_fp32}.npy`, 보정 후보 `calib_<set>_f32.npy`.

## 3. 모델 만들기

```bash
# FP32 ONNX (opset 17, PyTorch 대비 확인 포함)
python s2_export_onnx.py --out ../results/siglip2/G1_onnx_export --verify 1000
# 최종 INT8: 블록 5·9 MLP fc2만 FP32, 나머지 MatMul INT8, 활성값 U8 + reduce_range, Percentile 99.999, 보정 256장(시드 0)
RN50_RESULTS=../results/siglip2/G3_cumulative/qkv_precision python s2_quantize.py \
  --calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 \
  --act-type u8 --reduce-range --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/" --tag fc2L2 --dump-ops
```

- 같은 버전이면 결과 ONNX는 비트 단위로 같다 (새 가상환경에서 확인, `results/siglip2/G3_repro`). 보정에 실제로 쓰인 장수를 assert한다.

## 4. 정확도 (dev, 정수 커널)

```bash
RN50_RESULTS=../results/siglip2/G3_cumulative/qkv_precision/full python s2_paired_ci.py --models \
  siglip2_image_fp32.onnx siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
```

- 출력: 데이터셋별 top-1, 고화질 5종 평균의 FP32 대비 Δtop-1과 짝 부트스트랩 95% 구간(`paired_ci.jsonl`), 사진별 예측(`pred_<모델>.npz`).
- 기대값 (이 PC, x86 AVX-512 VNNI, ORT 1.25.0): FP32 고화질 평균 91.66%, 최종 모델 Δ −0.78 [−1.26, −0.31]%p, cos 0.955.
- 저장소 없이 다른 기기에서 빠르게 확인: `scripts/portable_eval.py` (numpy + onnxruntime만 필요, dev 6종 × 200장 묶음).

## 5. test (최종 1회)

```bash
# 데이터셋마다 한 번만 (done_<set>.json이 있으면 멈춤). ImageNetV2는 $SIGLIP2_DATA/extra/imagenetv2-mf.tar 필요 (scripts/test_g6.sh가 받음)
for s in pets caltech flowers cars cifar_test imagenetv2 food; do
  RN50_RESULTS=../results/siglip2/G6_test python s2_test_eval.py $s --threads 4     --models siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
done
```

- test 7종: CIFAR-100 test(1만), ImageNetV2 matched-frequency(1만), Oxford Pets·Flowers-102·Stanford Cars·Food-101·Caltech-101 공식 test 전체. 사진·클래스 이름·프롬프트는 `rn50/datasets.py`와 같고, 전처리·텍스트 분류기는 dev와 같은 SigLIP2 것이다.
- 출력: `test.jsonl`(top-1·top-5·FP32 cos), `test_<set>.npz`(사진별 예측). 결과: [test 수치는 평가 뒤 채움]

## 6. 속도·메모리·크기

```bash
# 지연: 배치 1, 워밍업 20, 200장 × 3회 모델 번갈아, 스레드 1·2·4 (rn50/bench_speed.py와 같은 방식)
for t in 1 2 4; do RN50_RESULTS=../results/siglip2/G3_cumulative/speed python s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads $t \
  --models fp32=siglip2_image_fp32.onnx final=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx; done
# 파라미터·GMACs·peak 메모리(모델마다 새 프로세스, 추론 50회 중 최대 RSS)·크기
RN50_RESULTS=../results/siglip2/G3_cumulative/model_stats python s2_model_stats.py --models \
  fp32=siglip2_image_fp32.onnx final=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
```

- 지연은 CPU와 전원 설정에 따라 절대값이 크게 바뀐다. **같은 측정 안에서 FP32 대비 비율로 본다.** 다른 작업이 없는 상태에서 잰다.
- 기대값: x86(VNNI)·ARM 모두 4스레드에서 FP32 대비 약 2배 빠름, peak 메모리 −52%, 크기 −70%.

## 7. 실행 환경 기록

- 결과 jsonl 한 줄마다 커밋 해시, 커밋 안 된 변경 여부(`dirty`), CPU 모델·명령어 플래그(AVX2·AVX-512·VNNI), onnxruntime 버전을 남긴다 (`scripts/env_info.py`, 결과 폴더의 `env.json`).
