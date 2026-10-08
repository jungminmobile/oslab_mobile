# CLIP-RN50 양자화 실험 코드 (정리본)

2026 모바일공학 공모전 트랙 2(자유주제). CLIP-RN50을 Raspberry Pi 5 + Hailo-10H에 올리기 위한 예선용 데스크톱 실험 코드예요.
모든 파일 맨 위에 **목적과 설명**을 주석으로 달았어요. 코드 검토 결과는 `CODE_REVIEW.md`에 있어요.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `common.py` | 공통 설정: 경로, 전처리, 토크나이저, 데이터 구간 규칙, ONNX 세션, 임베딩·분류기·지표 |
| `baseline_rn50.py` | PyTorch 원본 FP32 베이스라인 (CIFAR-100 test 정확도, 크기, 지연, 메모리) |
| `export_onnx.py` | PyTorch → ONNX 변환. `--attnpool-q1`로 AttnPool 최적화본 생성, `--verify`로 동일성 확인 |
| `quantize_onnx.py` | ONNX Runtime 정적 INT8 양자화. 노드 선택(`--exclude/--only`), 보정 데이터 선택(`--calib-source`) |
| `make_stem_combos.py` | stem 내부 조합 15개 모델 생성 (실험 2-2) |
| `datasets.py` | dev(설정 선택용, train 분할) 캐시·보정 후보 생성 / test 전체 최종 평가 |
| `evaluate.py` | 여러 모델 × 여러 데이터셋의 top-1/top-5/원본과의 유사도 |
| `bench_speed.py` | 속도 반복 측정 (번갈아 측정, 평균·흔들림) |
| `summarize.py` | evaluate.jsonl → 모델별 dev 정확도·FP32 대비 변화·유사도 표 |
| `clip_weights.py` | 4비트용 채널별 MSE 최적 가중치 잘라내기 (R6b) |
| `night_run.py` | 10/7 야간 실험(PRD R1~R8, X0~X8) 자동 실행기. 이어하기 지원 |
| `prompts/cifar100_templates.json` | CIFAR-100 프롬프트 18개 (CLIP 논문) |

## 환경

- Python 3.11, CPU 전용
- 모델 선택: 환경변수 `CLIP_MODEL=RN50`(기본) 또는 `RN50x4`(입력 288). RN50x4는 보정 배치 2 권장(`--calib-bs 2`, 메모리)
- `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`
- 기본 경로는 이 폴더 안(`data/`, `extra_data/`, `onnx/`, `cache/`, `results/`). 환경변수 `RN50_DATA`, `RN50_EXTRA`, `RN50_ONNX`, `RN50_CACHE`, `RN50_RESULTS`로 바꿀 수 있어요.

## 데이터 받기

```bash
mkdir -p weights extra_data/{flowers,cars,food,caltech}
# OpenAI 원본 가중치 (sha256 afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762)
curl -L -o weights/RN50.pt https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt
# CIFAR-100은 처음 실행 때 torchvision이 data/에 자동으로 받음
HF=https://huggingface.co/datasets
curl -L -o extra_data/imagenetv2-mf.tar  $HF/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz   # 이름은 .gz지만 실제로는 tar
curl -L -o extra_data/pets_test.parquet  $HF/timm/oxford-iiit-pet/resolve/main/data/test-00000-of-00001.parquet
for i in 0 1 2 3 4 5; do curl -L -o extra_data/flowers/test-$i.parquet $HF/dpdl-benchmark/oxford_flowers102/resolve/main/data/test-0000$i-of-00006.parquet; done
for i in 0 1;         do curl -L -o extra_data/cars/test-$i.parquet    $HF/tanganke/stanford_cars/resolve/main/data/test-0000$i-of-00002.parquet; done
for i in 0 1 2;       do curl -L -o extra_data/food/test-$i.parquet    $HF/ethz/food101/resolve/main/data/validation-0000$i-of-00003.parquet; done   # food101의 validation = 공식 test
for i in 0 1;         do curl -L -o extra_data/caltech/test-$i.parquet $HF/dpdl-benchmark/caltech101/resolve/main/data/test-0000$i-of-00002.parquet; done
```

## 실행 순서

```bash
# 1) 원본 베이스라인 (PyTorch)
python baseline_rn50.py --weights weights/RN50.pt --threads 2 --save-embeddings

# 2) ONNX 변환 + AttnPool 최적화본 + 동일성 확인
python export_onnx.py --weights weights/RN50.pt
python export_onnx.py --weights weights/RN50.pt --attnpool-q1 --image-only --verify

# 3) 개발용(dev) 캐시 + 보정 후보 (train 분할에서만, 없는 파일은 자동으로 받았다가 지움)
python datasets.py dev all

# 4) 양자화 예: 현재 기준 후보 ④ (stem·layer1 FP32, layer2~AttnPool INT8)
AP=/attnpool/MatMul,/attnpool/MatMul_1,/attnpool/MatMul_2,/attnpool/Gemm
python quantize_onnx.py --enc image --variant _q1 --calib-source highres --only /layer2,/layer3,/layer4 --only-exact $AP --tag c4
# → onnx/rn50_image_int8_q1_minmax_highres256_s0_c4.onnx

# 5) 정확도·유사도 비교
python evaluate.py --data cifar_A pets_dev flowers_dev cars_dev food_dev caltech_dev --models fp32 rn50_image_int8_q1_minmax_highres256_s0_c4.onnx

# 6) 속도 비교 (다른 작업과 동시에 돌리지 말 것)
python bench_speed.py --models fp32=rn50_image_fp32_q1.onnx c4=rn50_image_int8_q1_minmax_highres256_s0_c4.onnx

# 7) 최종 보고 (설정이 모두 정해진 뒤 1회): 테스트 전체, fake-quant 실행
for d in cifar_test imagenetv2 pets flowers cars food caltech; do
  python datasets.py test $d --models fp32 rn50_image_int8_q1_minmax_highres256_s0_c4.onnx; done
```

> 2026-10-07: 예전 `rn50_quant/` 코드는 보정 버그(256장 중 16장만 사용)가 있어 쓰지 않는다. 자세한 내용은 `CODE_REVIEW.md` 2차 검토.

## 지금까지 실험의 양자화 명령 모음

| 실험 | `quantize_onnx.py --enc image --variant _q1` 뒤에 붙일 옵션 |
|---|---|
| INT8 전체 | (없음) `--tag static_minmax_n256_q1` |
| 실험 D (attention 계산만 FP32) | `--exclude attnpool --keep $AP --tag D_q1` |
| 단계 하나만 INT8 | `--only /layer3 --tag only-layer3_q1` (stem은 `--only /conv1,/conv2,/conv3,/act1,/act2,/act3,/avgpool`) |
| stem + 단계 하나 | `--only /conv1,/conv2,/conv3,/act1,/act2,/act3,/avgpool,/layer3 --tag stem+layer3_q1` |
| 누적 ②~⑤ | `--only /layer4 --only-exact $AP` → `/layer3,/layer4` → `/layer2,...` → `/layer1,...` |
| stem 내부 15조합 | stem만 INT8 모델을 만든 뒤 `python make_stem_combos.py` |

`--variant` 없이 실행하면 AttnPool 최적화 전 그래프를 써요(초기 실험 A~D).
