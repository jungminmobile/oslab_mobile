# CLIP-RN50 양자화 실험 코드 (정리본)

2026 모바일공학 공모전 트랙 2(자유주제). CLIP-RN50을 Raspberry Pi 5 + Hailo-10H에 올리기 위한 예선용 데스크톱 실험 코드예요.
모든 파일 맨 위에 **목적과 설명**을 주석으로 달았어요. 현재 실험 계획은 저장소 루트의 `PRD.md`, 코드 검토 기록은 `docs/CODE_REVIEW.md`에 있어요.
이전 실험 전용 스크립트(night_run, make_stem_combos, clip_weights)는 `archive/2026-10-07_night/code/`로 옮겼어요.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `common.py` | 공통 설정: 경로, 전처리, 토크나이저, 데이터 구간 규칙, ONNX 세션, 임베딩·분류기·지표 |
| `baseline_rn50.py` | PyTorch 원본 FP32 베이스라인 (CIFAR-100 test 정확도, 크기, 지연, 메모리) |
| `export_onnx.py` | PyTorch → ONNX 변환. `--attnpool-q1`로 AttnPool 최적화본 생성, `--verify`로 동일성 확인 |
| `quantize_onnx.py` | ONNX Runtime 정적 INT8 양자화. 노드 선택(`--exclude/--only`), 보정 데이터 선택(`--calib-source`) |
| `datasets.py` | dev(설정 선택용, train 분할) 캐시·보정 후보 생성 / test 전체 최종 평가 |
| `evaluate.py` | 여러 모델 × 여러 데이터셋의 top-1/top-5/원본과의 유사도 |
| `bench_speed.py` | 속도 반복 측정 (번갈아 측정, 평균·흔들림) |
| `summarize.py` | evaluate.jsonl → 모델별 dev 정확도·FP32 대비 변화·유사도 표 |
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

## 최종 후보 B2 재현 (2026-10-20 기준)

B2 = stem·layer1.0은 FP32, 나머지(layer1.1~layer4, AttnPool 투영)는 INT8. Percentile 99.999, 고화질 보정 256장, 시드 0.
수치 요약: `../results/final_compare/RN50_제출후보_요약_2026-10-20.md`.

```bash
# 0) 위 "데이터 받기"와 아래 "실행 순서" 1)~3)을 먼저 한다 (가중치, ONNX q1 변환, dev 캐시·보정 후보)
#    고화질 파일 받기가 끊기면: python fetch_highres.py
# 1) B2 양자화
AP=/attnpool/MatMul,/attnpool/MatMul_1,/attnpool/MatMul_2,/attnpool/Gemm
REST=/layer1/layer1.1/,/layer1/layer1.2/,/layer2/layer2.0/,/layer2/layer2.1/,/layer2/layer2.2/,/layer2/layer2.3/,/layer3/layer3.0/,/layer3/layer3.1/,/layer3/layer3.2/,/layer3/layer3.3/,/layer3/layer3.4/,/layer3/layer3.5/,/layer4/layer4.0/,/layer4/layer4.1/,/layer4/layer4.2/
python quantize_onnx.py --enc image --variant _q1 --calib-source highres --calib-n 256 --seed 0 --method percentile --only $REST --only-exact $AP --tag B2
# → onnx/rn50_image_int8_q1_percentile99999_highres256_s0_B2.onnx
#   정민 노트북 sha256 6870979…dd3f19cd. CPU가 다르면 sha가 다를 수 있다 (정확도로 확인)
B2=rn50_image_int8_q1_percentile99999_highres256_s0_B2
HR="pets_dev flowers_dev cars_dev food_dev caltech_dev"
# 2) 정확도 (정수 커널) + FP32 대비 짝 부트스트랩
python evaluate.py --exec int --threads 4 --save-pred --data cifar_A $HR --models fp32 $B2.onnx
python bootstrap.py --a fp32 --b $B2 --data $HR
# 3) 속도·메모리 (다른 작업과 동시에 돌리지 말 것. 충전 중. 3번 돌려 중앙값)
python bench_speed.py --threads 1 2 4 --n 200 --repeats 3 --warmup 20 --models fp32=rn50_image_fp32_q1.onnx B2=$B2.onnx
python bench_mem.py --threads 4 --models fp32=rn50_image_fp32_q1.onnx B2=$B2.onnx
```

정민 노트북(Core Ultra 7 258V, 10/20) 결과는 다음과 같다.

| 지표 | 결과 |
|---|---|
| 고화질 5종 top-1 | 70.08 (FP32 71.51 대비 −1.43 %p [−2.18, −0.74]) |
| cos | 0.9606 |
| CIFAR A | 40.3 (FP32 40.7) |
| 4T 지연 | 13.82 ms (FP32 30.77 ms) |
| peak RSS | 119.4 MiB (FP32 325.5 MiB) |

Windows에서는 `run_*.bat`처럼 `chcp 65001`과 `PYTHONIOENCODING=utf-8`을 설정한다. 경로에 한글이 있으면 ORT 양자화가 실패할 수 있다 (`../docs/실험_유의사항.md`).

## 실행 순서

```bash
# 1) 원본 베이스라인 (PyTorch)
python baseline_rn50.py --weights weights/RN50.pt --threads 2 --save-embeddings

# 2) ONNX 변환 + AttnPool 최적화본 + 동일성 확인
python export_onnx.py --weights weights/RN50.pt
python export_onnx.py --weights weights/RN50.pt --attnpool-q1 --image-only --verify

# 3) 개발용(dev) 캐시 + 보정 후보 (train 분할에서만, 없는 파일은 자동으로 받았다가 지움)
python datasets.py dev all

# 4) 양자화 예 (이전 후보 c4, MinMax). 최종 후보 B2는 위 "최종 후보 B2 재현" 절
AP=/attnpool/MatMul,/attnpool/MatMul_1,/attnpool/MatMul_2,/attnpool/Gemm
python quantize_onnx.py --enc image --variant _q1 --calib-source highres --only /layer2,/layer3,/layer4 --only-exact $AP --tag c4
# → onnx/rn50_image_int8_q1_minmax_highres256_s0_c4.onnx

# 5) 정확도·유사도 비교
python evaluate.py --data cifar_A pets_dev flowers_dev cars_dev food_dev caltech_dev --models fp32 rn50_image_int8_q1_minmax_highres256_s0_c4.onnx

# 6) 속도 비교 (다른 작업과 동시에 돌리지 말 것)
# 감소율 기준(Baseline)은 q1 적용 전 원본 fp32다. q1은 따로 한 줄로 보고한다 (PRD 5절)
python bench_speed.py --models fp32=rn50_image_fp32.onnx fp32_q1=rn50_image_fp32_q1.onnx c4=rn50_image_int8_q1_minmax_highres256_s0_c4.onnx

# 7) 최종 보고 (설정이 모두 정해진 뒤 1회): 테스트 전체
#    ※ PRD E0에서 기본 실행을 정수 커널로 바꿀 예정. 보고 수치는 정수 커널만 사용
for d in cifar_test imagenetv2 pets flowers cars food caltech; do
  python datasets.py test $d --models fp32 rn50_image_int8_q1_minmax_highres256_s0_c4.onnx; done
```

> 2026-10-07: 예전 `rn50_quant/` 코드는 보정 버그(256장 중 16장만 사용)가 있어 쓰지 않는다. 자세한 내용은 `docs/CODE_REVIEW.md` 2차 검토.

## 양자화 명령 모음 (이전 실험에서 쓰던 구성, 참고)

| 실험 | `quantize_onnx.py --enc image --variant _q1` 뒤에 붙일 옵션 |
|---|---|
| INT8 전체 | (없음) `--tag static_minmax_n256_q1` |
| 실험 D (attention 계산만 FP32) | `--exclude attnpool --keep $AP --tag D_q1` |
| 단계 하나만 INT8 | `--only /layer3 --tag only-layer3_q1` (stem은 `--only /conv1,/conv2,/conv3,/act1,/act2,/act3,/avgpool`) |
| stem + 단계 하나 | `--only /conv1,/conv2,/conv3,/act1,/act2,/act3,/avgpool,/layer3 --tag stem+layer3_q1` |
| 누적 ②~⑤ | `--only /layer4 --only-exact $AP` → `/layer3,/layer4` → `/layer2,...` → `/layer1,...` |

`--variant` 없이 실행하면 AttnPool 최적화 전 그래프를 써요(초기 실험 A~D).
