# PRD — CLIP-RN50 이미지 인코더 양자화 (예선 메인 실험)

| 항목 | 내용 |
|---|---|
| 작성 | 2026-10-08, 정민 (CNN 담당) |
| 상태 | **현재 메인 계획**. 이전 실험(10/4~10/7)은 `archive/`에 참고용으로만 보관 |
| 대회 | 2026 모바일공학 하반기 공모전 트랙 2, 자유주제 |
| 마감 | 예선 보고서 11/1(일) 23:59 → 본선 11/6~11/30, 최종 발표 12/4 |
| 팀 분담 | 정민: CNN 계열(CLIP-RN50, RN50x4) · 준혁: 트랜스포머 계열(SigLIP2, 안 되면 CLIP ViT) |

---

## 1. 실험 목표

> **RN50 또는 RN50x4의 이미지 인코더를 8비트 중심으로 양자화해서 정확도·속도·용량·메모리의 트레이드오프 곡선(파레토)을 만든다. 최종 지점은 실험 후 팀원과 상의해서 고른다.**

| 지표 | 측정 | 비고 |
|---|---|---|
| **성능 개선** | 지연(ms)·처리량, 모델 크기(MiB), 최대 메모리(RSS) — 자기 FP32(q1 적용 전 원본 export, 5절) 대비 감소율 | 예선 배점 **25점** |
| **정확도** | 고화질 5종(Pets·Flowers·Cars·Food·Caltech) 평균 top-1의 FP32 대비 변화. CIFAR-100·ImageNetV2는 별도 표기 | 예선 배점 **15점**. 선택은 dev, 보고는 test |
| **이식성 (필수)** | 같은 구성이 Hailo DFC로 Hailo-10H용 `.hef` 컴파일까지 성공 | 본선 전제 |
| **속도 하한 (필수)** | 측정한 **모든 스레드 수**에서 자기 FP32보다 느리지 않아야 한다. 한 조건에서라도 느리면 제외 | "비정상적으로 느리지 않을 것" |

**대회 배점** (최종안내 4.4절): 예선은 보고서·실험 설계 35, 기법 25, **성능 개선(속도·메모리·크기) 25**, **정확도 유지 15**점.
성능 개선 점수가 정확도 점수보다 크다. 그래서 정확도만 지키는 보수적 구성 하나로 정하지 않는다. 손실 폭별로 여러 후보를 만들어 두고 E6에서 고른다.

**참고 기준선** (결정 규칙이 아니라 표에 표시하는 선): 고화질 평균 손실 0.5%p / 1.0%p / 2.0%p 지점별 최선 구성.

---

## 2. 요구사항 (제약)

| # | 요구사항 | 근거 |
|---|---|---|
| R1 | 예선은 채점자가 **자기 로컬 PC에서 직접 실행**하므로 **ONNX 파일**로 제출한다(베이스라인 + 최적화 모델). | 최종안내 4.2·4.3 (재현 실패 시 감점·심사 제외) |
| R2 | 본선은 예선 모델을 **Hailo DFC로 Hailo-10H용 `.hef`**로 컴파일해 써야 한다. Hailo-8/8L용 `.hef`는 호환되지 않는다. | 최종안내 5.2·6 |
| R3 | 그래서 **예선 최적화와 DFC 컴파일 가능성을 동시에 확보**한다. NPU 실측은 본선 전에는 어렵다. 따라서 성능 판단은 **예선 로컬 PC 측정**을 우선한다. | 본선 기간 약 3주 (최종안내 6) |
| R4 | 정확도·속도·용량의 우선순위는 미리 정하지 않는다. 실험으로 트레이드오프 곡선을 만든 뒤 팀원과 상의해 고른다. 단, 추론 시간이 비정상적으로 느리면 안 된다. | 최종안내 4.4 배점, 1절 |
| R5 | **정수 커널만 사용한다.** 보고 수치는 ONNX Runtime 기본 설정(그래프 최적화 ON)으로 낸다. **fake-quant는 쓰지 않는다**(디버깅에만 허용). | 채점자는 기본 설정으로 실행 → 정수 커널 |
| R6 | **이미지 인코더만 양자화한다.** 텍스트 인코더는 FP32로 클래스 문장 좌표를 1회 계산해 저장한다. | 4.2절 |

---

## 3. 기술 스택

| 구분 | 선택 | 버전 | 근거 |
|---|---|---|---|
| 언어 | Python | 3.11 | 기존 코드·두 측정 PC 모두 3.11 |
| 원본 모델 | OpenAI CLIP RN50 / RN50x4 가중치 | sha256 고정 (`rn50/README.md`) | Radford et al., 2021 |
| 변환 | PyTorch → ONNX (`torch.onnx.export`, dynamo=False) | torch 2.14.0, onnx 1.23.1, **opset 17** | DFC는 float ONNX에서 시작 (Hailo Community) |
| 그래프 최적화 | AttnPool **q1**: query를 0번 칸만 계산. 출력 동일(최대 차이 1.8e-7) | — | 이전 실험에서 검증 |
| 예선 양자화 | ONNX Runtime 정적 PTQ, **QDQ 형식** | onnxruntime 1.25.0 | ORT 공식 문서: QDQ + S8S8가 기본, 정확도·성능 균형 |
| 가중치 | INT8 대칭, **채널별(per-channel)** | — | ORT 문서, Nagel et al. 2021, Wu et al. 2020: 채널별이 손실을 크게 줄임 |
| 활성값 | 8비트 텐서별. 형식은 **S8S8(QDQ)을 기본**으로 한다. 단 x86에서는 S8S8도 실행 때 U8S8로 바뀌어 두 형식의 계산이 같다(아래 주석). 비-VNNI 대비책은 E0에서 결정 | — | ORT 문서: VNNI 없는 AVX2·AVX512에서 U8S8 포화 가능. 문서의 대응책은 **reduce_range(가중치 7비트) 또는 U8U8**. 채널별 가중치는 이런 CPU에서 reduce_range가 대체로 필요 |
| 보정 방법 | MinMax / **Percentile** / **Entropy** 비교 (E2) | — | Wu et al. 2020, Migacz 2017(TensorRT Entropy), ORT 지원 3종. **주의(ORT 1.25.0 소스 확인)**: Percentile 보정기의 기본은 *대칭* 범위(−t~t)라 ReLU 뒤 활성값에서 눈금의 절반을 버린다. `CalibTensorRangeSymmetric=False`를 명시해야 MinMax·Entropy와 공정하다. 백분위 값(기본 99.999)은 `CalibPercentile`로 지정한다. 현재 `quantize_onnx.py`는 둘 다 넘기지 않는다 → E0 |
| 혼합 정밀도 | 민감한 층은 FP32로 남김 (ORT `nodes_to_exclude`) | — | Nagel 2021 디버깅 절차, Wu 2020 partial quantization |
| 실행 | ORT CPUExecutionProvider, `ORT_ENABLE_ALL`, 스레드 고정 | — | 정수 커널(QLinearConv 등)로 융합됨 |
| 본선 컴파일 | **Hailo Dataflow Compiler 5.x**, `--hw-arch hailo10h` | Model Zoo 공개 `.hef` = DFC 5.4.0 | Hailo Model Zoo HAILO10H 문서 |
| DFC 환경 | 정민 PC의 WSL2 Ubuntu 22.04 또는 Docker (**WSL·Docker는 설치 확인, DFC는 미설치**) | DFC 릴리스 노트의 Python·Ubuntu 버전으로 **별도 가상환경** (예선 코드의 Python 3.11과 분리) | DFC는 x86 리눅스 전용 |
| Hailo 양자화 | 8비트(a8_w8) 기본, 민감 층 16비트(a16_w16) | — | Hailo Quantization Scheme, Model Zoo RN50 alls |

> **S8S8 vs U8S8 (확인 완료, 2026-10-08)**
> - ORT는 x86에서 S8S8 QDQ 모델의 활성값을 U8로 바꿔 U8S8 커널로 실행한다. `QDQIsInt8Allowed()`가 ARM이 아니면 false이고, 그러면 `QDQS8ToU8Transformer`가 적용된다(ORT main 소스).
> - 우리 확인: ORT 1.25.0, x86(AVX512-VNNI), 합성 3층 conv 모델에서 **S8S8과 U8S8의 출력 최대 차이 0.0**(정수 커널·fake-quant 모두). 둘 다 `QLinearConv`로 융합됐다. 그래서 x86에서 S8S8로 바꿔도 VNNI 없는 CPU의 포화는 그대로다. 두 형식의 차이는 ARM CPU에서만 생긴다.
> - 현재 코드(`quantize_onnx.py`)는 `activation_type=QUInt8`(U8S8)이다. E0에서 `--act-type`을 추가하고 기본값을 S8S8로 바꾼다. 기본값이 바뀌므로 CLAUDE.md의 "옵션을 끄면 이전 main과 같아야 한다" 규칙의 예외이며, E0 PR에 명시한다.
> - `session.x64quantprecision=1`은 VNNI 없는 CPU에서만 U8U8 계산으로 바꾼다(우리 확인: VNNI CPU에서는 출력 차이 0.0, 비-VNNI 동작은 E0에서 확인). 채점자가 우리 실행 스크립트를 쓰는지, 세션 옵션을 넣어도 되는지 대회 규정을 확인한다.
> - `reduce_range=True`는 **가중치만** 7비트로 줄인다(확인: 정수 범위 −127~127 → −64~64). 활성값 눈금은 그대로다.

> **4비트는 8비트 실험(E0~E8a)이 끝나고 시간이 남을 때 시도한다 (E9).**
> - ORT 문서가 4비트를 지원한다고 밝힌 연산은 MatMul·Gather뿐이다. conv를 4비트 QDQ로 만들면 정수 커널로 융합되지 않는다. 이전 실험(archive R7)에서 FP32보다 29~139% **느렸다**(우리 측정).
> - 그래서 예선용 4비트는 **AttnPool 투영(q·k·v·c_proj, MatMul 계열)**에만 ORT 4비트(블록 단위 가중치) 경로를 시험한다. AttnPool 투영은 RN50 이미지 인코더 파라미터의 약 38%(약 1470만/3830만 개)라 용량 효과가 크다.
> - Hailo 4비트(a8_w4)는 DFC 에뮬레이터로 시험한다. Hailo 문서는 4비트 층이 있으면 **QFT + 가중치 잘라내기가 필수**라고 한다.

---

## 4. 설계 원칙

### 4-1. 이미지 인코더만 양자화하는 이유

| 항목 | 이미지만 양자화 | 텍스트도 양자화 |
|---|---|---|
| 실행 빈도 | 사진마다 | 클래스 목록이 정해질 때 1회 |
| 장당 속도 이득 | 있음 | **없음** |
| 정확도 | 오차가 사진 쪽에서만 생김 | 클래스 좌표에도 오차가 생김 |
| 기기에 올리는 크기 | 클래스 좌표 표만(클래스 수 × 임베딩 차원 × 4B. RN50은 1024차원, RN50x4는 640차원. 100클래스 RN50 ≈ **0.4 MB**, ImageNetV2 1000클래스 RN50 ≈ 4 MB) | 텍스트 인코더 INT8 ≈ 60 MB |

CLIP 제로샷 분류는 원래 클래스 임베딩을 미리 계산하는 구조다(Radford et al., 2021).
텍스트 인코더는 트랜스포머라 활성값 이상치가 커서 양자화가 까다롭다. Hailo도 텍스트 인코더에 16비트, 미세조정 같은 특수 설정을 쓴다.

### 4-2. 예선 ONNX ↔ 본선 DFC 구성 짝 맞추기

DFC는 이미 양자화된 ONNX를 받지 않는다. **float ONNX에서 다시 양자화**한다(Hailo Community).
그래서 "예선에서 찾은 층별 정밀도 구성"을 **DFC 모델 스크립트(`.alls`)로 똑같이 옮길 수 있어야** 한다.

| 예선 ONNX (ORT) | 본선 DFC (`.alls`) | 비고 |
|---|---|---|
| INT8 층 (가중치 S8 채널별, 활성값 8비트) | `quantization_param(..., precision_mode=a8_w8)` | 기본값 |
| FP32로 남긴 층 (예: stem, layer1) | `precision_mode=a16_w16` | NPU에는 FP32가 없음 → 16비트로 대응 |
| AttnPool attention 계산(점수·softmax·가중합) FP32 | 해당 matmul·softmax 층 a16_w16 | 층 이름은 파싱 후 매핑 |
| 보정 사진 (같은 구간·같은 시드) | `calibset_size` + 같은 사진 | Hailo 권장 32~96장. ORT 보정 장수 비교(E3)와 별개로, Hailo 보정 장수는 E8a에서 비교한다 |
| E5·E5+ (CLE·Bias Correction·AdaRound) | DFC 자체 equalization·bias correction·AdaRound 옵션(이름은 DFC 문서로 확인) | ORT 눈금에 맞춘 Bias Correction·AdaRound 결과는 DFC가 다른 눈금으로 다시 양자화하므로 효과가 사라지거나 나빠질 수 있다. DFC에는 이 둘을 적용하기 전 float ONNX를 넣는다. CLE는 FP32 출력이 같으므로 적용한 ONNX를 넣어도 된다 |

**함정 1**: ONNX 노드 이름(`/layer3/...`)과 DFC 층 이름(`conv23` 등)이 다르다. E1에서 **노드 → DFC 층 이름 매핑표**를 만든다.
**함정 2 (가정)**: ORT와 DFC는 양자화 방식(스케일 단위, 기본 equalization 등)이 다르다. 그래서 ORT에서 INT8로 안전했던 블록이 Hailo a8_w8에서도 안전하다는 보장은 없다. E8a에서 **층별 에뮬레이터 오차(cos/SNR)**를 비교해 이 가정을 확인한다.

### 4-3. 정수 커널만 쓰는 측정 규칙

- 모든 정확도·속도·메모리는 `ORT_ENABLE_ALL`(기본)로 측정한다. 결과에 **CPU 모델명·VNNI 지원 여부·ORT 버전·스레드 수**를 자동으로 기록한다.
- 두 PC에서 교차 측정해 재현성을 보인다.

| PC | CPU | 8비트 가속 | 용도 |
|---|---|---|---|
| 클라우드 | Intel Xeon @ 2.10GHz, 2코어 | AVX512-VNNI | 대량 실험 |
| 정민 노트북 | Intel Core Ultra 7 258V, 8코어, 32GB | AVX-VNNI | 교차 검증·최종 측정, DFC |

- 채점자 PC는 VNNI를 지원하지 않을 수 있다. 이 경우 8비트 연산에서 포화가 생길 수 있다(ORT 문서).
  위 두 PC는 모두 VNNI가 있어서 **포화를 재현할 수 없다**.
  E0에서 다음 순서로 대응한다.
  ① VNNI 없는 PC를 하나 확보한다(준혁 PC·학과 PC의 CPU 플래그 확인). VNNI가 없는 CPU는 흔하다(AMD Ryzen 5000/Zen 3 이하, 인텔 데스크톱 10세대 이하).
  ② 문서의 대응책인 `reduce_range=True`와 U8U8 모델, 세션 옵션 `session.x64quantprecision=1`의 정확도·속도를 VNNI PC에서 미리 측정한다. S8S8과 U8S8은 x86에서 같으므로(3절 주석) 별도 대책이 아니다. E0에서 c3·c4로 같은지 다시 확인한다.
  ③ 비-VNNI PC를 확보하지 못하면, README에 "VNNI 없는 CPU용 모델(reduce_range)"을 따로 안내한다.
  ④ 확보하지 못하면 Intel SDE로 구형 CPU를 흉내 내서 포화를 재현한다(가능한지 확인 필요).
- **지연 측정 조건**: 정민 노트북은 스레드 1·2·4, 클라우드는 1·2로 잰다(258V는 성능 코어 4개 + 저전력 코어 4개라 "스레드 전체"는 흔들림이 크다). **주 측정 조건은 정민 노트북 스레드 4**(E0에서 확정)이고 나머지는 참고로 적는다. 자동 기록 항목에 OS·전원 연결 여부·전원 모드를 포함한다.

---

## 5. 평가 프로토콜 (팀 공통 · 준혁과 동일하게 맞춤)

| 구간 | 데이터 | 용도 |
|---|---|---|
| 보정 (calib) | CIFAR train[0:45000] 또는 고화질 5종 train(dev 제외 해시 구간). **E3 전까지의 기본값은 256장, 시드 0, 섞음**. E3 이후에는 `configs/<모델>_best.json`을 따른다 | 눈금 정하기 |
| dev | **6종**: CIFAR train[45000:46000](A), 고화질 5종 train에서 SHA-1 해시로 약 1000장씩. CIFAR B([46000:47000])는 최종 후보 확인에만 | 구성 비교·선택 |
| test | CIFAR-100, **ImageNetV2**(matched-frequency), Oxford Pets, Flowers-102, Stanford Cars, Food-101, Caltech-101 **전체** | **최종 1회만** (E7) |

| 지표 | 정의 | 비고 |
|---|---|---|
| top-1 / top-5 | 제로샷 정확도 (%) | 프롬프트 앙상블(CIFAR 18, ImageNetV2 7, Cars 8, Caltech 34, 그 외 1개) |
| Δtop-1 | 같은 모델 FP32 대비 (%p) | **같은 사진끼리 짝지어 비교**한다. 부트스트랩(1000회)은 데이터셋마다 따로 사진을 뽑아(층화) 95% 신뢰구간을 낸다. "고화질 평균"은 5종의 단순 평균이다. 구간이 0을 포함하면 "유의한 차이 없음"으로 쓰고 "손실 없음"이라고 쓰지 않는다. 기준선(0.5·1.0·2.0%p) 분류는 점추정으로 하고 구간을 함께 적는다. 두 양자화 구성끼리 비교할 때는 두 구성의 차이를 직접 짝 부트스트랩하고, 그 구간이 0을 포함하면 동률로 보고 cos로 고른다. (고화질 dev 약 5000장에서 FP32와 답이 갈리는 사진이 2~5%면 구간은 대략 ±0.4~0.6%p라 0.5%p 기준선과 크기가 비슷하다. E0 ④에서 실제 폭을 잰다) |
| cos | FP32 이미지 임베딩과의 평균 코사인 유사도 | 정답 없이도 재는 보조 지표. 층 민감도 분석의 주 지표 |
| 용량 | ONNX 파일 크기 (MiB), FP32 대비 감소율 | + 클래스 임베딩 표 크기 별도 표기 |
| 지연 | 장당 ms, 배치 1, 스레드: 노트북 1·2·4 / 클라우드 1·2 | 워밍업 20회, 200장 × 3회 번갈아 측정, 평균 ± 표준편차. 주 조건 = 노트북 스레드 4(4-3절). CPU 모델명·OS·전원 상태 자동 기록 |
| 처리량 | 장/초, 배치 8 | 보조 |
| 메모리 | 추론 중 최대 RSS (MiB) | 안내서 4.2 "메모리 사용량" 요구. 모델마다 새 프로세스에서 재고(`baseline_rn50.py` 방식), 세션 생성 때 최대값과 추론 중 최대값을 따로 기록 |

Caltech-101은 BACKGROUND_Google, Faces, Faces_easy 클래스를 포함해 채점한다. 이름은 각각 background, off-center face, centered face로 바꾼다.
CIFAR-100은 stem 변화에 과민하다(이전 실험에서 31~46%로 흔들림). 그래서 **구성 선택은 고화질 dev 평균과 cos를 주 지표로** 삼는다. CIFAR는 Hailo 공식 수치 비교용으로 쓴다.

**자기 FP32 (기준 모델)**: 모든 감소율·Δ의 기준은 q1을 적용하기 전의 원본 FP32 export(`<모델>_image_fp32.onnx`)다. q1은 출력이 같은 그래프 최적화(계산량 6.12→5.80 GMAC, −5%)라 따로 한 줄로 보고한다(원본 → q1 → q1+INT8). 이전 실험의 FP32 수치(146.1 MiB, 54.5 ms)는 q1 적용본(`rn50_image_fp32_q1.onnx`)이었다.

**test 재실행 규칙**: E7 뒤에 코드 버그가 발견되면 구성은 바꾸지 않고 코드만 고친 뒤 모든 후보를 다시 잰다. 이유와 고치기 전후 수치를 `results/E7_test/README.md`에 남긴다.

---

## 6. 실험 계획

시간은 이전 실험 기준 추정이다(RN50 모델 1개 × dev 6종 ≈ 5분, RN50x4 ≈ 11분, 클라우드 2코어).

### 6-0. 구성 이름 (범례)

| ID | INT8로 바꾸는 부분 | FP32로 남기는 부분 |
|---|---|---|
| c1 | AttnPool 투영(q·k·v·c_proj) | 나머지 전부 |
| c2 | + layer4 | stem, layer1~3 |
| c3 | + layer3 | stem, layer1~2 |
| c4 | + layer2 | stem, layer1 |
| c5 | + layer1 | stem |
| Fon | 전부(stem 포함) | AttnPool attention 계산(점수·softmax·가중합)만 |

- 모든 구성에서 AttnPool의 attention 계산(점수·softmax·가중합)은 FP32로 둔다.
- ONNX 파일 이름은 코드 규칙을 따른다: `<모델>_image_int8_q1_<방법>_<출처><장수>_s<시드>_<구성ID>.onnx`. 예: `rn50_image_int8_q1_minmax_highres256_s0_c4.onnx`. Percentile은 백분위 값도 이름에 넣고(`percentile99999`), 비-VNNI 대비 변형은 태그 뒤에 `_u8u8`·`_rr`를 붙인다(E0).
- E4의 블록 단위 구성은 `b_<FP32로 남긴 블록 목록>`으로 표기한다.

### 6-1. 실험 표

| ID | 실험 | 방법 | 근거 | 판단 기준 / 산출물 | 예상 |
|---|---|---|---|---|---|
| **E0** | 측정 기반 정비 | ① `evaluate.py`·`datasets.py test` 기본 실행을 정수 커널로(`exec` 기본 `int`) ② CPU 모델명(리눅스는 `/proc/cpuinfo`)·VNNI·ORT 버전·OS·전원 상태 자동 기록. `bench_speed.py`는 워밍업 20회, 스레드 수 반복 측정 ③ ONNX 모델용 최대 RSS 측정 추가(모델마다 새 프로세스) ④ 짝 부트스트랩 신뢰구간 계산 추가(데이터셋별 층화). c3·c4로 실제 구간 폭을 재고, 기준선 간격이 구간보다 좁으면 기준선을 조정한다 ⑤ 협업 기반: 결과 jsonl에 git 커밋 해시·`dirty`·사용 옵션 기록, `rn50/smoke_test.py`(5분 이내), `configs/rn50_best.json` 첫 버전 ⑥ `quantize_onnx.py` 옵션 추가: `--act-type {s8,u8}`(기본 s8), `--weight-type`, `--reduce-range`, `--percentile`, 대칭 범위 해제(`CalibTensorRangeSymmetric`). 파일 이름에 모두 반영 ⑦ 보정 후보 풀 확대: `datasets.py`의 `CAL_N` 64→256(5종 합계 약 1280장), `datasets.py dev all`로 보정 후보 재생성(dev 사진은 그대로) ⑧ 비-VNNI 대비: c3·c4를 S8S8 / U8S8 / reduce_range / U8U8 / `session.x64quantprecision=1`로 비교해 정확도·속도 확인(S8S8=U8S8인지 포함), 비-VNNI PC 확보 시도 ⑨ 결과 저장: 실행 때 `RN50_RESULTS=results/<실험 폴더>` 지정 | ORT 문서 (포화·reduce_range), ORT 소스 (S8→U8 변환) | 기본 형식 확정, 비-VNNI용 대체 모델 결정 / 공통 측정 코드. **E0 PR에서 활성값 기본이 U8→S8로 바뀐다고 명시** | 1~1.5일 |
| **E1** | **DFC 호환성 관문** | RN50·RN50x4 q1 float ONNX를 `hailo parser`(hailo10h) → optimize(기본 8비트, 64장) → compile `.hef`. 실패 연산 확인, ONNX 노드 ↔ DFC 층 이름 매핑표 작성. **추가: 대표 혼합 구성 1개(c4 형태: stem·layer1·attention 계산은 `a16_w16`, 나머지는 `a8_w8`)를 `.alls`로 컴파일하고 에뮬레이터로 dev 1종(예: pets) 정확도를 잰다.** DFC 프로파일러(`hailo profiler`) 사용 가능 여부도 확인 | 최종안내 6, Hailo Community, Hailo 공식 RN50 설정(`hailo/README.md`: 대부분 a16_w16, optimization_level=0) | **모델별로 판단**: 실패한 모델은 후보에서 빼고 나머지로 진행한다. E2·E3는 E1과 상관없이 진행할 수 있고, E4부터 E1 결과가 필요하다 / `hailo/` 스크립트, 매핑표 | 1.5~2.5일 (DFC 설치 포함) |
| **E2** | 보정 방법 | MinMax vs Percentile(99.99·99.999, **비대칭 범위**) vs Entropy × 구성 c3·c4·c5·Fon (RN50). **보정 출처·장수·시드는 highres 256 s0으로 고정**(이전 실험과 같음) | Wu 2020, Migacz 2017. 이전 결론: MinMax가 이상치에 끌림 | 고화질 dev Δtop-1(신뢰구간)·cos가 가장 좋은 방법 / 표 | 1일 |
| **E3** | 보정 데이터 | 출처(cifar·highres·mix) × 장수(64·256·512), E2 최선 방법으로. 시드 0·1·2. 최선 출처가 E2의 고정값(highres)과 다르면 E2 상위 2개 방법만 다시 확인한다. RN50x4는 c5·Fon에서 E2·E3 상위 2개 조건만 확인한다 | Hailo 권장 32~96장, 이전: CIFAR 보정이 cos 높음 | 시드 간 흔들림이 출처·장수 차이보다 작은지 / 표. 보정 후보 풀이 작으면 시드끼리 사진이 많이 겹쳐 흔들림이 작게 나오므로 E0 ⑦(CAL_N 확대) 뒤에 실행 | 1일 |
| **E4** | 블록 단위 혼합 정밀도 | 단계 단위 선택을 **블록 단위**로 세분화. 블록 수: RN50 16개(3+4+6+3), RN50x4 26개(4+6+10+6), 각각 + stem + AttnPool. ① 블록 하나만 INT8 → 민감도(cos) 순위와 블록별 크기 감소(MiB, 계산), 지연 감소(ms, 실측) (n개 모델) ② Fon(attention 계산만 FP32)에서 시작해 **"민감도 ÷ 비용" 순서**로 블록을 하나씩 FP32로 되돌린다(재평가 없는 순위 기반). 크기 곡선의 비용은 블록 크기(MiB), 지연 곡선의 비용은 ①의 블록별 지연 감소(ms)이며 **크기용·지연용 곡선을 따로** 만든다. ②에서 만든 모델은 모두 지연을 실측한다 | Nagel 2021 디버깅, Wu 2020 partial quantization. 이전: layer3만 INT8 → 크기 −20 MiB·지연 −11 ms, layer4만 → −43 MiB·−7 ms (크기와 속도에 중요한 블록이 다름) | **용량·지연-정확도 파레토 곡선** 전체, 참고 기준선(0.5·1.0·2.0%p)별 최선 구성 | RN50 약 3시간 + 지연 측정, RN50x4 약 10시간 + 지연 측정 (≈1.5~2일) |
| **E5a** | 가중치 보정 ① CLE | **PyTorch에서 export 전에** Cross-Layer Equalization 적용(병목 블록 안 conv1→conv2→conv3 쌍만. 잔차 합 너머로는 스케일하지 않음). **E4 전에** Fon·c5로 효과를 확인하고, 켜기로 하면 E4를 CLE 모델로 진행한다 | Nagel et al. 2019 (DFQ). CLE는 FP32 출력을 바꾸지 않지만 블록 민감도를 바꾸므로 E4 순위를 뒤집을 수 있다 | 개선 신뢰구간이 0을 넘으면 채택. 채널별 가중치에서는 효과가 작을 수 있음(DFQ의 이득은 주로 텐서별). DFC는 자체 equalization이 기본이라, 효과 없음도 결과로 기록 | 0.5일 |
| **E5b** | 가중치 보정 ② Bias Correction | **E4 뒤**, E4 최선 구성에 사진 기반(empirical) Bias Correction을 적용해 재양자화. 쓰는 사진은 **보정 구간에서만** 뽑는다(dev 사용 금지) | Nagel et al. 2019 | 개선 신뢰구간이 0을 넘으면 채택. DFC 이식 주의는 4-2절 | 0.5일 |
| **E5+** | (여유 시) AdaRound | 블록별 반올림 학습 → 가중치를 눈금에 맞춰 저장, ORT 스케일 고정. 학습 사진은 보정 구간에서만 뽑는다 | Nagel et al. 2020 | 선택 실험. DFC 이식 주의는 4-2절 | 1.5일 |
| **E8a** | Hailo 사전 검증 (후보 전체) | E4·E5의 파레토 후보들(기준선별 최선)을 `.alls`로 옮겨 컴파일. **DFC 에뮬레이터**로 dev 정확도 측정: 양자화 에뮬레이션(`InferenceContext.SDK_QUANTIZED`)과 float 최적화 모델(`SDK_FP_OPTIMIZED`)을 같이 재서 양자화 오차와 최적화 오차를 분리. 층별 cos를 ORT와 비교. `a16_w16` 비중을 늘린 후보 1~2개도 함께 컴파일해 정확도를 비교 | 최종안내 5.2 (서버 vs 엣지 비교) | 컴파일 성공 여부 + 에뮬레이터 Δtop-1 / E6 입력 | 1일 |
| **E6** | 최종 모델 선택 | RN50 최선 vs RN50x4 최선. 8절 규칙 | Hailo Model Zoo 수치 | 최종 후보 1~2개 | 0.5일 |
| **E7** | 최종 test (1회) | 후보 + 각 FP32 베이스라인. test 7종 전체 + CIFAR B, 정수 커널, **두 PC** | 최종안내 4.2 | 보고서 표 원본 | RN50 3시간 / RN50x4 8시간 |
| **E8b** | Hailo 최종 패키지 | 선택 모델의 `.hef`, `.alls`, 컴파일 로그, 에뮬레이터 test 정확도 | 최종안내 5.3 | 본선 이식 패키지 / 보고서 "이식 계획" | 0.5일 |
| **E9a** | (시간 여유 시) ORT 4비트 AttnPool | 8비트 최선 구성에서 AttnPool 투영만 4비트 블록 가중치(ORT `MatMulNBits`, 블록 32·128)로. c_proj Gemm은 MatMul+Add로 변환 후 적용. 잘라내기(archive `clip_weights.py`) 유무 비교 | ORT 문서 (MatMul 4비트), archive R6b | 용량 감소 대비 정확도·**지연**. FP32보다 느리면 탈락 | 1일 |
| **E9b** | (시간 여유 시) Hailo 4비트 | DFC `a8_w4`를 큰 층(AttnPool·layer4)에 적용 + 가중치 잘라내기 + QFT(가능한 범위). 에뮬레이터 정확도 | Hailo Quantization Scheme | 본선용 크기 추가 감소 후보 | 1~2일 |

**순서와 관문**: E0 → **E1(관문, 혼합 구성 확인 포함)** → E2 → E3 → E5a(CLE) → E4 → E5b(Bias Correction) → (여유 시 E5+) → **E8a** → (여유 시 E9a·E9b) → E6 → E7 → E8b.
4비트(E9)는 8비트 결과가 모두 나온 뒤에만 시작한다. **10/23까지 끝나지 않으면 중단하고 보고서에 미완 실험으로 기록한다**(H2는 Hailo 작업이라 E9a를 넘기는 곳이 아니다).
E1이 실패하면(파싱 불가 연산) 그래프를 수정(q1, 연산 치환)한 뒤에 진행한다.
DFC 설치가 늦어지면 E1과 E2·E3를 병행한다(E2·E3는 DFC 없이 가능).
여러 팀원이 같은 모델에 기법을 추가할 때는 `CLAUDE.md` "협업 규칙"을 따른다. 기법은 옵션으로 추가하고, 조합(A+B)까지 측정한 뒤에만 `configs/<모델>_best.json`을 바꾼다.

### 본선 단계 후보 (예선 이후, 참고)

| ID | 내용 | 근거 |
|---|---|---|
| H1 | Pi 5 + Hailo-10H 실측: 같은 dev/test, FPS·지연·메모리, 서버 결과와 차이 분석 | 최종안내 5.2 |
| H2 | E9b를 이어서: 4비트 혼합(a8_w4) + QFT + 잘라내기로 크기 추가 감소. QFT용 train 사진 2048장(dev 제외)은 보정 후보 풀을 `CAL_N` 450 이상으로 더 키워 미리 확보한다(구간이 넓어지는 방식이라 기존 후보는 그대로 포함). 노트북에 NVIDIA GPU가 없어 현실적으로 어려울 수 있다 | Hailo Quantization Scheme (QFT 기본 2048장) |
| H3 | Hailo 공식 설정 비교: 공식 RN50은 대부분 16비트, CIFAR 42.1→41.2, 142 FPS. 우리 8비트 위주 구성이 더 빠르면서 정확도를 지키는지 | Hailo Model Zoo, `hailo/README.md` |

---

## 7. 이전 실험에서 가져오는 출발점 (`archive/2026-10-07_night/`, dev·참고용)

| 구성 (RN50) | 용량 | 장당 ms (2코어) | 고화질 Δtop-1 | 비고 |
|---|---|---|---|---|
| FP32 | 146.1 MiB | 54.5 | 0 | |
| c3 (stem·L1·L2 FP32) | 41.5 (−72%) | 33.4 (−39%) | −0.72 | CIFAR A 포함 6종 평균은 약 −1.08 |
| c4 (stem·L1 FP32) | 38.1 (−74%) | 25.5 (−53%) | −1.4 ~ −1.6 | 보정 개선 시 목표 근접 |
| RN50x4 c5 (stem만 FP32, CIFAR 보정) | 84.9 | 70.1 | −0.54 (자기 FP32 대비) | RN50 FP32보다 정확도 +3~11%p |

- **주의**: 위 정확도 일부는 fake-quant 값이 섞여 있다. 4비트 결과는 fake 계산 구조다. 보정은 MinMax뿐이다. **E0 이후 정수 커널로 다시 측정한 값만 보고서에 쓴다.**
- FP32 146.1 MiB·54.5 ms는 q1 적용본이다. 새 Baseline(원본 FP32 export)은 계산량이 5% 많아 더 느릴 것이므로 다시 잰다(5절).
- 이어받는 결론: ① 보정 버그(16장만 사용) 수정 후 stem INT8 이득은 사라졌다 ② MinMax는 이상치에 끌린다 → E2 ③ 정수 커널과 fake-quant의 순위는 같았다.

---

## 8. 최종 모델 선택 (E6) — 팀 상의

1. **자동 제외**: E8a에서 `.hef` 컴파일에 실패한 구성, 측정한 모든 스레드 수에서 자기 FP32보다 느리지 않다는 조건을 만족하지 못한 구성.
2. **후보표 작성**: 남은 구성으로 다음 표를 만든다. 모델마다 참고 기준선(손실 0.5·1.0·2.0%p)별로 **"크기 최선"과 "지연 최선" 두 줄**을 만든다(같은 구성이면 한 줄). 지연은 주 측정 조건(노트북 스레드 4, 4-3절)의 값이고, 기준선 분류는 점추정으로 하되 95% 구간을 함께 적는다. "Hailo 예상 FPS"는 DFC 프로파일러 값이다(`a16_w16` 층이 NPU 속도에 주는 영향은 CPU 측정으로 알 수 없다).

   | 모델 | 기준선 | 유형(크기/지연) | 구성 | 크기 (감소율) | 지연 (감소율) | 메모리 | 고화질 Δtop-1 (95% 구간) | CIFAR·ImageNetV2 Δ | Hailo 에뮬레이터 Δ | Hailo 예상 FPS |
   |---|---|---|---|---|---|---|---|---|---|---|

3. **팀 상의**: 후보표를 보고 정민·준혁이 트레이드오프 지점을 고른다. 판단 재료는 다음과 같다.
   - 예선 배점은 성능 개선 25 > 정확도 15다.
   - 본선은 실측 성능 30, 정확도 20이다.
   - Hailo 공식 수치: RN50 142 FPS(CIFAR 41.2) vs RN50x4 62.6 FPS(CIFAR 50.9).
   - 트랜스포머 쪽(준혁)과 같은 기준선으로 비교할 수 있어야 한다.
4. 선택한 모델의 FP32가 보고서의 Baseline이 된다. 고른 이유는 보고서 "실험 설계" 절에 후보표와 함께 쓴다. **Baseline은 q1을 적용하기 전의 원본 FP32 export다.** q1은 출력이 같은 그래프 최적화 기법으로 따로 한 줄 보고한다(원본 → q1 → q1+INT8).

---

## 9. 일정 (10/8 → 11/1)

| 기간 | 작업 |
|---|---|
| 10/8 ~ 10/11 | E0(옵션 추가·CAL_N 확대·캐시 재생성), DFC 설치(Developer Zone 계정·다운로드), E1(혼합 구성 확인 포함) |
| 10/12 ~ 10/15 | E2, E3, E5a(CLE). E2·E3는 E1과 병행 가능 |
| 10/16 ~ 10/20 | E4, E5b (여유 시 E5+) |
| 10/21 ~ 10/23 | E8a, 후보표 → **팀 상의(E6)**. E9a·E9b는 여유가 있을 때만 하고 10/23까지 끝나지 않으면 중단(보고서에 미완 실험으로 기록) |
| 10/24 ~ 10/26 | E7 최종 test (두 PC), E8b |
| 10/27 ~ 11/1 | 보고서 작성, 재현 패키지 점검(새 환경에서 README대로 실행), 제출 |

---

## 10. 위험과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| DFC가 AttnPool 연산(Softmax·MatMul 모양 등)을 파싱하지 못함 | 본선 이식 불가 | E1을 가장 먼저 실행. q1 그래프·연산 치환. 최후에는 AttnPool을 CPU 후처리로 분리 |
| 채점자 PC에 VNNI가 없음 → 8비트 포화 | 보고 수치 재현 실패 | E0에서 reduce_range·U8U8 대체 모델 준비, 비-VNNI PC 확보 시도. README에 측정 CPU와 예상 범위 명시 |
| DFC 고급 최적화에 GPU 필요 | E8a·E8b 품질 제한 | 기본 최적화 수준으로 진행. 필요하면 GPU 서버 사용 검토. 노트북에 NVIDIA GPU가 없으므로 E9b·H2의 QFT는 현실적으로 어렵다 |
| Percentile·Entropy 보정의 메모리 사용량 큼 | 클라우드(7GB)에서 OOM | 보정 배치 축소, 노트북(32GB)에서 실행 |
| test 재사용 유혹 | 점수 부풀림 | test는 E7에서 1회만 실행. 실행 기록(log)을 커밋 |
| 채점 PC가 Apple Silicon(ARM)이다 | S8S8이 U8 변환 없이 ARM 커널로 실행되어 수치가 x86과 달라질 수 있다 | README에 측정 CPU를 명시한다. 가능하면 ARM 맥에서 한 번 확인한다 |
| Hailo에서 8비트 위주 구성의 정확도가 부족하다(공식 설정은 대부분 16비트) | 예선 구성을 그대로 옮기면 본선 정확도가 떨어진다 | E1에서 혼합 구성을 에뮬레이터로 미리 확인한다. E8a에 `a16_w16` 비중을 늘린 후보를 준비한다 |
| 본선 장비의 HailoRT 버전이 DFC 버전과 맞지 않는다 | `.hef`가 실행되지 않는다 | 주최측에 HailoRT 버전을 미리 묻고, 그에 맞는 DFC 버전을 쓴다 |
| E7 뒤에 코드 버그가 발견된다 | test를 다시 써도 되는지 논란 | 5절 "test 재실행 규칙"을 따른다 |

---

## 11. 산출물 (저장소 구조)

```
oslab_mobile/
├── PRD.md            ← 이 문서 (현재 메인 계획)
├── README.md         ← 구조·빠른 시작
├── CLAUDE.md         ← AI 작업 규칙 (협업 규칙 포함)
├── AGENTS.md         ← 다른 AI 도구용 요약 → CLAUDE.md
├── rn50/             ← 현재 실험 코드
├── configs/          ← 팀이 고른 최선 설정 (<모델>_best.json, E0에서 생성)
├── hailo/            ← DFC 스크립트(.alls), 매핑표, 컴파일 로그
├── results/          ← 현재 실험 결과 (실험 ID별 폴더: E0_measure/ ...). 실행 때 `RN50_RESULTS=results/<폴더>`로 지정
├── docs/             ← 팀 공통 규칙, 코드 검토
├── siglip2/          ← 준혁 (트랜스포머 계열)
└── archive/          ← 이전 실험 (참고용, 보고서 수치로 쓰지 않음)
```

---

## 12. 참고 문헌·자료

1. 모바일공학전공 학술부, 「2026 모바일 하반기 공모전 트랙 2 최종 안내」 (2026).
2. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP), ICML 2021. arXiv:2103.00020
3. Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR 2018. arXiv:1712.05877
4. Krishnamoorthi, *Quantizing deep convolutional networks for efficient inference: A whitepaper*, 2018. arXiv:1806.08342
5. Nagel et al., *A White Paper on Neural Network Quantization*, 2021. arXiv:2106.08295
6. Wu et al., *Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation*, 2020. arXiv:2004.09602
7. Migacz, *8-bit Inference with TensorRT*, NVIDIA GTC 2017 (Entropy 보정).
8. Nagel et al., *Data-Free Quantization Through Weight Equalization and Bias Correction*, ICCV 2019. arXiv:1906.04721
9. Nagel et al., *Up or Down? Adaptive Rounding for Post-Training Quantization* (AdaRound), ICML 2020. arXiv:2004.10568
10. ONNX Runtime, *Quantize ONNX models*. https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
11. Hailo, *Hailo Quantization Scheme*. https://hailo.ai/de/developer-zone/training-and-support/hailo-quantization-scheme-2/
12. Hailo Model Zoo, *HAILO10H Zero-shot Classification* (DFC v5.4.0). https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_zero_shot_classification.rst
13. Hailo Model Zoo 설정 `cfg/alls/generic/clip_resnet_50_image_encoder.alls`. 요약: `hailo/README.md` (상세 분석은 Claude 프로젝트 문서 `claude/01_Hailo_공식_CLIP_양자화설정_분석.md`)
14. Hailo Community, *How to replicate TFLite's int8 quantization in Hailo optimization* (DFC는 float 모델에서 시작). https://community.hailo.ai/t/how-to-replicate-tflite-s-int8-quantization-in-hailo-optimization/13351
15. ONNX Runtime 소스(2026-10-08 확인): `core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.h`의 `QDQIsInt8Allowed()`, `core/optimizer/graph_transformer_utils.cc`의 `QDQS8ToU8Transformer`. 모두 main 브랜치 기준이다. 1.25.0 동작은 E0에서 출력을 비교해 확인한다(합성 모델에서는 S8S8 = U8S8, 차이 0.0).
16. ONNX Runtime 1.25.0 `quantization/calibrate.py`: Percentile 보정기 기본값 `symmetric=True`, `percentile=99.999`, `num_bins=2048`.
