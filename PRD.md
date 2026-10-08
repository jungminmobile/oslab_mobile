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

> **RN50 또는 RN50x4의 이미지 인코더를 양자화해서, 정확도는 최대로 지키고 용량은 최소로 줄인다.**

| 우선순위 | 지표 | 목표 (성공 기준) |
|---|---|---|
| 1 | **정확도** | 같은 모델 FP32 대비 **고화질 5종(Pets·Flowers·Cars·Food·Caltech) 평균** top-1 손실 **≤ 1.0%p**. 선택은 dev, 보고는 test. CIFAR-100·ImageNetV2는 별도 표기(5절) |
| 2 | **용량** | 위 조건을 지키는 구성 중 **ONNX 파일 크기 최소**. 목표 FP32 대비 −70% 이상 |
| 3 | **추론 시간** | "비정상적으로 느리지 않을 것" = **같은 PC에서 자기 FP32보다 빠를 것**(필수) |
| 필수 | **이식성** | 같은 구성이 Hailo DFC로 Hailo-10H용 `.hef` 컴파일까지 성공할 것 |

**대회 배점 참고** (최종안내 4.4절): 예선은 보고서·실험 설계 35, 기법 25, **성능 개선(속도·메모리·크기) 25**, **정확도 유지 15**점.
안내서에 "정확도 우선"이라는 문구는 없다. 목표의 우선순위(정확도 → 용량 → 속도)는 팀의 설계 선택이다.
보고서에는 속도·메모리 개선 폭도 함께 제시해서 성능 개선 25점을 놓치지 않는다.

---

## 2. 요구사항 (제약)

| # | 요구사항 | 근거 |
|---|---|---|
| R1 | 예선은 채점자가 **자기 로컬 PC에서 직접 실행**하므로 **ONNX 파일**로 제출한다(베이스라인 + 최적화 모델). | 최종안내 4.2·4.3 (재현 실패 시 감점·심사 제외) |
| R2 | 본선은 예선 모델을 **Hailo DFC로 Hailo-10H용 `.hef`**로 컴파일해 써야 한다. Hailo-8/8L용 `.hef`는 호환되지 않는다. | 최종안내 5.2·6 |
| R3 | 그래서 **예선 최적화와 DFC 컴파일 가능성을 동시에 확보**한다. NPU 실측은 본선 전에는 어렵다. 따라서 성능 판단은 **예선 로컬 PC 측정**을 우선한다. | 본선 기간 약 3주 (최종안내 6) |
| R4 | 정확도를 우선하되, 추론 시간이 비정상적으로 느리면 안 된다. | 1절 기준 3 |
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
| 활성값 | 8비트 텐서별. 기본은 ORT 기본값 **S8S8(QDQ)**. 비-VNNI 대비책은 E0에서 결정 | — | ORT 문서: VNNI 없는 AVX2·AVX512에서 U8S8 포화 가능. 문서의 대응책은 **reduce_range(가중치 7비트) 또는 U8U8**. 채널별 가중치는 이런 CPU에서 reduce_range가 대체로 필요 |
| 보정 방법 | MinMax / **Percentile** / **Entropy** 비교 (E2) | — | Wu et al. 2020, Migacz 2017(TensorRT Entropy), ORT 지원 3종 |
| 혼합 정밀도 | 민감한 층은 FP32로 남김 (ORT `nodes_to_exclude`) | — | Nagel 2021 디버깅 절차, Wu 2020 partial quantization |
| 실행 | ORT CPUExecutionProvider, `ORT_ENABLE_ALL`, 스레드 고정 | — | 정수 커널(QLinearConv 등)로 융합됨 |
| 본선 컴파일 | **Hailo Dataflow Compiler 5.x**, `--hw-arch hailo10h` | Model Zoo 공개 `.hef` = DFC 5.4.0 | Hailo Model Zoo HAILO10H 문서 |
| DFC 환경 | 정민 PC의 WSL2 Ubuntu 22.04 또는 Docker (**WSL·Docker는 설치 확인, DFC는 미설치**) | DFC 릴리스 노트의 Python·Ubuntu 버전으로 **별도 가상환경** (예선 코드의 Python 3.11과 분리) | DFC는 x86 리눅스 전용 |
| Hailo 양자화 | 8비트(a8_w8) 기본, 민감 층 16비트(a16_w16) | — | Hailo Quantization Scheme, Model Zoo RN50 alls |

> **4비트를 예선에서 빼는 이유**: ORT 문서가 4비트 양자화를 지원한다고 밝힌 연산은 MatMul·Gather뿐이다. conv를 4비트 QDQ로 만들면 정수 커널로 융합되지 않는다. 이전 실험(archive R7)에서 FP32보다 29~139% **느렸다**(우리 측정). R5(정수 커널만)를 만족하지 못한다.
> 4비트(a8_w4)는 Hailo 단계(H2)의 후보로 넘긴다. Hailo 문서는 4비트 층이 있으면 **QFT + 가중치 잘라내기가 필수**라고 한다.

---

## 4. 설계 원칙

### 4-1. 이미지 인코더만 양자화하는 이유

| 항목 | 이미지만 양자화 | 텍스트도 양자화 |
|---|---|---|
| 실행 빈도 | 사진마다 | 클래스 목록이 정해질 때 1회 |
| 장당 속도 이득 | 있음 | **없음** |
| 정확도 | 오차가 사진 쪽에서만 생김 | 클래스 좌표에도 오차가 생김 |
| 기기에 올리는 크기 | 클래스 좌표 표만(100클래스 × 1024 × 4B ≈ **0.4 MB**) | 텍스트 인코더 INT8 ≈ 60 MB |

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
| 보정 사진 (같은 구간·같은 시드) | `calibset_size` + 같은 사진 | Hailo 권장 32~96장. 64·256·512 비교 (E3) |

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
  ① VNNI 없는 PC를 하나 확보한다(준혁 PC·학과 PC의 CPU 플래그 확인).
  ② 문서의 대응책인 `reduce_range=True`와 U8U8 모델의 정확도 손실을 VNNI PC에서 미리 측정한다.
  ③ 비-VNNI PC를 확보하지 못하면, README에 "VNNI 없는 CPU용 모델(reduce_range)"을 따로 안내한다.

---

## 5. 평가 프로토콜 (팀 공통 · 준혁과 동일하게 맞춤)

| 구간 | 데이터 | 용도 |
|---|---|---|
| 보정 (calib) | CIFAR train[0:45000] 또는 고화질 5종 train(dev 제외 해시 구간). **256장, 시드 0, 섞음** | 눈금 정하기 |
| dev | **6종**: CIFAR train[45000:46000](A), 고화질 5종 train에서 SHA-1 해시로 약 1000장씩. CIFAR B([46000:47000])는 최종 후보 확인에만 | 구성 비교·선택 |
| test | CIFAR-100, **ImageNetV2**(matched-frequency), Oxford Pets, Flowers-102, Stanford Cars, Food-101, Caltech-101 **전체** | **최종 1회만** (E7) |

| 지표 | 정의 | 비고 |
|---|---|---|
| top-1 / top-5 | 제로샷 정확도 (%) | 프롬프트 앙상블(CIFAR 18, ImageNetV2 7, Cars 8, Caltech 34, 그 외 1개) |
| Δtop-1 | 같은 모델 FP32 대비 (%p) | **같은 사진에 대한 짝 비교**. 사진 단위 부트스트랩(1000회)으로 95% 신뢰구간을 낸다. 구간이 0을 포함하면 "차이 없음"으로 본다. 동률이면 cos로 판단 |
| cos | FP32 이미지 임베딩과의 평균 코사인 유사도 | 정답 없이도 재는 보조 지표. 층 민감도 분석의 주 지표 |
| 용량 | ONNX 파일 크기 (MiB), FP32 대비 감소율 | + 클래스 임베딩 표 크기 별도 표기 |
| 지연 | 장당 ms, 배치 1, 스레드 1·2·전체 | 워밍업 20회, 200장 × 3회 번갈아 측정, 평균 ± 표준편차 |
| 처리량 | 장/초, 배치 8 | 보조 |
| 메모리 | 추론 중 최대 RSS (MiB) | 안내서 4.2 "메모리 사용량" 요구 |

Caltech-101은 BACKGROUND_Google, Faces, Faces_easy 클래스를 포함해 채점한다. 이름은 각각 background, off-center face, centered face로 바꾼다.
CIFAR-100은 stem 변화에 과민하다(이전 실험에서 31~46%로 흔들림). 그래서 **구성 선택은 고화질 dev 평균과 cos를 주 지표로** 삼는다. CIFAR는 Hailo 공식 수치 비교용으로 쓴다.

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
| Fon | 전부(stem 포함) | 없음 |

- 모든 구성에서 AttnPool의 attention 계산(점수·softmax·가중합)은 FP32로 둔다.
- 이름 앞에 모델(`rn50_`, `rn50x4_`), 뒤에 보정 조건(`_<방법>_<출처><장수>_s<시드>`)을 붙인다. 예: `rn50_c4_percentile_highres256_s0`
- E4의 블록 단위 구성은 `b_<FP32로 남긴 블록 목록>`으로 표기한다.

### 6-1. 실험 표

| ID | 실험 | 방법 | 근거 | 판단 기준 / 산출물 | 예상 |
|---|---|---|---|---|---|
| **E0** | 측정 기반 정비 | ① `evaluate.py`·`datasets.py test` 기본 실행을 정수 커널로 ② CPU·VNNI·ORT 버전 자동 기록 ③ ONNX 모델용 최대 RSS 측정 추가 ④ 짝 부트스트랩 신뢰구간 계산 추가 ⑤ 비-VNNI 대비: c3·c4를 **S8S8(기본) / reduce_range / U8U8**로 만들어 정확도·속도 비교, 비-VNNI PC 확보 시도 | ORT 문서 (포화·reduce_range) | 기본 S8S8 확정, 비-VNNI용 대체 모델 결정 / 공통 측정 코드 | 0.5~1일 |
| **E1** | **DFC 호환성 관문** | RN50·RN50x4 q1 float ONNX를 `hailo parser`(hailo10h) → optimize(기본 8비트, 64장) → compile `.hef`. 실패 연산 확인, ONNX 노드 ↔ DFC 층 이름 매핑표 작성 | 최종안내 6, Hailo Community | **두 모델 모두 `.hef` 생성 성공**이 이후 실험의 전제 / `hailo/` 스크립트, 매핑표 | 1~2일 (DFC 설치 포함) |
| **E2** | 보정 방법 | MinMax vs Percentile(99.99·99.999) vs Entropy × 구성 c3·c4·c5·Fon (RN50) | Wu 2020, Migacz 2017. 이전 결론: MinMax가 이상치에 끌림 | 고화질 dev Δtop-1(신뢰구간)·cos가 가장 좋은 방법 / 표 | 1일 |
| **E3** | 보정 데이터 | 출처(cifar·highres·mix) × 장수(64·256·512), E2 최선 방법으로. 시드 0·1·2 | Hailo 권장 32~96장, 이전: CIFAR 보정이 cos 높음 | 시드 간 흔들림이 출처·장수 차이보다 작은지 / 표 | 1일 |
| **E4** | 블록 단위 혼합 정밀도 | 단계 단위 선택을 **블록 단위**로 세분화. 블록 수: RN50 16개(3+4+6+3), RN50x4 26개(4+6+10+6), 각각 + stem + AttnPool. ① 블록 하나만 INT8 → 민감도(cos) 순위 (n개 모델) ② 전부 INT8에서 **민감도 순위대로** 하나씩 FP32로 되돌림(재평가 없는 순위 기반, n개 모델) → 용량-정확도 곡선 | Nagel 2021 디버깅, Wu 2020 partial quantization | **고화질 dev Δtop-1 ≤ 1.0%p를 만족하는 최소 용량 구성** / 파레토 곡선 | RN50 약 3시간, RN50x4 약 10시간 (≈1.5일) |
| **E5** | 양자화 전 가중치 보정 | **PyTorch에서 export 전에** Cross-Layer Equalization 적용(병목 블록 안 conv1→conv2→conv3 쌍만. 잔차 합 너머로는 스케일하지 않음). 이후 사진 기반(empirical) Bias Correction 적용 → E4 최선 구성 재양자화 | Nagel et al. 2019 (DFQ) | 개선 신뢰구간이 0을 넘으면 채택. 채널별 가중치에서는 효과가 작을 수 있음(DFQ의 이득은 주로 텐서별). DFC는 자체 equalization이 기본이라, 효과 없음도 결과로 기록 | 1일 |
| **E5+** | (여유 시) AdaRound | 블록별 반올림 학습 → 가중치를 눈금에 맞춰 저장, ORT 스케일 고정 | Nagel et al. 2020 | 선택 실험 | 1.5일 |
| **E8a** | Hailo 사전 검증 (후보 전체) | E4·E5의 파레토 후보들을 `.alls`로 옮겨 컴파일. **DFC 에뮬레이터**로 dev 정확도 측정: 양자화 에뮬레이션(`InferenceContext.SDK_QUANTIZED`)과 float 최적화 모델(`SDK_FP_OPTIMIZED`)을 같이 재서 양자화 오차와 최적화 오차를 분리. 층별 cos를 ORT와 비교 | 최종안내 5.2 (서버 vs 엣지 비교) | 컴파일 성공 여부 + 에뮬레이터 Δtop-1 / E6 입력 | 1일 |
| **E6** | 최종 모델 선택 | RN50 최선 vs RN50x4 최선. 8절 규칙 | Hailo Model Zoo 수치 | 최종 후보 1~2개 | 0.5일 |
| **E7** | 최종 test (1회) | 후보 + 각 FP32 베이스라인. test 7종 전체 + CIFAR B, 정수 커널, **두 PC** | 최종안내 4.2 | 보고서 표 원본 | RN50 3시간 / RN50x4 8시간 |
| **E8b** | Hailo 최종 패키지 | 선택 모델의 `.hef`, `.alls`, 컴파일 로그, 에뮬레이터 test 정확도 | 최종안내 5.3 | 본선 이식 패키지 / 보고서 "이식 계획" | 0.5일 |

**순서와 관문**: E0 → **E1(관문)** → E2 → E3 → E4 → E5 → **E8a** → E6 → E7 → E8b.
E1이 실패하면(파싱 불가 연산) 그래프를 수정(q1, 연산 치환)한 뒤에 진행한다.
DFC 설치가 늦어지면 E1과 E2·E3를 병행한다(E2·E3는 DFC 없이 가능).

### 본선 단계 후보 (예선 이후, 참고)

| ID | 내용 | 근거 |
|---|---|---|
| H1 | Pi 5 + Hailo-10H 실측: 같은 dev/test, FPS·지연·메모리, 서버 결과와 차이 분석 | 최종안내 5.2 |
| H2 | 4비트 혼합(a8_w4) + QFT + 잘라내기로 크기 추가 감소. QFT용 train 사진 2048장(dev 제외)을 미리 준비 | Hailo Quantization Scheme (QFT 기본 2048장) |
| H3 | Hailo 공식 설정 비교: 공식 RN50은 대부분 16비트, CIFAR 42.1→41.2, 142 FPS. 우리 8비트 위주 구성이 더 빠르면서 정확도를 지키는지 | Hailo Model Zoo, `claude/01` 분석 |

---

## 7. 이전 실험에서 가져오는 출발점 (`archive/2026-10-07_night/`, dev·참고용)

| 구성 (RN50) | 용량 | 장당 ms (2코어) | 고화질 Δtop-1 | 비고 |
|---|---|---|---|---|
| FP32 | 146.1 MiB | 54.5 | 0 | |
| c3 (stem·L1·L2 FP32) | 41.5 (−72%) | 33.4 (−39%) | −0.72 | 고화질 기준으로는 목표 만족. CIFAR A 포함 6종 평균은 약 −1.08 |
| c4 (stem·L1 FP32) | 38.1 (−74%) | 25.5 (−53%) | −1.4 ~ −1.6 | 보정 개선 시 목표 근접 |
| RN50x4 c5 (stem만 FP32, CIFAR 보정) | 84.9 | 70.1 | −0.54 (자기 FP32 대비) | RN50 FP32보다 정확도 +3~11%p |

- **주의**: 위 정확도 일부는 fake-quant 값이 섞여 있다. 4비트 결과는 fake 계산 구조다. 보정은 MinMax뿐이다. **E0 이후 정수 커널로 다시 측정한 값만 보고서에 쓴다.**
- 이어받는 결론: ① 보정 버그(16장만 사용) 수정 후 stem INT8 이득은 사라졌다 ② MinMax는 이상치에 끌린다 → E2 ③ 정수 커널과 fake-quant의 순위는 같았다.

---

## 8. 최종 모델 선택 규칙 (E6)

1. **이식성**: E8a에서 `.hef` 컴파일에 실패한 구성은 제외한다.
2. **정확도**: 고화질 5종 dev 평균 Δtop-1 ≤ 1.0%p (자기 FP32 대비)인 구성만 남긴다. 1절 성공 기준과 같은 지표다.
3. **용량**: 남은 구성 중 ONNX 크기가 가장 작은 것을 고른다.
4. **속도**: 같은 PC에서 자기 FP32보다 느리면 제외한다.
5. **RN50 vs RN50x4**:
   - 절대 정확도는 RN50x4가 높다.
   - 크기·속도는 RN50이 유리하다(Hailo 공식: 142 vs 62.6 FPS).
   - 두 후보를 같은 표에 놓고 팀이 결정한다. 기본 제안은 **"정확도 최대화" 목표에 맞춰 RN50x4, 용량 목표가 우선이면 RN50**.
   - 선택한 모델의 FP32가 보고서의 Baseline이 된다.

---

## 9. 일정 (10/8 → 11/1)

| 기간 | 작업 |
|---|---|
| 10/8 ~ 10/10 | E0, DFC 설치(Developer Zone 계정·다운로드), E1 |
| 10/11 ~ 10/15 | E2, E3 |
| 10/16 ~ 10/20 | E4, E5 (여유 시 E5+) |
| 10/21 ~ 10/23 | E8a → E6 |
| 10/24 ~ 10/26 | E7 최종 test (두 PC), E8b |
| 10/27 ~ 11/1 | 보고서 작성, 재현 패키지 점검(새 환경에서 README대로 실행), 제출 |

---

## 10. 위험과 대응

| 위험 | 영향 | 대응 |
|---|---|---|
| DFC가 AttnPool 연산(Softmax·MatMul 모양 등)을 파싱하지 못함 | 본선 이식 불가 | E1을 가장 먼저 실행. q1 그래프·연산 치환. 최후에는 AttnPool을 CPU 후처리로 분리 |
| 채점자 PC에 VNNI가 없음 → 8비트 포화 | 보고 수치 재현 실패 | E0에서 reduce_range·U8U8 대체 모델 준비, 비-VNNI PC 확보 시도. README에 측정 CPU와 예상 범위 명시 |
| DFC 고급 최적화에 GPU 필요 | E8a·E8b 품질 제한 | 기본 최적화 수준으로 진행. 필요하면 GPU 서버 사용 검토 |
| Percentile·Entropy 보정의 메모리 사용량 큼 | 클라우드(7GB)에서 OOM | 보정 배치 축소, 노트북(32GB)에서 실행 |
| test 재사용 유혹 | 점수 부풀림 | test는 E7에서 1회만 실행. 실행 기록(log)을 커밋 |

---

## 11. 산출물 (저장소 구조)

```
oslab_mobile/
├── PRD.md            ← 이 문서 (현재 메인 계획)
├── README.md         ← 구조·빠른 시작
├── rn50/             ← 현재 실험 코드
├── hailo/            ← DFC 스크립트(.alls), 매핑표, 컴파일 로그
├── results/          ← 현재 실험 결과 (E0~E8b, 실험별 폴더)
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
13. Hailo Model Zoo 설정 `cfg/alls/generic/clip_resnet_50_image_encoder.alls`. 분석: 프로젝트 문서 `claude/01_Hailo_공식_CLIP_양자화설정_분석.md`
14. Hailo Community, *How to replicate TFLite's int8 quantization in Hailo optimization* (DFC는 float 모델에서 시작). https://community.hailo.ai/t/how-to-replicate-tflite-s-int8-quantization-in-hailo-optimization/13351
