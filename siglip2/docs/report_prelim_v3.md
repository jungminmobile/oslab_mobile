# 2026학년도 모바일공학전공 하반기 공모전 [트랙 2] 예선 결과 보고서 — 자유주제

> 초안 v3 (2026-10-21, Claude 작성 → 쭈리·팀 검토). 학술부 배포 양식(「트랙2_예선_결과보고서_양식_자유주제.docx」)의 장 순서 1~9를 그대로 따른다.
> - 확정 사항: 예선 제출 = **SigLIP2 ViT-B/16 이미지 인코더, M2rot_ort** (D-046), 본선 HEF = **B16_rb_bk** (D-047). 성능은 원본 대비 개선율로 채점한다(학술부 10/21).
> - 모든 숫자는 저장소 `results/siglip2/` 아래 raw 파일에서 왔다(괄호 안 경로). 정확도는 정수 커널(ONNX Runtime 1.25.0, `ORT_ENABLE_ALL`)로 실제 실행한 값이다.
> - `[작성 필요: …]`는 사람이 채울 칸, `[확인 필요: …]`는 제출 전 확인할 칸이다. PDF로 옮길 때 이 머리말과 표시는 지운다.
> - 분량 목표: 본문 15쪽 내외. 이 초안은 길게 썼으니 PDF로 옮기며 줄인다 (줄일 후보는 각 절 끝 "※ 줄일 때").

---

## 1. 팀 정보

| 구분 | 내용 |
|---|---|
| 팀명 | [작성 필요] |
| 선택 주제 | 자유주제 — 자유 모델 최적화 |
| 대상 모델명 | SigLIP2 ViT-B/16 (224×224) 이미지 인코더 |
| 팀원 (성명 / 학번 / 역할) | 손준혁 / [학번] / SigLIP2 트랙 총괄: 원인 분석, CPU(ONNX Runtime) 양자화·평가·재현, 보고서 |
| | [준용 성명] / [학번] / Hailo-10H 컴파일·실칩 평가, 회전·레지스터 캐시 기법, 교차 측정 |
| | 전정민 / [학번] / 공통 평가 코드(데이터 분할·평가·속도 측정), RN50 비교 트랙, 교차 측정 |
| 대표자 연락처 | [작성 필요] |
| 제출일 | 2026년 10월 [ ]일 |

---

## 2. 결과 요약

정확도 지표는 **제로샷 분류 Top-1 정확도**(고화질 5종 평균, test 분할)다. 지연·메모리는 같은 PC에서 FP32와 같은 실행으로 번갈아 잰 값이다.

| 항목 | Baseline (FP32) | 최적화 후 (INT8, M2rot_ort) | 변화율 |
|---|---|---|---|
| 모델 크기 (MB) | 371.67 | 96.18 | **−74.1%** |
| 파라미터 수 (M) | 92.88 | 92.88 (INT8로 저장) | 0% |
| 연산량 (GFLOPs) | 35.45 | 35.45 (INT8 연산으로 실행) | 0% |
| 추론 Latency (ms, CPU 4스레드, 배치 1) | 251.65 | 126.16 | **−49.9%** |
| Throughput (FPS, 배치 1, 4스레드) | 3.97 | 7.93 | **+99.5% (약 2배)** |
| 피크 메모리 사용량 (MB) | 456.9 | 220.7 | **−51.7%** |
| 정확도 (Top-1, 고화질 5종 평균, test) | 90.34% | 90.00% | **−0.34%p** [95% 구간 −0.50, −0.18] |

(근거: `G8_candidates/b16/A4_metrics`, `G6_m2rot_ort/test`. 측정 PC Intel i7-1165G7, Windows 11, 고성능 전원 모드)

**요약.** 이미지-텍스트 대조 학습 모델 SigLIP2 ViT-B/16의 이미지 인코더를 **INT8 정적 사후 양자화(PTQ)**로 최적화했다. 단순 INT8은 정확도가 91.66% → 28.17%로 무너졌다. 원인을 분석해 보니 특정 위치의 패치 토큰 2개가 특정 채널 4개에 수백 배 큰 값(거대 활성값, attention sink)을 쓰고, 텐서당 눈금 하나인 8비트 양자화가 이 값 때문에 나머지 토큰을 0으로 뭉갰다. 이 원인을 두 축으로 직접 고쳤다.
- **토큰 축 — 레지스터 캐시(M2)**: 문제 토큰의 K·V와 MLP 입력을 보정 데이터 평균 상수로 바꾼다.
- **채널 축 — 오프라인 직교 회전**: 수학적으로 같은 모델로 바꿔 채널 쏠림을 편다.

그 결과 **행렬곱 79개를 모두 INT8로 하고도** 정확도 손실을 test −0.34%p로 줄였다. 크기 −74%, CPU 지연 약 절반(−50~58%), 메모리 −52%를 얻었다. 같은 모델을 Hailo-10H용으로 컴파일해 실칩에서 정확도 손실 −0.41%p(dev), 배치 1 지연 40.5 ms를 확인했다. 같은 처방이 다른 모델(SigLIP2 B/32)에도 그대로 통하는 것도 확인했다.

---

## 3. 대상 모델 및 선정 사유

### 3.1 대상 모델

| 구분 | 내용 |
|---|---|
| 모델명 / 버전 | SigLIP2 ViT-B/16, 입력 224×224 (`hf-hub:timm/ViT-B-16-SigLIP2`, open_clip 3.3.0 / timm 1.0.30로 불러옴) |
| 모델 유형 | 멀티모달 (이미지-텍스트 대조 학습). 이 과제의 최적화 대상은 **이미지 인코더** |
| 저장소 URL | 우리 코드: https://github.com/jungminmobile/oslab_mobile (현재 비공개 — [확인 필요: 학술부 접근 권한 부여 방법]) / 원 모델: https://huggingface.co/timm/ViT-B-16-SigLIP2 |
| 커밋 해시 또는 릴리스 태그 | 제출 시점 main에 `prelim-submit` 태그 [작성 필요: 해시]. 모델 파일: Release `siglip2-m2rot-ort-20261019` (INT8), `siglip2-xplat-20261015` (FP32) |
| 사전학습 가중치 출처 | Google SigLIP2 (Tschannen et al., 2025) 공개 가중치의 timm/open_clip 변환본, Hugging Face 스냅샷 `eee10eff6dd8…` |
| 라이선스 | Apache-2.0 [확인 필요: 모델 카드에서 최종 확인] |

구조: 패치 16×16 → 토큰 196개, 폭 768, Transformer 블록 12개, 헤드 12개, MLP 3072. CLS 토큰 없이 attention pooling(MAP) 헤드로 768차원 임베딩을 낸다. 파라미터 92.88 M, 17.73 GMACs = 35.45 GFLOPs(fvcore, `G2_fp32_baseline`). FP32 ONNX 354.45 MiB.

**범위.** 이미지 인코더만 양자화한다. 텍스트 쪽은 클래스 문장 임베딩을 FP32로 한 번 계산해 저장해 두고 다시 쓴다. 엣지 기기에서 사진마다 실행되는 것은 이미지 인코더뿐이고, 분류할 클래스가 바뀔 때만 텍스트 임베딩을 다시 만들면 되기 때문이다.

### 3.2 선정 사유

1. **엣지에서 쓰임새가 넓다.** 이미지 임베딩 하나로 제로샷 분류, 이미지 검색, 콘텐츠 필터링을 모두 할 수 있다. 분류할 대상이 바뀌어도 재학습 없이 텍스트만 바꾸면 된다. 카메라 기기에서 "무엇이 찍혔는지"를 클라우드 없이 판단하는 기본 부품이 된다.
2. **최신 모델이고 엣지 최적화 가치가 크다.** SigLIP2(2025)는 CLIP 계열의 최신판으로, 같은 크기에서 CLIP보다 제로샷 정확도가 높다. FP32 이미지 인코더가 371.7 MB, CPU 4스레드에서 사진 한 장에 250 ms 안팎이라 그대로는 모바일·엣지에서 무겁다.
3. **Hailo-10H로 옮길 수 있지만 공식 결과가 없다.** 같은 계열인 SigLIP(1세대) B/16이 Hailo Model Zoo에 있으므로 구조가 Hailo에서 돈다는 근거가 있다. 하지만 SigLIP2 B/16용 공식 컴파일 결과는 없어 직접 해결해야 할 기술적 과제가 분명하다.
4. **양자화가 어렵다는 점 자체가 과제다.** 단순 INT8로 정확도가 무너진다(5.3절). 왜 무너지는지 찾아 원인을 고치는 과정이 이 과제의 핵심 기법이 된다.

### 3.3 Hailo-10H 호환성 사전 검토

| 항목 | 검토 결과 |
|---|---|
| 지원 연산 | Conv(패치 임베딩), MatMul, Add, Softmax, GELU, LayerNorm, Reshape/Transpose로 구성 → 모두 Hailo DFC가 지원. LayerNorm은 DFC가 기본 연산(평균·분산·곱)으로 분해해서 실행 |
| 비표준 층 | 없음. 우리가 넣은 레지스터 캐시(M2)도 상수 Mul·Sub·Add로만 표현해 표준 연산으로 컴파일됨. 토큰을 지우는 형태(R1, Concat)는 DFC의 LayerNorm 분해와 충돌해 쓰지 않고, 토큰 수를 유지하는 마스크 형태(M2)로 바꿨다 |
| 실제 컴파일 | Hailo DFC 5.4.0, `--hw-arch hailo10h`로 **HEF 컴파일 성공**. 최종 B16_rb_bk: 110.9 MB, 컨텍스트 16 (`G6c`, 준용) |
| CPU Fallback | **없음.** `hailortcli parse-hef` 확인 결과 후처리(NMS 등) 연산이 없고 모든 층이 칩에서 실행된다. 입력 정규화만 HEF 밖(호스트)에서 하며 0.18 ms다 |
| 혼합 정밀도 | 대부분 8비트(a8_w8). 원인 분석이 가리킨 층만 16비트(a16_w16): 패치 임베딩 conv1, 블록 0·5 MLP, 블록 9 fc2, 잔차 덧셈 26개 |
| 실칩 검증 | NVIDIA AGX Orin + Hailo-10H(HailoRT 5.3.0)에서 dev 전체 정확도 손실 −0.41%p, 배치 1 지연 40.51 ms (+정규화 0.18 ms) |
| INT4 | Hailo-10H는 INT4를 지원한다. CPU에서는 4비트가 느려 넣지 않았고(5.3절), Hailo INT4는 본선에서 시험한다 |
| 버전 위험 | HEF는 DFC와 실행 기기 HailoRT 버전이 맞아야 한다. 팀 DFC 5.4.0, 본선 Pi의 HailoRT 버전은 [확인 필요: 주최측 확인 중] → 버전에 맞게 다시 컴파일하는 경로를 확보해 둠 |

※ 줄일 때: 3.3 표는 그대로 두고 3.2를 3줄로.

---

## 4. 실험 환경

| 구분 | 사양 및 버전 |
|---|---|
| CPU | Intel Core i7-1165G7 (4코어 8스레드, AVX2·AVX-512F·AVX-512 VNNI). 교차 측정: Intel Core Ultra 9 285K (AVX-VNNI), Intel Core Ultra 7 258V (AVX-VNNI), NVIDIA Jetson AGX Orin CPU (ARM Cortex-A78AE) |
| GPU (모델 / VRAM) | 사용 안 함 (양자화·평가·속도 측정 모두 CPU) |
| RAM | 16 GB |
| OS | Windows 11 (빌드 26300). Hailo 작업: WSL2 Ubuntu 22.04 / Linux |
| Python 버전 | 3.11.9 |
| 딥러닝 프레임워크 / 버전 | PyTorch 2.14.0 (CPU), open_clip_torch 3.3.0, timm 1.0.30, transformers 5.19.0 (토크나이저) |
| CUDA / cuDNN 버전 | 사용 안 함 |
| 양자화·최적화 도구 / 버전 | ONNX 1.23.1 (opset 17), **ONNX Runtime 1.25.0** (정적 양자화 `quantize_static`, 실행 `ORT_ENABLE_ALL`), numpy 2.4.4, Hailo Dataflow Compiler 5.4.0, HailoRT 5.3.0 |

- 전체 의존성: `siglip2/requirements.txt` (공통 부분은 `rn50/requirements.txt`). 결과마다 CPU 명령어 집합·ORT 버전·커밋 해시를 `env.json`과 결과 jsonl에 남겼다.
- 지연은 노트북 전원 설정에 크게 영향받아(조용 모드는 약 1.7배 느림) **고성능 모드**에서만 쟀고, 공용 잠금으로 다른 무거운 작업과 겹치지 않게 했다.

---

## 5. 적용한 최적화·양자화 기법

### 5.1 기법 요약

단독 효과는 dev 전체(고화질 4,923장) Top-1 손실(FP32 대비)이다. "2×2"는 같은 양자화 설정(MatMul 전부 INT8, U8 + reduce_range, Percentile 99.999)에서 한 가지만 켜고 끈 비교다(`G8_candidates/b16/A1_ablation`).

| 적용 기법 | 적용 대상 | 주요 설정값 | 단독 적용 효과 |
|---|---|---|---|
| ① 정적 PTQ (INT8) | 이미지 인코더 MatMul 79개 | ORT `quantize_static`, QDQ, 가중치 채널별 S8, 활성값 U8 + reduce_range, 보정 256장 | 이것만 하면 **붕괴**: −63.49%p (cos 0.657) |
| ② 보정 방법: Percentile | 활성값 범위 | 99.999 백분위, 비대칭 범위, 히스토그램 16장씩 누적 | 원인 층(fc2)을 보호한 상태에서 MinMax 대비 cos 0.936 → 0.959, Entropy 0.940 (`G3_calib_method`, 200장 스크리닝) |
| ③ 레지스터 캐시 M2 (토큰 축) | sink 토큰(패치 55·13)의 K·V(블록 5·9부터)와 MLP 입력 | 보정 train 256장 평균 상수, 토큰 196개 유지 마스크 형태 | **−63.49 → −0.62%p** (+62.87 [+61.59, +64.17]) |
| ④ 오프라인 직교 회전 (채널 축) | 잔차 흐름 768채널 | 무작위 부호 Hadamard × Householder(Q·1 = 1), LayerNorm γ·β 흡수 | 단독: −63.49 → −48.97%p (+14.52). M2 위에 추가: −0.62 → −0.31 (+0.32, CPU에서는 유의하지 않음). Hailo 에뮬레이터 cos 0.980 → 0.987, 실칩 회전만: −1.79 → −0.94%p |
| ⑤ 활성값 U8 + reduce_range | x86 정수 커널 선택 | 활성값 UInt8, 가중치 7비트 범위 | 속도: S8은 QKV 투영이 정수로 안 합쳐져 4스레드 FP32 대비 0~14%만 빨라지지만 U8은 약 2배. reduce_range는 U8의 포화 손실을 S8 수준으로 회복 |
| ⑥ (발전 단계) 혼합 정밀도 R0 | 블록 5·9 MLP fc2만 FP32 | 나머지 77개 INT8 | −0.78%p (최종에서는 ③④로 대체, 크기 105.3 → 91.7 MiB) |
| ⑦ Hailo 혼합 정밀도 컴파일 | HEF | a16_w16: conv1, 블록 0·5 fc1·fc2, 블록 9 fc2, 잔차 덧셈 26개 / 나머지 a8_w8, LayerNorm 분해, optimization_level 1, 보정 256장 | 실칩 손실 −0.41%p (M2rot 그래프와 함께) |
| ⑧ Hailo 컴파일 자원 설정 | HEF 배치·자원 할당 | `resources_param(1.0, 0.95, 0.75)` + 수동 bucket 3개 | 출력 그대로(비트 단위 동일), 배치 1 지연 **−14.8%** (47.55 → 40.51 ms), 컨텍스트 21 → 16 |

**최종 구성(M2rot_ort) = ① + ② + ③ + ④ + ⑤.** 본선 HEF = 같은 M2rot 그래프 + ⑦ + ⑧.

### 5.2 양자화 상세

| 구분 | 내용 |
|---|---|
| 양자화 방식 (PTQ / QAT) | **PTQ** (정적 사후 양자화, 재학습 없음) |
| 정밀도 (INT8 / INT4 / 혼합) | **INT8** — 이미지 인코더의 MatMul 79개 전부 (가중치 채널별 INT8, 활성값 UINT8 + reduce_range). Hailo HEF는 8/16비트 혼합 |
| 캘리브레이션 데이터셋 | 고화질 5종(Pets·Flowers·Cars·Food·Caltech)의 **train 분할**에서 dev·test와 겹치지 않게 뽑은 보정 후보(약 311장) |
| 캘리브레이션 샘플 수 | **256장** (시드 0). 실제로 보정에 들어간 장수를 코드에서 확인(assert)한다 — ORT 보정기가 일부 배치를 버리는 문제를 막기 위함 |
| 양자화 제외 레이어 | 행렬곱(MatMul)은 **제외 없음**. MatMul 이외 연산(패치 임베딩 Conv, LayerNorm, GELU, 잔차 덧셈, M2 마스크 연산)은 FP32로 계산한다. Softmax는 ORT가 정수 Softmax(QLinearSoftmax)로 실행한다 |

### 5.3 적용 근거 및 과정

**단계 1 — 출발점: 단순 INT8은 무너진다.** ORT 정적 INT8(MinMax, 모든 연산)로 양자화하면 고화질 Top-1이 91.66% → 1.37%(cos 0.445)로 무너졌다(`G1_int8_minmax_all`). Hailo DFC 기본 8비트 설정도 cos 0.12였다.

**단계 2 — 어디가 민감한가: 민감도 지도.** 블록·연산 하나만 INT8로 바꾸고 나머지는 FP32로 둔 모델을 20개 만들어 dev 6종 × 200장으로 쟀다(`G3_sensitivity`).
- 블록 1~4는 거의 무손실(cos ≥ 0.989)이지만 **블록 5~11은 어느 하나만 INT8이어도 붕괴**(cos 0.41~0.51)한다.
- 연산 유형별로는 QKV·attention 행렬곱은 둔감하고(cos ≥ 0.996), **MLP fc2만 붕괴**(cos 0.648)한다.

**단계 3 — 왜 민감한가: 거대 활성값과 attention sink.** 블록마다 잔차 흐름·fc2 입력·attention을 들여다봤다(`G3_why`).
- **블록 5와 9의 MLP**가 이미지와 상관없이 **고정 위치 패치 토큰 1~2개(패치 55, 13)**의 **특정 채널 4개(67·301·145·483)**에 최대 수백의 값을 써 넣는다. 다른 토큰들은 attention의 27~52%를 이 토큰에 몰아준다. CLS 토큰이 없는 SigLIP2가 패치 토큰 일부를 "register"처럼 쓰는 attention sink 현상으로 해석된다 (Darcet et al., 2023; Sun et al., 2024).
- 텐서 하나에 눈금 하나(per-tensor)인 8비트에서는 이 몇 개 값 때문에 눈금 간격이 22~78배 넓어지고, **보통 토큰 값의 84~99%가 0으로 반올림**된다. 망가지는 것은 거대 값이 아니라 나머지 토큰이다.
- 검증(메커니즘 확인용 fake-quant): fc2를 8비트로 하되 토큰별 또는 채널별 눈금을 쓰면 cos 0.999로 회복된다. 즉 **"8비트가 부족한 것이 아니라 눈금 단위가 맞지 않는 것"**이다.

**단계 4 — 첫 해결: 원인 층만 보호(R0).** 블록 5·9 fc2 두 층만 FP32로 두고 나머지를 INT8로 했다. 보정 방법을 MinMax·Percentile 99.99·99.999·Entropy로 비교해 Percentile 99.999를 골랐고(`G3_calib_method`), 보호할 fc2를 7개 → 4개 → 2개로 줄여도 정확도가 동률이라 2개만 남겼다(`G3_cumulative`).

**단계 5 — x86에서 실제로 빨라지게: 활성값 형식.** 같은 모델이라도 활성값 형식에 따라 x86에서 실행되는 커널이 달랐다.
- S8 활성값에서는 QKV 투영 MatMul 13개가 정수로 합쳐지지 않아, Q/DQ 변환이 4스레드 시간의 36%를 차지했다(프로파일링).
- U8로 바꾸니 77개가 모두 정수로 합쳐져 약 2배 빨라졌다.
- U8에서 생긴 작은 손실은 가중치를 7비트 범위로 줄이는 **reduce_range**로 회복했다. 이것은 VNNI가 없는 x86에서 생기는 포화 문제도 막는다.
- 결과 R0: dev −0.78%p, test −0.83%p, 105.3 MiB.

**단계 6 — 원인을 직접 고치기: 레지스터 캐시 + 회전(최종).** 단계 3의 두 원인(특정 토큰, 특정 채널)을 각각 처방했다. 판정 기준은 결과를 보기 전에 정했다.
- **토큰 축 — 레지스터 캐시(M2)**: 거대 값을 쓰는 고정 위치 토큰(패치 55·13)의 K·V를 보정 데이터 평균 상수로 바꾸고, 그 토큰 행의 MLP 입력을 평범한 값으로 바꿔 끼운다. 토큰 수는 그대로 두고 상수 마스크(Mul·Sub·Add)로 표현해 Hailo에서도 컴파일된다. 원래 모델의 근사이지만 FP32 출력 cos 최소 0.998로 거의 같다.
- **채널 축 — 오프라인 회전**: 잔차 흐름을 직교 행렬 Q로 돌리고 역회전을 앞뒤 가중치에 흡수한다(QuaRot·SpinQuant 계열). LayerNorm의 γ·β를 접어 넣어 원래 모델과 **수학적으로 같다**(FP32 cos 0.9999999). 4개 채널에 몰린 거대 값이 768채널에 고르게 퍼진다.
- **단독 효과(2×2, 5.1표)**: M2 하나로 −63.5 → −0.62%p가 되어 **토큰 축이 결정적**이었다. 회전만으로는 여전히 무너졌다(−49.0). 둘을 합치면 CPU에서 −0.31(이 PC 빌드)/−0.23(배포 파일)이다.
- **결과**: 블록 5·9 fc2 예외 없이 **MatMul 79개 전부 INT8**로도 R0보다 정확해졌다(dev +0.56 [+0.10, +1.01], test +0.49 [+0.32, +0.68]). 크기도 105.3 → 91.7 MiB로 줄었다.
- 예선 ONNX와 본선 HEF가 같은 M2rot 그래프다.

**단계 7 — Hailo-10H 이식.** 같은 M2rot 그래프를 DFC로 컴파일했다(준용).
- 원인 분석이 가리킨 층(블록 0·5 MLP, 블록 9 fc2, 잔차 덧셈)만 16비트로 두었다. Hailo 공식 SigLIP 설정도 블록 5 MLP를 16비트로 두는데, 우리 분석의 "거대 값을 처음 쓰는 MLP"와 같은 위치다.
- 보정 256장과 optimization_level 1(바이어스 보정)로 cos가 0.949 → 0.967로 올랐다.
- 회전을 넣자 실칩 손실이 −1.79 → −0.94%p로 줄었고, M2까지 넣으니 −0.41%p가 됐다.
- 마지막으로 컴파일 자원 설정(`resources_param` + 수동 bucket)을 넣어, 출력은 그대로 두고 지연을 15% 줄였다(47.55 → 40.51 ms).

**시도했지만 넣지 않은 것 (실패 사례, 결과 전에 정한 기준으로 판정).**

| 시도 | 결과 | 넣지 않은 이유 |
|---|---|---|
| 가중치 4비트 (ORT MatMulNBits, 블록 32, 4구성) | 크기 105.3 → 73.6 MiB(−30%), 정확도 R0와 동률 | x86에서 4스레드 지연이 R0의 1.4~2.05배(FP32 수준) — 4비트 커널이 매번 가중치를 풀고 활성값을 동적으로 양자화 (`G3_w4`) |
| 활성값 S8 (fc2L2 S8) | 정확도는 U8보다 조금 나음 (−0.50) | x86에서 QKV가 정수로 안 합쳐져 4스레드 속도 이득이 0~14% (ARM에서는 2.26배) |
| 토큰 삭제형 레지스터 캐시 (R1) | 크기 −13%, MatMul 전부 INT8 | 사전 비열등 기준 미달, Hailo LayerNorm 분해와 충돌 → 마스크형(M2)으로 대체 |
| CPU 저장 형식 변경 (QOperator / 데이터 이동 연산 정수 / 연산 전부 정수) | 최대 −2.1% (흔들림 수준) | 사전 기준 15% 미달. 정수 구간이 LayerNorm·GELU에서 끊겨 변환 노드가 오히려 늘었다(66 → 118·137개) (`G8_candidates/b16/A5_format`) |
| 동적 양자화 | CPU에서 9~19% 빠름 | CIFAR −27%p, Hailo에서 쓸 수 없음 (준용 G7) |
| Hailo 16비트 층 더 줄이기 (B16_both) | 에뮬레이터는 정상 | 실칩에서 출력 붕괴(50.68%) — compile 결과물 이상, 원인 미확인 (`G6c`) |

※ 줄일 때: 단계 4·5를 한 문단으로, 실패 표는 유지 (양식: "실패한 시도도 근거가 명확하면 평가에 반영").

---

## 6. 벤치마크 설계 및 측정 결과

### 6.1 선정한 벤치마크

| 구분 | 내용 |
|---|---|
| 데이터셋 / 벤치마크명 | **제로샷 이미지 분류 7종**: Oxford-IIIT Pets, Oxford Flowers-102, Stanford Cars, Food-101, Caltech-101 (고화질 5종, 주 지표) + CIFAR-100, ImageNetV2 matched-frequency (보조 지표) |
| 공개 여부 및 출처 URL | 모두 공개. Hugging Face Hub: `timm/oxford-iiit-pet`, `dpdl-benchmark/oxford_flowers102`, `tanganke/stanford_cars`, `ethz/food101`, `dpdl-benchmark/caltech101` / CIFAR-100: torchvision / ImageNetV2: https://github.com/modestyachts/ImageNetV2 (matched-frequency) |
| 평가셋 규모 (샘플 수) | **test** (최종 1회): Pets 3,669 · Flowers 6,149 · Cars 8,041 · Food 25,250 · Caltech 6,084 · CIFAR-100 10,000 · ImageNetV2 10,000 (총 69,193장) / **dev** (구성 선택용, train에서 뗌): 고화질 5종 4,923장 + CIFAR 1,000장 |
| 평가 지표 | ① 제로샷 **Top-1 정확도** (고화질 5종 평균, FP32 대비 Δ와 짝 부트스트랩 95% 구간) ② FP32 대비 **이미지 임베딩 코사인 유사도** ③ 크기·지연·처리량·피크 메모리 (FP32 대비 변화율) |
| 지표 선정 사유 | SigLIP2의 실제 쓰임새(텍스트로 정한 클래스에 사진을 맞추기)를 그대로 잰다. 텍스트 분류기를 FP32로 고정해 정확도 차이가 이미지 인코더 양자화에서만 나오게 했다. 코사인은 정답과 무관하게 양자화 오차 크기를 직접 본다 |

### 6.2 벤치마크 선정 사유 및 타당성

- **분야가 다른 세밀 분류 5종을 평균**: 동물(Pets 37종), 식물(Flowers 102종), 사물(Cars 196종), 음식(Food 101종), 일반 사물(Caltech 101종)로 분야가 다르다. 비슷한 클래스끼리 구분해야 해서 작은 임베딩 오차도 정확도에 드러난다. 한 데이터셋에 맞춘 최적화가 다른 분야에서 무너지지 않는지 함께 본다.
- **제로샷 프롬프트는 CLIP 논문 공식 템플릿**을 썼다. SigLIP 권장 템플릿과 dev에서 비교해 CLIP 템플릿이 1.53%p 유의하게 좋아서 정했다.
- **보조 지표 2종을 따로 적는다.**
  - CIFAR-100은 32×32 사진을 224로 늘린 것이라 앞쪽 층 변화에 지나치게 민감하다(INT8에서 −4%p 안팎). 주 지표에 섞으면 실제 사용 환경과 다른 저해상도 영향이 커져서 따로 표기한다.
  - ImageNetV2는 대표 벤치마크(ImageNet) 분포의 새 test 집합이라 일반성 확인용으로 넣었다.
- **선택하지 않은 대안**
  - ImageNet-1k 검증 5만 장: 데이터 접근이 등록제이고 학술부 재현이 어렵다. 대신 공개된 ImageNetV2(1만 장)를 썼다.
  - 이미지-텍스트 검색(COCO·Flickr 검색 Recall@K): 텍스트 인코더까지 실행해야 해 "이미지 인코더 최적화" 효과가 섞인다.
  - 선형 분류(linear probe): 학습이 필요해 재현 시간이 길고, 실제 쓰임새(제로샷)와 다르다.
- **데이터 구간을 섞지 않았다.**
  - 사진 선택은 이미지 바이트의 SHA-1 해시로 정해진다(누가 돌려도 같은 사진).
  - 보정(256장)과 구성 선택(dev)은 train에서만 했고, **test는 최종 모델로 한 번만** 썼다.
  - 기록: 이전 최종(R0)으로 test를 1회 쓴 뒤, dev 결과만으로 M2rot_ort 교체를 확정하고 test를 1회 더 썼다. test는 고르는 데 쓰지 않았다.
- **통계**: 후보 간 차이가 1%p 안팎이라 같은 사진끼리 짝지은 부트스트랩(데이터셋별 층화, 1,000회, 시드 0)으로 95% 구간을 내고, 구간이 0을 넘는지로 "유의한 차이"를 판정했다. 판정 기준은 결과를 보기 전에 정해 기록했다.

### 6.3 재현 가능성 확보 방안

- **공개 데이터셋만 쓴다.** 데이터는 실행 때 Hugging Face Hub·torchvision에서 자동으로 받는다(`rn50/datasets.py`, 사진 선택 규칙 포함).
- **소규모 평가 서브셋 제공**: 전체 데이터 없이도 확인할 수 있게 **dev 6종 × 200장 묶음**(전처리된 입력·정답·FP32 텍스트 분류기·FP32 임베딩, `dev200/<set>.npz`)과, numpy + onnxruntime만으로 도는 `portable_eval.py`를 함께 낸다(Release `siglip2-xplat-20261015`). 구성: 각 데이터셋 dev의 앞 200장(SHA-1 해시 순서).
- **모델 파일과 sha256 공개**: FP32 `333cad08…`, 최적화 `e84c3727…`.
- **CPU가 달라도 같은 결과**: 같은 최적화 ONNX가 x86 세 세대(AVX-512 VNNI i7-1165G7, AVX-VNNI Ultra 9 285K·Ultra 7 258V)에서 dev 결과가 같고, i7과 258V는 사진별 출력까지 비트 단위로 같다.
- **채점·재현은 배포한 ONNX 파일로 한다.** 같은 명령으로 모델을 처음부터 다시 만들면 같은 PC에서는 비트 단위로 같은 파일이 나온다(새 가상환경에서 확인). PC(CPU·수학 라이브러리)가 다르면 행렬 곱 반올림이 달라 파일이 달라지고, 정확도가 0.1%p 안팎 다를 수 있다(`G8_candidates/b16/A2_fresh_env`, `repro_fingerprint`). 9.4절에 적었다.

### 6.4 측정 프로토콜

| 측정 조건 | 설정값 |
|---|---|
| 입력 해상도 | 224×224 (SigLIP2 전처리: 바로 크기 맞춤 bicubic, 정규화 평균·표준편차 0.5) |
| 배치 크기 | 1 (엣지 실시간 사용 상황). 정확도 평가만 배치 50 |
| Warm-up 횟수 | 20회 (모델마다) |
| 측정 반복 횟수 | 사진 200장 × 3회를 **모델끼리 번갈아** (같은 실행 안에서 FP32와 최적화 모델을 교대로) × 세션 3번, 스레드 1·2·4 |
| 통계 방식 | 세션마다 장당 평균 지연 → **3세션 중앙값**. 개별 호출 지연의 p50·p95도 기록. 처리량 FPS = 1000 / 지연(배치 1) |
| 측정 도구 및 방법 | ONNX Runtime 1.25.0 CPU, `ORT_ENABLE_ALL`, intra-op 스레드 고정·inter-op 1 (`rn50/bench_speed.py`, `siglip2/scripts/cross_latency.py`). 피크 메모리: 모델마다 새 프로세스에서 세션 생성 + 추론 50회 중 최대 RSS. 크기: ONNX 파일 크기. 노트북 고성능 모드, 충전 중, 측정 전 CPU 30초 평균 25% 미만일 때만 시작 |

### 6.5 측정 결과

**정확도 (test, 최종 1회, `G6_m2rot_ort/test`)**

| 데이터셋 | FP32 | 최적화 후 | Δ (%p) | cos |
|---|---|---|---|---|
| Pets (3,669) | 95.28 | 94.77 | −0.51 | 0.984 |
| Flowers (6,149) | 85.69 | 85.02 | −0.67 | 0.978 |
| Cars (8,041) | 93.28 | 93.06 | −0.22 | 0.973 |
| Food (25,250) | 92.76 | 92.19 | −0.57 | 0.977 |
| Caltech (6,084) | 84.70 | 84.98 | +0.28 | 0.969 |
| **고화질 5종 평균** | **90.34** | **90.00** | **−0.34 [−0.50, −0.18]** | 0.976 |
| (보조) CIFAR-100 (10,000) | 76.92 | 72.83 | −4.09 | 0.914 |
| (보조) ImageNetV2 (10,000) | 71.28 | 70.56 | −0.72 | 0.981 |

**정확도 (dev 전체, 구성 비교)** — FP32 91.66%. 최종 M2rot_ort −0.23 [−0.64, +0.18] (유의하지 않음), cos 0.976. 이전 R0 −0.78 [−1.26, −0.31].

**효율 (이 PC i7-1165G7, 고성능, 같은 실행, `G8_candidates/b16/A4_metrics`)**

| 항목 | Baseline | 최적화 후 | 변화율 (%) |
|---|---|---|---|
| 정확도 지표 1 (Top-1, 고화질 5종, test) | 90.34% | 90.00% | −0.34%p |
| 정확도 지표 2 (임베딩 cos, 고화질 5종, test) | 1 | 0.976 | — |
| 추론 Latency (ms) 1스레드 / 2스레드 / 4스레드 | 380.39 / 330.70 / 251.65 | 180.24 / 165.24 / 126.16 | **−52.6 / −50.0 / −49.9** |
| Latency p50 / p95 (4스레드, ms) | 253.76 / 274.66 | 125.90 / 136.99 | −50.4 / −50.1 |
| Throughput (FPS, 배치 1) 1 / 4스레드 | 2.63 / 3.97 | 5.55 / 7.93 | **+111 / +99.5** |
| 모델 크기 (MB) | 371.67 | 96.18 | **−74.1** |
| 피크 메모리 (MB) | 456.9 | 220.7 | **−51.7** |

**다른 기기에서의 개선율** (같은 파일, 각 기기에서 FP32와 같은 실행으로 번갈아)

| 기기 | 4스레드 FP32 → 최적화 | 변화율 |
|---|---|---|
| i7-1165G7 (이 PC, 10/20 다른 실행) | 173.88 → 83.99 ms | −51.7% (1스레드 −58.3%) |
| Core Ultra 9 285K (준용) | 67.14 → 29.71 ms | −55.7% (1스레드 −62.3%) |
| Core Ultra 7 258V (정민) | 156.8 → 76.6 ms | −51.2% (1스레드 −62.7%) |

- 지연 절대값은 PC·날짜에 따라 다르지만(같은 PC 4스레드 FP32 174~252 ms), **FP32 대비 개선율은 기기와 날짜가 달라도 −50~63%로 일정**하다.

**Hailo-10H 실칩** (본선 이식 확인, AGX Orin + Hailo-10H, dev 전체, `G3RQ`·`G6c`, 준용)

| 항목 | 값 |
|---|---|
| 정확도 (고화질 5종, dev) | FP32 91.66 → 91.26%, Δ −0.41 [−0.84, +0.05] (유의하지 않음), cos 0.986 |
| 배치 1 지연 | 40.51 ms (+ 호스트 정규화 0.18 ms) = 약 24.6 FPS |
| 배치 8 처리량 | 44.8 장/초 |
| HEF 크기 / 컨텍스트 | 110.9 MB / 16 |

### 6.6 결과 해석

- **손실은 어디서 생기나.** 남은 손실은 MatMul 79개를 모두 8비트로 한 데서 오는 작은 임베딩 오차다(cos 0.976). 원인 분석이 가리킨 "거대 값 때문에 나머지 토큰이 0으로 뭉개지는" 손실은 M2와 회전으로 사라졌다. 2×2 비교에서 M2만으로 손실이 −63.5 → −0.62%p가 된 것이 그 근거다.
- **어떤 입력에서 취약한가.**
  1. **원래 비슷한 클래스 쌍의 경계에 있는 사진**: 새 오답은 특정 클래스에 몰리지 않고 클래스마다 1~4장씩 흩어져 있다. marigold ↔ english marigold, beef tartare ↔ tuna tartare처럼 원래 헷갈리는 쌍에서 순위가 뒤집힌다(`G3_error_classes`). 특정 클래스가 망가진 것이 아니라, 작은 오차가 경계 근처 사진의 순위를 바꾼 것이다.
  2. **저해상도 사진(CIFAR-100, −4.09%p)**: 32×32를 224로 늘린 사진은 앞쪽 층 값 분포가 고화질 사진과 달라 8비트 눈금에 더 민감하다. 같은 모델의 B/32(패치가 커서 저해상도 영향이 작음)는 CIFAR 손실이 −0.2%p였다.
  3. 데이터셋별로는 Flowers(−0.67)·Food(−0.57)처럼 클래스가 많고 비슷한 것끼리 구분해야 하는 쪽이 조금 더 잃고, Caltech(+0.28)는 오히려 올랐다(구간 안의 흔들림).
- **dev와 test가 일치한다.** dev −0.23 [−0.64, +0.18], test −0.34 [−0.50, −0.18]로 같은 수준이다. 이전 단계(R0) 대비 개선도 dev +0.56, test +0.49로 같다. dev에서 여러 변형을 고르며 생길 수 있는 과적합이 test에서 드러나지 않았다.
- **속도가 2배에 그친 이유.** 행렬곱은 INT8로 빨라졌지만, 정수로 계산할 수 없는 연산(LayerNorm·GELU) 앞뒤의 변환(Q/DQ)이 4스레드 시간의 12.8%, M2 마스크 연산이 약 4~5%를 차지한다(`G6_m2rot_ort/profile`). 저장 형식을 바꿔도 줄지 않아(5.3절 실패 표) ORT CPU 경로에서는 이것이 한계로 보인다.
- **같은 처방이 다른 모델에도 통한다 (일반화).** SigLIP2 B/32(256×256, 토큰 64개)에 같은 실험을 반복했다(`G8_candidates/b32`).
  - 민감도 지도(블록 5부터 붕괴, 원인 fc2), 보정 방법, 혼합 정밀도, 정수 개입 실험 결과가 B/16과 같았다.
  - 2×2도 같은 경향이었다: 원래 −61.0 → M2만 −1.58 → 회전+M2 −0.79%p.
  - Hailo 실칩에서는 Hailo 공식 Model Zoo의 B/32 HEF(−4.22%p)보다 우리 방식이 +3.38%p 정확했다(준용 G6).

※ 줄일 때: 6.5의 "다른 기기" 표와 Hailo 표를 한 표로, 6.6의 일반화는 3줄로.

---

## 7. 한계점 및 엣지 디바이스 이식 계획

### 7.1 현재 한계점

1. **CPU 속도 향상이 약 2배에서 멈춘다.** 정수로 계산할 수 없는 LayerNorm·GELU 앞뒤의 변환 비용(약 13%) 때문이다. ORT CPU 경로에서는 저장 형식으로 줄일 수 없었다.
2. **저해상도 입력에서 손실이 크다.** CIFAR-100은 −4.09%p다. 실제 엣지 카메라는 고해상도라 주 지표에서는 영향이 작지만, 작은 사진을 키워 쓰는 상황에서는 주의가 필요하다.
3. **레지스터 캐시는 근사다.** sink 토큰 위치(패치 55·13)와 상수는 보정 데이터로 정했다. 분포가 아주 다른 입력에서는 맞지 않을 수 있다. 다만 dev·test 7종에서 확인했다.
4. **다른 PC에서 다시 만들면 파일이 달라진다.** 회전 행렬 계산부터 CPU·수학 라이브러리의 반올림이 달라진다. 그래서 채점·재현은 배포 파일로 해야 한다.
5. **ARM CPU에서 최종 모델 정확도는 아직 확인하지 않았다.** 이전 단계(R0)는 ARM(Orin)에서 정확도 차이 0.00 [−0.6, +0.5]%p, 4스레드 2.27배였다.
6. **4비트는 CPU에서 느려 넣지 않았다.** Hailo INT4는 본선에서 시험한다.
7. **원인 분석의 일부(토큰별 눈금 실험)는 메커니즘 확인용 fake-quant다.** 지금 ORT·Hailo 정수 커널에서 그대로 쓸 수는 없다.

### 7.2 본선 이식 계획 (11/6 ~ 12/4)

본선은 예선 모델(M2rot)을 그대로 쓰고, 이미 만들어 실칩에서 확인한 HEF(B16_rb_bk)를 Raspberry Pi 5 + AI HAT+ 2(Hailo-10H)에 올린다. 예선과 같은 벤치마크·측정 방법(배치 1 지연 주 지표)으로 서버(ORT INT8)와 엣지를 비교한다.

| 시기 | 수행 내용 | 예상 리스크 |
|---|---|---|
| 1주차 (11/6~11/12) | Pi 5 + AI HAT+ 2 환경 구성, HailoRT 버전 확인 → 그 버전에 맞는 DFC로 HEF 다시 컴파일(필요 시), 예선 HEF로 동작 확인 | Pi의 HailoRT와 DFC 5.4.0 HEF 버전 불일치 → 버전에 맞춘 DFC 환경을 미리 준비 |
| 2주차 (11/13~11/19) | Pi에서 dev 전체·test 정확도, 배치 1 지연(워밍업 20, 200장 × 3회 × 3세션), 메모리·전력 측정. 호스트 전처리(크기 맞춤·정규화) 시간 포함 | Pi CPU의 전처리가 병목 → 전처리 최적화 또는 정규화를 HEF 안으로 이동 |
| 3주차 (11/20~11/26) | 추가 최적화: Hailo INT4 가중치(a8_w4, MLP부터), 16비트 층 축소 재시도, 컨텍스트 분할 조정. 예선 대비 변경 사항을 같은 판정 규칙(결과 전 고정)으로 평가 | INT4 정확도 손실, 16비트 축소판 compile 이상(예선 때 실칩 붕괴 사례) → 에뮬레이터·실칩 둘 다 확인 후 채택 |
| 4주차 (11/27~12/4) | 데모 앱(카메라 → 제로샷 분류), 최종 보고서(11/30)·발표 자료, 재현 코드 정리 | 발열에 따른 지연 변동 → 측정 전 온도 기록, 같은 조건 반복 |

---

## 8. 제출물 체크리스트

| No | 제출 항목 | 파일명 또는 링크 | 확인 |
|---|---|---|---|
| 1 | 결과 보고서 (PDF) | `[트랙2-예선]_자유_[팀명]_보고서.pdf` | □ |
| 2 | 베이스라인 모델 파일 | `siglip2_image_fp32.onnx` (371.67 MB, sha256 `333cad08e68dc85a…`) — Release `siglip2-xplat-20261015` | □ |
| 3 | 최적화 모델 파일 | `siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx` (96.18 MB, sha256 `e84c3727f51e3b97…`) — Release `siglip2-m2rot-ort-20261019` | □ |
| 4 | 모델 포맷 및 버전 정보 명시 | ONNX opset 17, 입력 `image` float32 N×3×224×224(SigLIP2 전처리 후), 출력 768차원 임베딩. ONNX Runtime 1.25.0 CPU, `ORT_ENABLE_ALL` | □ |
| 5 | 벤치마크 실행 스크립트 | `siglip2/s2_paired_ci.py`(dev), `siglip2/s2_test_eval.py`(test), `siglip2/s2_bench_speed.py`·`siglip2/scripts/cross_latency.py`(지연), `siglip2/s2_model_stats.py`(크기·메모리), `siglip2/scripts/portable_eval.py`(소규모 묶음) | □ |
| 6 | 의존성 목록 | `siglip2/requirements.txt` (+ `rn50/requirements.txt`) | □ |
| 7 | 실행 방법 README | `siglip2/README.md` | □ |
| 8 | 데이터셋 경로 또는 다운로드 방법 | 실행 때 자동 다운로드(`rn50/datasets.py`, 6.1절 출처). 소규모 묶음 `dev200/` (Release `siglip2-xplat-20261015`) | □ |

[확인 필요: 저장소 비공개 — 학술부 계정 초대 또는 공개 저장소/드라이브 링크, 심사 종료(11/5)까지 접근 유지. 코드 main 병합과 `prelim-submit` 태그]

---

## 9. 재현 실행 절차

### 9.1 환경 구성

```bash
git clone https://github.com/jungminmobile/oslab_mobile.git
cd oslab_mobile
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
python -m pip install -r siglip2/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# 모델 파일 받기 (GitHub CLI)
gh release download siglip2-xplat-20261015 --repo jungminmobile/oslab_mobile -p "siglip2_image_fp32.onnx" -p "dev200/*" -p "portable_eval.py" -D release
gh release download siglip2-m2rot-ort-20261019 --repo jungminmobile/oslab_mobile -D release
# 경로 (예시, 모두 저장소 밖)
export SIGLIP2_DATA=$HOME/siglip2_data SIGLIP2_ARTIFACTS=$HOME/siglip2_art HF_HOME=$HOME/hf_cache OPENBLAS_NUM_THREADS=1
mkdir -p $SIGLIP2_ARTIFACTS/onnx && cp release/*.onnx $SIGLIP2_ARTIFACTS/onnx/
```
[확인 필요: Release 자산 이름·폴더 구조를 실제 다운로드로 한 번 검증, Windows용 명령 병기]

### 9.2 벤치마크 실행 명령

**(가) 빠른 확인 — 소규모 묶음 (저장소 코드·데이터 다운로드 없이, 약 10분)**
```bash
python release/portable_eval.py --data release/dev200 --out out_quick --threads 4 \
  --models fp32=release/siglip2_image_fp32.onnx \
           int8=release/siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
```

**(나) 전체 정확도 — dev (Baseline·최적화 모델 함께, 짝 비교)**
```bash
cd siglip2
python s2_build_dev.py all --threads 4            # dev 캐시·텍스트 분류기·FP32 기준 (데이터 자동 다운로드, 처음 1~2시간)
RN50_RESULTS=../results/siglip2/repro_dev python s2_paired_ci.py --models \
  siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
```

**(다) 전체 정확도 — test 7종 (Baseline과 최적화 모델을 같은 실행에서)**
```bash
for s in pets caltech flowers cars cifar_test imagenetv2 food; do
  RN50_RESULTS=../results/siglip2/repro_test python s2_test_eval.py $s --threads 4 \
    --models siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
done
RN50_RESULTS=../results/siglip2/repro_test python s2_test_summary.py
```

**(라) 속도·메모리·크기 (Baseline·최적화 같은 실행에서 번갈아)**
```bash
python scripts/cross_latency.py --out out_speed --mem \
  --models b16_fp32=$SIGLIP2_ARTIFACTS/onnx/siglip2_image_fp32.onnx \
           b16_int8=$SIGLIP2_ARTIFACTS/onnx/siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
RN50_RESULTS=../results/siglip2/repro_stats python s2_model_stats.py --models fp32=siglip2_image_fp32.onnx \
  final=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
```

### 9.3 기대 출력

| 명령 | 출력 파일 | 기대값 | 보고서 위치 |
|---|---|---|---|
| (가) | `out_quick/result.json` | 고화질 5종(각 200장) cos 0.9756, Δ +0.3%p (x86 VNNI PC에서는 같은 값) | 6.3 |
| (나) | `paired_ci.jsonl` | FP32 91.66%, 최적화 91.44% (Δ −0.23 [−0.64, +0.18]), cos 0.976, CIFAR A −4.3 | 6.5 dev |
| (다) | `test.jsonl`, 요약 출력 | 고화질 5종 평균 90.34 → 90.00 (Δ −0.34 [−0.50, −0.18]), CIFAR-100 −4.09, ImageNetV2 −0.72 | 2, 6.5 test |
| (라) | `out_speed/cross_latency.json` | 최적화 모델이 FP32 대비 4스레드 약 −50%, 1스레드 약 −52~63% (절대 ms는 PC마다 다름). 피크 메모리 약 −52%. 크기 371.67 → 96.18 MB | 2, 6.5 효율 |

### 9.4 재현 시 유의사항

1. **채점은 배포한 ONNX 파일로 해 주십시오.**
   - 같은 명령으로 모델을 처음부터 다시 만들면, **같은 PC에서는 비트 단위로 같은 파일**이 나옵니다(새 가상환경에서 확인).
   - **CPU나 수학 라이브러리(OpenBLAS 커널)가 다르면** 회전 행렬 계산의 마지막 자리부터 달라져 파일이 달라지고, dev 정확도가 0.1%p 안팎 다를 수 있습니다(두 PC에서 −0.23 vs −0.31로 확인).
   - 만드는 순서와 명령은 `siglip2/README.md` 3절에 있습니다.
2. **지연 절대값은 PC·전원 설정에 따라 크게 다릅니다.** 노트북은 고성능 모드와 충전 상태에서, 다른 프로그램 없이, FP32와 같은 실행에서 비율로 비교해 주십시오.
3. **정수 커널 정확도는 CPU 명령어 집합에 따라 다를 수 있습니다.** 우리 파일은 x86 VNNI 세 PC(AVX-512 VNNI, AVX-VNNI)에서 같은 결과를 확인했습니다. 활성값 U8 + reduce_range는 VNNI가 없는 x86의 포화 문제도 막도록 고른 형식입니다.
4. **처음 실행 시간**: 데이터 다운로드와 dev 캐시 만들기에 1~2시간, test 7종 평가에 약 7시간(4코어 노트북)이 걸립니다. 빠른 확인은 9.2 (가)를 써 주십시오.
5. **별도 다운로드**: 모델 가중치(Hugging Face `timm/ViT-B-16-SigLIP2`, 약 1.5 GB)는 텍스트 분류기를 만들 때 처음 한 번 받습니다. ImageNetV2는 `scripts/test_g6.sh`가 받습니다.
6. 디스크 약 20 GB(데이터·캐시·ONNX), GPU는 필요 없습니다.

---

## 부록 A. 참고 문헌 [작성 필요: 형식 정리]
- Tschannen et al., "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features," 2025.
- Darcet et al., "Vision Transformers Need Registers," ICLR 2024.
- Sun et al., "Massive Activations in Large Language Models," 2024.
- Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs," 2024 / Liu et al., "SpinQuant," 2024.
- [레지스터 캐시 원 논문 — 확인 필요]
- Radford et al., "Learning Transferable Visual Models From Natural Language Supervision (CLIP)," 2021 — 프롬프트 템플릿.

## 부록 B. 이 초안에서 PDF로 옮길 때 확인할 것 (제출 전 지울 것)
- [ ] 1장 팀 정보, 3.1 저장소 접근·태그, 라이선스, Release 다운로드 명령 실제 검증
- [ ] 2장 표: 학술부는 원본 대비 개선율로 채점 — 변화율 열을 크게 강조
- [ ] 그림 후보: ① 민감도 지도(블록별 cos 막대) ② 거대 활성값 위치(토큰 × 채널 히트맵) ③ 2×2 단독 효과 막대 ④ FP32 vs 최적화 지연·크기·메모리 막대
- [ ] 15쪽 내외로 줄이기 (각 절 "※ 줄일 때")
- [ ] 레지스터 캐시·회전은 준용 실험(G3R·G3Q·G3RQ) — 팀원 역할 칸과 본문 표현 확인
