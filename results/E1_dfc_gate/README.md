# E1 DFC 호환성 관문 (진행 중)

환경: 정민 노트북 WSL2 Ubuntu-22.04, Hailo Dataflow Compiler **v5.4.0** (`~/hailo10_venv`), GPU 없음, WSL 메모리 23GB.
작업 폴더: WSL `~/hailo10_work/` (onnx·har·hef 같은 큰 파일은 커밋하지 않는다). 스크립트: `hailo/scripts/`.

| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 준비: 시험 모델 컴파일, RN50x4 float ONNX 변환, 보정 사진 64장 | ✅ 10/12 |
| 1 | 파싱 (`hailo parser onnx --hw-arch hailo10h`) | ✅ 10/12 |
| 2 | 양자화 (optimize, 기본 8비트, 64장, 정규화 층 포함) | ✅ RN50·RN50x4 10/12 |
| 3 | 컴파일 (`.hef`) | ✅ RN50·RN50x4 10/12 |
| 4 | 섞인 구성(c4 형태) `.alls` 컴파일 + 에뮬레이터 Pets dev 정확도 | ⚠️ RN50 진행 중: c4가 Hailo(수준 0)에서 정확도 붕괴, 원인 layer4 |

## 0단계 (step0_smoke/)

- conv 1개짜리 임시 모델로 parse → optimize → compile 성공, `tiny.hef` 생성. DFC 설치 정상.
- `hailo profiler`가 있다. 단, 이 값은 컴파일러의 **추정치**다. 실제 FPS는 Pi에서 프로파일링을 끈 상태로 따로 잰다(런타임 프로파일러는 오버헤드가 크다).
- DFC 안내: 보정 사진이 정규화되지 않았으면 정규화를 NPU 밖(CPU)에서 해야 한다 → 2단계부터 Hailo 공식 설정처럼 모델 앞에 `normalization` 층을 넣고, 보정 사진은 0~255 원본(NHWC uint8)으로 준다.
- 보정 사진: 고화질 5종 보정 후보(311장)에서 시드 0으로 64장. RN50용 224×224, RN50x4용 288×288 (`quantize_onnx.py`의 `calib_images`와 같은 방식).
- RN50x4 float ONNX(q1)를 노트북에서 변환했다. 크기 348,386,042 B로 클라우드 파일과 같다.

## 1단계 결과 (step1_parse/) — **두 모델 모두 끝까지 파싱 성공**

| 모델 | 시작 노드 | 끝 노드 | DFC 층 수 | 걸린 시간 | 최대 메모리 |
|---|---|---|---|---|---|
| RN50 | `image` | `/attnpool/Gather_11` (최종 출력) | 95 (conv 59, ew_add 17, matmul 2, softmax 1 …) | 21초 | 2.8 GB |
| RN50x4 | `image` | `/attnpool/Gather_11` (최종 출력) | 135 (conv 89, ew_add 27, matmul 2, softmax 1 …) | 26초 | 4.7 GB |

- **AttnPool까지 NPU 층으로 변환됐다.** CPU로 떼어 낼 부분이 없다(PRD 10절의 최대 위험 해소).
- RN50 conv 59개 = stem 3 + 병목 블록 16×3 + 다운샘플 4 + AttnPool 투영 4.

### AttnPool ↔ DFC 층 이름 (RN50 기준, 전체는 `hailo/layer_map_rn50.csv`)

| ONNX (AttnPool) | 역할 | DFC 층 | RN50x4 |
|---|---|---|---|
| `/attnpool/MatMul` | q 투영 | `conv58` | `conv88` |
| `/attnpool/MatMul_1` | k 투영 | `conv56` | `conv86` |
| `/attnpool/MatMul_2` | v 투영 | `conv57` | `conv87` |
| `/attnpool/Gemm` | c_proj (출력 투영) | `conv59` | `conv89` |
| `/attnpool/MatMul_3`, `Softmax`, `MatMul_4` | attention 점수·softmax·가중합 | `matmul1`, `softmax1`, `matmul2` | 같음 |
| `/attnpool/Mul_2`, `Mul_3` | 스케일 | `mul_and_add1·2` (normalization) | 같음 |

- Hailo 공식 RN50 설정의 `conv56~58 = a8_w8`은 **q·k·v 투영**이다(이전 분석의 추정이 확인됨). 공식 설정은 c_proj(`conv59`)를 포함한 나머지를 16비트로 둔다.
- 우리 구성 매핑: "AttnPool 투영 INT8" = `conv56~59` `a8_w8`, "attention 계산 FP32" = `matmul1·2`, `softmax1`, `mul_and_add1·2` `a16_w16`.

## 2단계 결과 (step2_optimize/) — RN50 양자화 성공

- 설정: `hailo/scripts/e1_base.alls` = 정규화 층(CLIP 평균·표준편차×255) + `optimization_level=0, compression_level=0` + 보정 64장. 정밀도는 DFC 기본(전부 8비트).
- 걸린 시간 **1분 12초**, 최대 메모리 4.0GB. 보정 통계 수집 15초.
- GPU가 없어서 최적화 수준 0으로 돌았다. 고급 정확도 보정(Finetune, Bias Correction, AdaRound, QFT)은 모두 건너뜀. DFC 경고: "수준 0은 실제 배포에 권장하지 않음" → 정확도는 4단계 에뮬레이터로 확인한다.
- DFC 안내: 파라미터가 1000만 개 이상이면 `compression_level=4`(4비트 압축)를 권장 → E9(4비트) 검토 대상.

## 3단계 결과 (step3_compile/) — **RN50 `.hef` 생성 성공 (Hailo-10H)**

| 항목 | 값 |
|---|---|
| `.hef` 크기 | 28.2 MB (SHA-256은 `rn50_hef.sha256`) |
| 컴파일 시간 | 10분 28초 (분할·배치 9분 57초), 최대 메모리 4.1GB |
| 컨텍스트 수 | **5개** (한 번에 올리지 못해 5조각으로 나눠 실행) |

- 한 번에 올리기(single context)는 자원 부족으로 실패했다. 필요/가용 예: 계산 단위(SC) 444/180, APU 108/90, L3 가중치 1162/480, LCU 107/60. 그래서 컴파일러가 자동으로 **5개 컨텍스트**로 나눴다. 컨텍스트 전환 비용만큼 FPS가 낮아질 수 있다.
- `hailo profiler`(DFC, 오프라인): 계산량 5.86 GMAC, 파라미터 38.25M, 컨텍스트 5개. **멀티 컨텍스트 모델은 처리량·지연이 `N/A`로 나온다** → 오프라인 FPS 추정은 불가. FPS는 본선 Pi에서 프로파일링을 끈 상태로 직접 잰다(PRD C6 수정 필요).
- 참고: `hailo profiler`는 `--hw-arch` 옵션을 받지 않는다(HAR에 이미 들어 있음). `e1_step3.sh`의 profiler 호출을 고쳐야 한다.

## RN50 vs RN50x4 (2·3단계, 같은 설정 `e1_base.alls`)

| 항목 | RN50 | RN50x4 |
|---|---|---|
| 입력 | 224×224×3 | 288×288×3 |
| 파라미터 / 계산량 | 38.25M / 5.86 GMAC | 87.02M / 20.70 GMAC |
| 양자화 시간 (최대 메모리) | 1분 12초 (4.0GB) | 2분 3초 (7.5GB) |
| 컴파일 시간 (최대 메모리) | 10분 28초 (4.1GB) | 29분 45초 (5.7GB) |
| 한 번에 올리기 필요/가용 계산 단위(SC) | 444 / 180 | 909 / 180 |
| **컨텍스트 수** | **5** | **11** |
| `.hef` 크기 | 28.2 MB | 60.7 MB |
| profiler 처리량 | N/A (멀티 컨텍스트) | N/A (멀티 컨텍스트) |

- **두 모델 모두 Hailo-10H용 `.hef` 생성 성공 → E1 관문(전부 8비트 기준) 통과.** 남은 것은 4단계(섞인 구성 + 에뮬레이터 정확도).
- RN50x4는 컨텍스트가 2배 이상 많아(11개) Pi에서 컨텍스트 전환 비용이 클 것으로 예상된다. Hailo 공식 수치도 RN50 142 FPS vs RN50x4 62.6 FPS로 차이가 크다. E6에서 모델을 고를 때 반영한다.

## 4단계 (step4_mixed/) — RN50 c4: **Hailo에서 정확도 붕괴 발견** (진행 중)

에뮬레이터: DFC `InferenceContext` (FP = `SDK_FP_OPTIMIZED`, 양자화 = `SDK_QUANTIZED`), Pets dev 앞 16장, 노트북 CPU. 스크립트 `hailo/scripts/emu_eval.py`.
기준: 같은 16장의 ORT FP32 정답률 81.25%. cos = ORT FP32 임베딩과의 유사도.

| 설정 (`hailo/scripts/`) | 8비트로 둔 부분 | top-1 (16장) | cos |
|---|---|---|---|
| FP 에뮬레이션 (정규화 층 확인) | — | 81.25 | **1.000** |
| `e1_base.alls` | 전부 | 0.0 | 0.349 |
| `rn50_c4.alls` (예선 c4와 같게) | layer2~4, AttnPool 투영 | 0.0 | 0.413 |
| c4 + equalization 강제 (`rn50_c4eq`) | 같음 | 0.0 | 0.413 (변화 없음) |
| c4 + c_proj 16비트 (`rn50_c4cp16`) | layer2~4, q·k·v | 0.0 | 0.414 (변화 없음) |
| Hailo 공식 설정 (`rn50_official`) | q·k·v만 | 75.0 | **0.939** |
| 공식 + layer2만 8비트 (`rn50_off8L2`) | layer2, q·k·v | 81.25 | 0.849 |
| 공식 + layer3만 8비트 (`rn50_off8L3`) | layer3, q·k·v | 56.25 | 0.892 |
| 공식 + **layer4만 8비트** (`rn50_off8L4`) | layer4, q·k·v | 6.25 | **0.395** |

- **원인은 layer4를 8비트로 바꾸는 것**이다. ORT에서는 layer4만 INT8이어도 cos 0.990으로 멀쩡했다(archive R1). 그러나 Hailo(최적화 수준 0, GPU 없음)에서는 layer4 8비트만으로 붕괴한다. → **"예선 구성 = 본선 구성" 가정이 지금 조건에서는 깨진다** (PRD 4-2 함정 2가 실제로 확인됨).
- equalization을 모델 스크립트로 강제로 켜도 결과가 같았다. 수준 0에서는 적용되지 않는 것으로 보인다.
- **DFC 5.4.0 제약**: 데이터×데이터 matmul(attention 점수·가중합)은 입력 16비트를 지원하지 않는다(`a8_w8`, `a8_w8_a8`, `a8_w8_a16`만). softmax는 DFC가 reduce_max·ew_sub·reduce_sum·ew_mult로 나누고, 스케일(mul_and_add)은 앞 conv에 합친다. → ORT의 "attention 계산 FP32"는 그대로 옮길 수 없다.
- `hailo analyze-noise`(층별 노이즈 분석)는 conv56(zp_comp 채널 2048 vs 2080)에서 오류로 멈췄다(`analyze_noise_crash.log`).
- 16장 표본이라 정답률은 흔들림이 크다. 판단은 cos 위주로 했다.
- c4 `.hef` 컴파일은 정확도 문제가 정리될 때까지 보류.
