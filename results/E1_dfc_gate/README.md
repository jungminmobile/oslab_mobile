# E1 DFC 호환성 관문 (진행 중)

환경: 정민 노트북 WSL2 Ubuntu-22.04, Hailo Dataflow Compiler **v5.4.0** (`~/hailo10_venv`), GPU 없음, WSL 메모리 23GB.
작업 폴더: WSL `~/hailo10_work/` (onnx·har·hef 같은 큰 파일은 커밋하지 않는다). 스크립트: `hailo/scripts/`.

| 단계 | 내용 | 상태 |
|---|---|---|
| 0 | 준비: 시험 모델 컴파일, RN50x4 float ONNX 변환, 보정 사진 64장 | ✅ 10/12 |
| 1 | 파싱 (`hailo parser onnx --hw-arch hailo10h`) | ✅ 10/12 |
| 2 | 양자화 (optimize, 기본 8비트, 64장, 정규화 층 포함) | ⏳ |
| 3 | 컴파일 (`.hef`) | ⏳ |
| 4 | 섞인 구성(c4 형태) `.alls` 컴파일 + 에뮬레이터 Pets dev 정확도 | ⏳ |

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
