# G6c: B16 M2rot에 G6b의 속도 개선을 똑같이 적용해 B16 vs B32를 공정하게 비교 (설계 2026-10-21)

> 상태: **설계·판정 고정 (결과 전).** 1~4절은 결과 전에 커밋한다. 결과는 5절부터 붙인다.
> 브랜치 `siglip2/g6c-b16-speed` (`siglip2/g6b-speed` 3c5841c에서 만듦), worktree `~/oslab_g6c`. 담당: 용 (용 요청). **test는 쓰지 않는다.**
> 배경: `results/siglip2/G6b/README.md`. B32에서는 resources_param + 16비트 Zoo 집합(M_both)이 지연을 M +28.2% → +11.35%로 줄였고, 정확도는 M과 동률이었다.

## 1. 구성 (B16, 모두 B16 M2rot2와 같은 HAR·보정·optimization_level 1)

같은 재료 (G3RQ repro `run_info.txt`):
- parse된 HAR `~/siglip2_work/hailo_repro/siglip2_m2rot.har` (sha256 `ab7b324a…`, ONNX `siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad_b1.onnx` `d1df3f60…`)
- 보정 `~/siglip2_work/artifacts/dfc/calib_highres256_s0_nhwc.npy` (`91f551e3…`, 256장, 정규화된 값 −1~1)
- 기준 .alls `siglip2/hailo/g3r_M2rot_D256L1equiv.alls` (`485adb36…`)

| 이름 | 16비트 conv | resources_param | 컴파일 |
|---|---|---|---|
| B16 M2rot2 (G3RQ, 다시 컴파일 안 함) | `conv1`, `conv4, conv5, conv24, conv25, conv41` | 없음 | — (`040a3ae1…`, 110.4 MB, 21 ctx) |
| **B16_rb** | M2rot2와 같음 | Zoo `siglip_b_16_image_encoder.alls`(v5.4.0) 줄 `resources_param(max_compute_utilization=1.0, max_control_utilization=0.95, max_memory_utilization=0.75)` | 1 |
| **B16_both** | Zoo SigLIP1 B16 집합을 옮긴 것: `conv1, conv1_s2d`, `conv4, conv5` (블록 0 fc1·fc2), `conv24, conv25` (블록 5 fc1·fc2) | Zoo 줄 | 2 |

- 두 구성 모두 잔차 ew_add 26개는 M2rot2와 같은 목록이다 (Zoo `{ew_add*}`를 잔차 26개로 옮기는 G6b 규칙).
- **Zoo 집합 대응 (G6 대응표 방식):** M2rot 그래프의 conv 번호는 일반 그래프와 같다 (G6 `alls/check_vs_M2rot.log`: conv4·5 = 블록 0 fc1·fc2, conv24·25 = 블록 5 fc1·fc2, conv41 = 블록 9 fc2).
  - 그래서 B16_both는 M2rot2에서 **`conv41`(블록 9 fc2)을 빼고 `conv1_s2d`를 더한 것**이다.
  - `conv1_s2d`는 parse 단계 층이 아니라 optimize 때 생기는 층이다. Zoo S1B16 .alls와 G6 S1B16 R에서 같은 이름으로 컴파일됐다.
  - Zoo .alls의 `norm1`은 이 그래프에 없다. 정규화가 HEF 밖이라, M2rot2와 같은 입력 형식을 유지한다.
- **bucket 3줄을 뺀 이유 (용 확인):** Zoo S1B16 bucket 줄이 가리키는 층 대부분이 optimize 때 DFC가 새로 만드는 이름이다.
  - LN 분해 층 (`reduce_mean1_layer_normalization9`, `conv_var_inv_layer_normalization9` 등)
  - 정밀도 변환 층 (`precision_change15`)
  - `precision_change`의 번호는 16비트 층 집합에 따라 달라지고, M2rot 그래프에는 마스크 연산 층도 있다. 그래서 같은 블록 경계가 우리 그래프에서 어떤 이름이 될지 parse 단계에서 확인할 수 없다 → 애매함 → 뺀다. B16_both는 B32 M_both와 같은 꼴(16비트 Zoo 집합 + resources_param)이 된다.
- .alls 2개(`alls/g6c_b16_rb.alls`, `alls/g6c_b16_both.alls`)는 결과 전에 커밋하고, dev를 본 뒤 고치지 않는다. 컴파일 실패 때만 최소한으로 고치고 기록한다.
- 구현: G6 `g6_dfc.sh`에 기본값이 꺼진 옵션 두 개를 넣는다.
  - `HAR=` (parse를 건너뛰고 그 HAR에서 시작)
  - `CALIB_DOMAIN=normalized` (보정 값 −1~1 검사)
  - 옵션을 끄면 이전과 같은 경로다 (diff로 확인).

### 1-1. 계획에서 달라진 점: B16_rb 1차 compile 실패와 최소 수정 (결과 전, 용 결정)

- 1차 B16_rb (`alls/g6c_b16_rb.alls`, Zoo 줄 그대로 `max_compute_utilization=1.0`): optimize는 통과했다 (보정 256장 assert OK, 27분).
- compile은 116분 뒤 **실패**했다 (`logs/b16_rb_fail/compile_errors.txt`).
  - DFC가 그래프를 5개 컨텍스트로 나눈 뒤 (`Applying selected partition to 5 contexts`), `siglip2_m2rot_bucket_1 - Partition and Allocation Failed (Timeout, Processing time: 1h 55m 28s)`, `BackendAllocatorException: Compilation failed: Resolver didn't find possible solution.`
  - B32에서는 같은 줄로 컴파일됐다 (M_rb 12 컨텍스트). B16은 토큰이 3배라 같은 압축이 맞지 않은 것으로 보인다 (추정).
- **처리 (용 결정):**
  1. 설계대로 B16_both를 먼저 빌드한다. B16_both가 실패하면 다시 묻는다.
  2. B16_rb는 최소 수정으로 `max_compute_utilization`만 1.0 → **0.9**로 낮춘 `alls/g6c_b16_rb09.alls`로 다시 빌드한다 (`max_control_utilization=0.95`, `max_memory_utilization=0.75`와 다른 줄은 그대로). 이름은 **B16_rb09**로 적는다.
  - 비트 확인(2-1)과 속도 측정은 B16_rb 대신 B16_rb09로 한다.
  - B16_both의 resources_param은 그대로 1.0이다. B32 M_both와 같은 줄이다.

### 1-2. B16_both도 compile 실패 → 수동 bucket으로 compile만 다시 (결과 전, 용 지시)

- **B16_both** (`alls/g6c_b16_both.alls`): optimize는 통과했다 (256장, 29분). compile은 116분 뒤 **실패**했다.
  - B16_rb와 같은 모습이다: `Applying selected partition to 6 contexts` → 다시 5개로 나눔 → `siglip2_m2rot_bucket_1 - Partition and Allocation Failed (Timeout, Processing time: 1h 55m 31s)`, `Resolver didn't find possible solution.`
  - 근거: `logs/b16_both_fail/compile_errors.txt`
- B16_rb09(0.9)는 **보류**한다 (용 지시).
- **가설 (용):** B16은 B32보다 커서, resources_param 제한이 있으면 Zoo S1B16처럼 수동 bucket이 있어야 배치를 찾는다. bucket을 뺀 것이 시간 초과의 원인일 수 있다.
- **방법:**
  - B16_rb·B16_both 각각 **이미 optimize가 끝난 HAR을 그대로 쓰고**(다시 optimize하지 않음), 원래 .alls에 bucket 3줄만 더한 model script로 `hailo compiler --model-script`를 다시 돌린다 → **B16_rb_bk**, **B16_both_bk**.
  - compile 상한은 3시간이다 (넘으면 멈추고 보고).
- **bucket 대응 (Zoo S1B16 optimized HAR ↔ 우리 optimized HAR, 두 HAR에서 같은 이름):**
  - 근거: 층 종류, `original_names`(ONNX 노드), 그래프 이웃(입력·출력)
  - 잔차 덧셈은 검증된 .alls 26개 목록의 순서(R[0] = 위치 임베딩, R[2b+1] = 블록 b attention, R[2b+2] = 블록 b MLP, R[25] = 헤드)로 정했다. Zoo도 같은 규칙으로 번호가 매겨져 있다.

| bucket | Zoo 층 | 블록·연산 (Zoo 근거) | 우리 층 | 우리 근거 |
|---|---|---|---|---|
| b1 | `input_layer1` | 입력 | `input_layer1` | 입력 |
| b1 | `const_input1` | 위치 임베딩 상수 → conv1 (`/embeddings/Add_input`) | `const_input1` | → conv1, `/trunk/Add_input` |
| b1 | `conv17` | 블록 3 fc2 (`layers.3/mlp/fc2`) | `conv17` | fc2 순서 3번째, `/trunk/blocks.3/mlp/fc2`, 출력 → ew_add9 |
| b1 | `ew_add8` | 블록 3 attention 잔차 (`layers.3/Add`, 입력 conv15 = proj) | `ew_add8` | `/trunk/blocks.3/Add`, 입력 conv15 |
| b2 | `ew_add9` | 블록 3 MLP 잔차 (`layers.3/Add_1`, 입력 conv17) | `ew_add9` | `/trunk/blocks.3/Add_1`, 입력 conv17 |
| b2 | `reduce_mean1·ew_sub1·square1·conv_var_inv·ew_mult1 _layer_normalization9` | 블록 4 norm1 분해 (입력 ew_add9) | 같은 이름 5개 | `/trunk/blocks.4/norm1`, 입력 ew_add9 |
| b2 | `conv32`, `conv33` | 블록 7 fc1·fc2 | `conv32`, `conv33` | fc1·fc2 순서 7번째 |
| b2 | `ew_add17` | 블록 7 MLP 잔차 (입력 conv33) | **`ew_add26`** | R[16], `/trunk/Add_15`, 입력 conv33 |
| b3 | `const_input2` | 헤드 attention 상수 → matmul25 (`/head/attention/MatMul_2_input`) | **`const_input27`** | → matmul25, `/trunk/MatMul_14_input` (헤드 쪽 MatMul 상수는 이것 하나) |
| b3 | `… _layer_normalization17` ×5 | 블록 8 norm1 분해 (입력 ew_add17) | 같은 이름 5개 | `/trunk/norm1_3` (M 블록 5·6·7·8 중 4번째 = 블록 8), 입력 ew_add26 |
| b3 | `ew_add18` | 블록 8 attention 잔차 (입력 conv35) | **`ew_add29`** | R[17], `/trunk/Add_18`, 입력 conv35 (= 3 + 4·8, 블록 8 proj) |
| b3 | `precision_change15` | 블록 8 norm1 출력 16 → 8비트 (→ qkv splitter 3개) | **`precision_change16`** | ew_mult1_layer_normalization17 → conv_feature_splitter9 ×3, a16_w16_a8 |
| b3 | `conv53` | 헤드 MLP fc2 | `conv53` | fc2 순서 13번째 (헤드) |
| b3 | `ew_add26` | 헤드 잔차 (`/head/Add;Gather`) | **`ew_add57`** | R[25], `/trunk/Add_46;/trunk/Gather` |
| b3 | `output_layer1` | 출력 | `output_layer1` | 출력 |

- 만든 model script: `alls/g6c_b16_rb_bk.alls`, `alls/g6c_b16_both_bk.alls` (원래 .alls + bucket 3줄, 다른 줄은 그대로)
- M 그래프에만 있는 층(마스크 연산의 ew_mult·const_input 등)은 bucket 줄에 넣지 않았다. Zoo 줄도 경계 층만 나열하므로, 나머지는 DFC가 정하게 둔다.
- 대응을 못 찾은 층은 없다. 그래서 5번 대안(Zoo `clip_vit_b_16` .alls의 `context_switch_param`)은 쓰지 않는다.
- **비트 확인 (2-1을 바꿈):**
  - B16_rb_bk는 M2rot2 실칩 출력과 dev 6×200으로 비트 비교한다.
  - B16_both_bk는 B16_both 실칩 결과가 없다. 그래서 같은 optimized HAR의 에뮬레이터(SDK_QUANTIZED) 출력과 비교한다. G5b에서 에뮬레이터와 실칩이 비트 단위로 같지 않은 경우가 있었으므로, 비트 일치가 안 되면 cos·예측 일치율로 기록한다.

## 2. 측정

### 2-1. 정확도 (AGX 실칩, G6 `g6_hailo_chip.py`, G6 Zoo 전처리 캐시·HF FP32 기준·분류기)

- **B16_rb 비트 확인:** B16_rb와 B16 M2rot2의 출력을 dev 6×200(각 앞 200장)으로 비교한다. **비트 단위로 같으면** dev 전체 eval은 생략하고 M2rot2(G6b) 값을 쓴다.
- **B16_both:** dev 전체(고화질 4,923장 + CIFAR A 1,000장)
- 쌍 비교 95% 구간 (`s2_paired_ci.boot_hr`)
  - **B16_both − B16 M2rot2** (판정)
  - B16_both − B32 M_both (기록, 같은 사진)
  - (B16_rb가 M2rot2와 다르면) B16_rb − M2rot2

### 2-2. 속도 (G6b와 같은 규칙)

- HEF 6개를 같은 세션에서 번갈아 잰다: B16 M2rot2, B16_rb, B16_both, B32 M_both, B32 Z, S1B16 Z. 배치 1·8 × 3세션.
- 세션·HEF마다 칩 < 64.6 °C에서 시작한다. 95 °C를 넘으면 즉시 멈추고 보고한다.
- 보조로 `run2 -t 10`, parse-hef(컨텍스트·크기)를 낸다.
- **호스트 정규화 (용 지시):** B16 계열 HEF는 정규화가 HEF 밖이다. 그래서 AGX에서 호스트 정규화 시간을 따로 잰다 (`chip_hostnorm.py`).
  - 배치 1·8, 워밍업 20번 → 200번, 중앙값, 3세션
  - (a) eval 코드와 같은 구현: uint8 → `float32 / 127.5 − 1` (**표에 쓰는 값**)
  - (b) 참고: (a) + UINT8 양자화 (bench가 HEF에 넣는 형식)
- B32 M_both·Z와 S1B16 Z는 정규화가 HEF 안이라 호스트 정규화 시간이 없다.

### 2-3. HEF가 호스트에서 실행하는 연산 확인 (용 추가 지시, 판정 없음)

- 대상: B16 M2rot2, B16_rb, B16_both, B32 M_both, B32 Z (+ 참고 S1B16 Z)
- `chip_hefinfo.py`로 HEF마다 다음을 저장한다.
  - `hailortcli parse-hef --parse-streams --parse-vstreams` 전문
  - `hailo_platform.HEF` API의 vstream·stream 정보: 형식 type·order·flags, 모양, quant_info, data_bytes, nms_shape, 원래 층 이름
- 확인 방법
  - **후처리 연산(NMS 등):** vstream이 입출력 각 1개이고, `nms_shape`가 없고(NMS 순서가 아니면 HailoRT가 예외를 냄), parse-hef 전문에 ops·NMS 항목이 없으면 "없음"
  - **형식 변환:** vstream(사용자 쪽)과 stream(칩 쪽)의 format type·order·모양을 비교한다. 다르면 HailoRT가 변환을 한다. 변환이 어디서(AGX 호스트 CPU 또는 Hailo-10H 쪽) 도는지는 이 출력으로는 알 수 없다.
  - **정규화:** 입력 quant_info가 (scale 1, zp 0)이면 정규화가 HEF 안이고, 아니면 호스트에서 정규화가 필요하다 (2-2)
  - **사용자 형식 선택:** eval은 입출력을 FLOAT32로 두어 HailoRT가 양자화·역양자화를 한다. bench는 UINT8 입력과 HEF 기본 출력 형식(UINT16)을 쓴다. 이것은 HEF가 아니라 호출 쪽 선택이라 따로 적는다.

## 3. 판정 (결과 전 고정)

| 항목 | 규칙 |
|---|---|
| 16비트 축소 (B16) | B16_both − B16 M2rot2가 **"동률"**이면 **"B16에서도 회전 + sink로 16비트 축소 가능"**. "열위"면 "축소하면 손실", "우위"면 그대로 기록 |
| 쌍 비교 | 95% 구간 하한 > 0 → 우위, 상한 < 0 → 열위, 그 밖 → 동률 |
| 그 밖 | 판정하지 않는다. G6b 7절 "B16 유지 vs B32 전환" 표를 B16 개선판으로 갱신한다 ("HEF 지연"과 "HEF 지연 + 호스트 정규화" 두 줄) |

- 이 기준은 결과를 본 뒤 바꾸지 않는다.

## 4. 데이터 구간

- dev만 쓴다 (G6와 같은 Zoo 전처리 dev 전체). 보정은 기존 B16 M2rot2와 같은 파일(train highres 256장, 시드 0)이다. test는 쓰지 않는다.

## 5. 결과
(측정 후)
