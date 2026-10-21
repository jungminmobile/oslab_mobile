# G6c: B16 M2rot에 G6b의 속도 개선을 똑같이 적용해 B16 vs B32를 공정하게 비교 (설계 2026-10-21)

> 상태: **완료 (10/21).** 1~4절은 결과 전에 고정했다 (04679f4, 사후 기록 1-1·1-2는 각 빌드 전에 커밋 af38904·855e99f). 결과는 5절부터다.
> **한 줄 결론:**
> - B16에 resources_param을 넣으면 Zoo처럼 **수동 bucket이 있어야** compile된다.
> - **B16_rb_bk**는 M2rot2와 출력이 비트 단위로 같고, 배치 1 지연은 47.55 → **40.51 ms (−14.8%)**, 컨텍스트는 21 → 16이다.
> - 16비트를 Zoo 집합으로 줄인 B16_both_bk는 실칩에서 무너졌다 (50.68%). 같은 HAR의 에뮬레이터는 정상이라 compile 결과물의 이상으로 보며, 원인은 확인하지 못했다.
> - B16 개선판도 B32 M_both(18.4 ms)보다 2.2배 느리고, 정확도는 +2.34 [+1.62, +3.06]%p 높다.
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
  - (사후 확인, 독립 검토) optimized HAR을 보면 `conv1_s2d`는 B16_rb에서도 이미 a16_w16_a16이었다. 그래서 실제 차이는 **conv41의 8비트화뿐**이다 (5-2 참고).
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

### 5-0. 판정 (3절 규칙)

| 항목 | 결과 | 판정 |
|---|---|---|
| 16비트 축소 (B16) | B16_both_bk − B16 M2rot2 = **-40.57 [-42.03, -39.14]%p** | 규칙상 **열위 ("축소하면 손실")** — **다만 측정한 HEF가 자기 양자화 모델과 다르게 동작한다 (5-2).** 이 값은 16비트 축소의 효과가 아니라 compile 결과물의 이상으로 보이므로 **16비트 축소에 대한 근거로 쓰지 않는다** (판정 문구는 팀 확인 필요) |

- **정확도를 지킨 B16 개선판은 B16_rb_bk다** (resources_param 1.0 + 수동 bucket). 출력이 M2rot2와 비트 단위로 같고, 배치 1 지연은 47.55 → **40.51 ms (−14.8%)**, 컨텍스트는 21 → 16이다.

### 5-1. 구성표 (`build.jsonl`, `chip/hefinfo/parsehef_full_*.txt`)

| HEF | 만든 방법 | HEF 크기 B | 컨텍스트 | HEF sha256 (앞 16) | 결과 |
|---|---|---|---|---|---|
| B16 M2rot2 | G3RQ repro | 110,436,352 | 21 | `040a3ae1ebd81d99` | 기준 |
| B16_rb (1차) | M2rot2 .alls + resources_param 1.0 | — | (5로 분할) | — | **compile 실패** (할당 시간 초과 116분) |
| B16_both (1차) | Zoo 16비트 + resources_param 1.0 | — | (6 → 5로 분할) | — | **compile 실패** (할당 시간 초과 116분) |
| **B16_rb_bk** | 1차 B16_rb의 optimized HAR(`1cdb4a50…`) + bucket 3줄로 compile만 | 110,886,912 | **16** | `01fa7e3d70e236e2` | compile 81분 성공 |
| **B16_both_bk** | 1차 B16_both의 optimized HAR(`7d22df79…`) + bucket 3줄로 compile만 | 108,154,880 | **16** | `32accb8fb6a8d5f6` | compile 79분 성공 |
| B16_rb09 | resources_param 0.9 | — | — | — | **보류** (용 지시) |

- bk 빌드의 시작 기록: B16_rb_bk는 `dirty=0`, B16_both_bk는 `dirty=1`이다 (`build.jsonl`).
  - 원인: 앞 커밋에 들어간 `build.jsonl`에 실행 기록이 덧붙었기 때문이다 (`git status`: `M results/siglip2/G6c/build.jsonl`만).
  - 코드·.alls·스크립트는 커밋 `855e99f` 그대로다.

### 5-2. 정확도 (`results_chip.jsonl`, 실칩 dev 전체 5,923장, G6 eval 코드·분류기)

| HEF | 고화질 top-1 | FP32 대비 Δ [95% 구간] | cos 평균 / 최소 | CIFAR A (Δ) |
|---|---|---|---|---|
| B16 M2rot2 (G6b) | 91.26 | -0.41 [-0.84, +0.05] | 0.9862 / 0.262 | 75.2 (-2.0) |
| B16_rb_bk | = M2rot2 (dev 6×200, 1,200장 모두 비트 일치 → 2-1 규칙대로 전체 eval 생략) | | | |
| **B16_both_bk** | **50.68** | **-40.98 [-42.42, -39.50]** | **0.4028** / -0.010 | 39.2 (-38.0) |
| B32 M_both (G6b) | 88.91 | -1.05 [-1.57, -0.52] | 0.9835 / 0.242 | 74.4 (-0.8) |

- 쌍 비교
  - B16_both_bk − M2rot2 = -40.57 [-42.03, -39.14] (열위)
  - (기록) B16_both_bk − B32 M_both = -38.23 [-39.77, -36.80]

**B16_both_bk가 무너진 원인 확인 (에뮬레이터, `emu_check.json`, dev 6×200, `g6c_emulate.py` SDK_QUANTIZED):**

| 데이터 | B16_both 에뮬레이터 cos (FP32 대비) | **B16_both_bk 실칩 cos** | cos(에뮬레이터, 실칩) | 에뮬레이터 top-1 | **실칩 top-1** | FP32 top-1 | (대조) B16_rb 에뮬레이터 cos |
|---|---|---|---|---|---|---|---|
| cifar_A | 0.9607 | **0.3072** | 0.3123 | 69.0 | **35.0** | 74.0 | 0.9607 |
| pets_dev | 0.9897 | **0.4409** | 0.4393 | 92.5 | **53.5** | 93.5 | 0.9897 |
| flowers_dev | 0.9882 | **0.3680** | 0.3690 | 86.5 | **40.5** | 85.0 | 0.9882 |
| cars_dev | 0.9868 | **0.4168** | 0.4144 | 93.5 | **49.5** | 93.5 | 0.9868 |
| food_dev | 0.9838 | **0.4036** | 0.4040 | 87.0 | **43.5** | 87.5 | 0.9838 |
| caltech_dev | 0.9850 | **0.3910** | 0.3909 | 95.0 | **78.5** | 95.0 | 0.9850 |

- **양자화 모델(같은 optimized HAR의 에뮬레이터)은 정상이다.** FP32 대비 cos 0.96~0.99이고 top-1도 FP32와 비슷하다.
- **compile된 HEF만 무너졌다** (cos 0.31~0.44). 그래서 16비트 집합 자체가 아니라 compile 결과물의 문제로 본다.
- 대조
  - B16_rb의 에뮬레이터와 M2rot2 실칩도 비트 단위로는 다르다 (cos(에뮬레이터, 실칩) 0.977~0.991). 그래서 B16_both_bk의 "비트 동일" 확인은 이 방법으로는 할 수 없다. 대신 정확도 수준으로 판단했다.
  - B16_rb에서는 에뮬레이터와 실칩의 FP32 대비 cos가 같은 수준이다 (예: Pets 0.9897 대 0.9899).
- 같은 bucket 줄을 쓴 B16_rb_bk는 실칩에서 M2rot2와 비트 단위로 같았다.
- **두 optimized HAR의 실제 차이 (독립 검토로 찾고 HAR에서 확인):**
  - `conv1_s2d`는 둘 다 a16_w16_a16이다 (B16_rb의 .alls에 없어도 16비트) → 차이가 아니다.
  - B16_rb: `conv41`(블록 9 fc2, a16_w16_a16)에 블록 9 MLP 잔차 덧셈 `/trunk/Add_27`이 **합쳐져 있다** (`conv41` 입력 `conv40, ew_add36`).
  - B16_both: `conv41`이 a8_w8_a16이 되고, `conv40` 출력은 a8이 된다. 그러면서 `/trunk/Add_27`이 **별도 층 `ew_add38`**(a16_w16_a16)로 떨어져 나왔다 (488 → 489층). `ew_add36`·LN21·`ew_add43`의 연결도 이 층을 거치도록 바뀌었다.
  - 그래서 B16_both는 사실상 "M2rot2 − conv41 16비트"다. 떨어져 나온 `ew_add38`(bucket 경계 밖)이 HEF 이상의 가장 구체적인 후보다. 다만 그것이 원인인지는 **확인 못 함**.

### 5-3. 속도 (`results_speed.jsonl`, `chip/bench/`, HEF 6개를 같은 세션에서 번갈아, 배치 1·8 × 3세션)

- AGX Orin + Hailo-10H, HailoRT 5.3.0, G6 최종 `run.sh`·`g6_hailo_chip.py` (sha256 `f6439972…`·`70a5de57…`)
- 세션·HEF마다 칩 < 64.6 °C에서 시작했다 (최고 시작 64.58 °C, `run_g6c.log`의 49번).
- B16_rb_bk 200장 비트 확인(07:44, run.sh 밖에서 직접 실행)은 로그가 `eval_info.json`과 `temps_g6c.jsonl`의 `pre_rbbk` 48.9 °C → `post_rbbk` 77.4 °C뿐이다. 정확도 출력이라 온도와 무관하다.
- **최고 온도 94.18 °C**는 B16_both_bk dev 전체 eval이 끝난 직후였다. 속도 측정 중 최고는 88.8 °C였다. 95 °C는 넘지 않았다.
- 세 세션 사이 차이는 배치 1에서 모든 HEF가 0.3% 이내, 배치 8에서 1.6% 이내다 (B16 M2rot2 1.51%).
- **호스트 정규화** (`results_hostnorm.json`, AGX CPU, 세 세션 중앙값의 중앙값)
  - (a) eval과 같은 구현(`float32 / 127.5 − 1`): 배치 1 **0.180 ms**, 배치 8 **1.322 ms**
  - (b) 참고: + UINT8 양자화 (bench가 HEF에 넣는 형식, 세션 1 배치 1은 0.835 ms로 튀었고 세션 2·3은 0.53 ms): 배치 1 0.529 ms, 배치 8 6.579 ms
  - 처음에 zp를 127로 잘못 넣고 쟀다 (실제 quant_info zp는 128). 시간에는 영향이 없지만 128로 다시 쟀고, 잘못 넣은 기록은 `chip/hostnorm_g6c_zp127_wrong.json`에 남겼다.
- 표의 "+ 호스트 정규화"는 HEF 지연 + (a)이다. 정규화가 HEF 안인 HEF(B32·S1B16 Z)는 0이다.

| HEF | 배치 1 HEF 지연 ms | 배치 1 + 호스트 정규화 | M2rot2 대비 | run2 FPS b1 | 배치 8 ms/묶음 | 배치 8 + 정규화 | 배치 8 장/초 | run2 FPS b8 | 고화질 top-1 |
|---|---|---|---|---|---|---|---|---|---|
| B16 M2rot2 | 47.55 | 47.73 | +0.0% | 21.6 | 241.01 | 242.34 | 33.2 | 41.7 | 91.26 |
| **B16_rb_bk** | 40.51 | 40.69 | -14.8% | 25.6 | 178.41 | 179.74 | 44.8 | 60.8 | = M2rot2 (dev 6×200 비트 일치) |
| B16_both_bk | 35.90 | 36.08 | -24.5% | 29.0 | 152.51 | 153.83 | 52.5 | 72.0 | 50.68 (HEF 이상, 5-2) |
| B32 M_both (G6b) | 18.38 | — | -61.3% | 59.1 | 62.12 | — | 128.8 | 183.1 | 88.91 |
| B32 Z | 16.48 | — | -65.3% | 64.5 | 53.77 | — | 148.8 | 220.7 | 85.74 |
| S1B16 Z | 31.49 | — | -33.8% | 32.9 | 137.69 | — | 58.1 | 77.6 | 89.61 |

### 5-4. HEF가 호스트에서 실행하는 연산 (`chip/hefinfo/`, 판정 없음)

| HEF | 후처리 연산 (NMS 등) | 입력 vstream (사용자) | 입력 stream (칩) | 입력 quant_info (scale, zp) | 정규화 위치 | 출력 vstream / stream | 출력 quant_info (scale, zp) |
|---|---|---|---|---|---|---|---|
| B16 M2rot2 | 없음 | UINT8 NHWC 224×224×3 | UINT8 **NHCW** | 0.007874, 128 | **HEF 밖 (호스트)** | UINT16 NHWC / NHCW 1×1×768 | 0.000303, 8156 |
| B16_rb_bk | 없음 | 같음 | 같음 | 0.007874, 128 | HEF 밖 | 같음 | 0.000303, 8156 |
| B16_both_bk | 없음 | 같음 | 같음 | 0.007874, 128 | HEF 밖 | 같음 | 0.000303, 8156 |
| B32 M_both | 없음 | UINT8 NHWC 256×256×3 | UINT8 **NHCW** | 1.0, 0 | HEF 안 (`norm1`) | UINT16 NHWC / NHCW | 0.000331, 7973 |
| B32 Z | 없음 | UINT8 NHWC 256×256×3 | UINT8 **NHCW** | 1.0, 0 | HEF 안 | UINT16 NHWC / NHCW | 0.000296, 8373 |
| S1B16 Z (참고) | 없음 | UINT8 NHWC 224×224×3 | UINT8 **NHCW** | 1.0, 0 | HEF 안 | UINT16 NHWC / NHCW | 0.000326, 9506 |

- **확인 방법과 근거 줄** (예: `parsehef_full_b16_both_bk.txt`)
  - 후처리 연산: `VStream infos:` 아래 줄이 `Input … UINT8, NHWC(224x224x3)`와 `Output … UINT16, NHWC(1x1x768)` 두 줄뿐이다. parse-hef 전문에 NMS·ops 항목이 없다 (`parsehef_mentions_ops: []`). HEF API `nms_shape`는 6개 모두 "NMS 순서 아님" 예외다 (`any_nms: false`). → 후처리 연산 **없음**.
  - 형식 변환: `Stream infos:`는 `Input … UINT8, NHCW(224x224x3)`·`Output … UINT16, NHCW(1x1x768)`, `VStream infos:`는 `NHWC`다. 6개 HEF **모두 순서가 다르다** (`in_vstream_eq_stream: false`, `out_vstream_eq_stream: false`). 그래서 HailoRT가 입력은 NHWC → NHCW, 출력은 NHCW → NHWC로 재배치한다. 이 변환은 Zoo HEF를 포함한 모든 HEF에 똑같이 있고, bench 지연(`run_async` → `wait`)에 들어간다. **변환이 AGX 호스트 CPU에서 도는지 Hailo-10H 쪽에서 도는지는 이 출력으로 알 수 없다 (확인 못 함).** stream과 vstream의 형식 type(UINT8·UINT16)과 모양은 같다.
  - 정규화: 입력 quant_info가 (1.0, 0)이면 픽셀을 그대로 넣는다 (정규화가 HEF 안). B16 계열은 (0.007874 = 1/127, 128)이라 호스트에서 `x/127.5 − 1`(정규화)과 양자화가 필요하다 (5-3).
  - 호출 쪽 선택: eval은 입출력 FLOAT32로 HailoRT가 양자화·역양자화를 하고, bench는 UINT8 입력과 UINT16 출력을 쓴다. HEF가 정하는 것이 아니라 호출하는 쪽의 선택이다.

### 5-5. B16 유지 vs B32 전환 (G6b 7절 갱신, 판정 없음)

| 경로 | 모델 | 고화질 top-1 | FP32 대비 Δ | CIFAR A | 배치 1 **HEF 지연** | 배치 1 **HEF 지연 + 호스트 정규화** | 배치 8 장/초 | 크기 |
|---|---|---|---|---|---|---|---|---|
| Hailo | B16 M2rot2 (이전) | 91.26 | −0.41 | 75.2 | 47.55 ms | 47.73 ms | 33.2 | 110.4 MB |
| Hailo | **B16_rb_bk (개선판)** | **91.26** (= M2rot2) | −0.41 | 75.2 | **40.51 ms** | **40.69 ms** | 44.8 | 110.9 MB |
| Hailo | B32 M_both (G6b) | 88.91 | −1.05 | 74.4 | 18.38 ms | 18.38 ms | 128.8 | 112.0 MB |
| Hailo | B32 M (G6) | 89.12 | −0.84 | 74.7 | 21.16 ms (G6b) | 21.16 ms | 91.5 (G6b) | 113.0 MB |
| ORT CPU (G6b, 비공식, t4) | B16 M2rot_ort | 91.44 | −0.23 | 72.6 | 29.7 ms | — | — | 91.7 MiB |
| ORT CPU (G6b, 비공식, t4) | B32 INT8 | 89.09 | −0.87 | 75.4 | 13.6 ms | — | — | 98.1 MiB |

- Hailo 정확도 차이 (같은 사진의 짝 비교, G6b): B16 − B32 M = +2.14 [+1.48, +2.84]%p. B16_rb_bk는 M2rot2와 출력이 같으므로 같은 값이다. B16 M2rot2(= B16_rb_bk) − B32 M_both = **+2.34 [+1.62, +3.06]%p** (CIFAR +0.8, `results_chip.jsonl`의 `b16_M2rot2-b32_M_both` 줄, B16_rb_bk가 6×200에서 비트 일치하므로 M2rot2 dev 전체 임베딩 사용).
- Hailo 지연: B16_rb_bk 40.5 ms 대 B32 M_both 18.4 ms → B32가 **0.45배**다 (호스트 정규화를 넣으면 0.45배로 같음). 개선 전 B16 대비 B32는 0.39배였다.
- 정확도와 지연의 교환점은 팀이 정한다.

## 6. 계획과 달라진 점

1. bucket 3줄을 처음에는 뺐다 (설계 1절). 1차 B16_rb·B16_both가 둘 다 compile에 실패했고 (1-1, 1-2), 용의 가설에 따라 **수동 bucket**(Zoo S1B16 대응, 1-2 표)으로 **compile만 다시** 했다. 둘 다 compile에 성공했다.
2. B16_rb09(resources_param 0.9)는 보류했다 (용 지시).
3. B16_both_bk의 비트 확인은 에뮬레이터로 했다. 대조군에서 에뮬레이터와 실칩이 비트 단위로 다르다는 것을 확인했고, 정확도 수준으로 판단했다 (5-2).
4. 호스트 정규화 측정에서 zp 값을 처음에 잘못 넣었다 (5-3, 기록 남김).
5. 판정 문구: 규칙상 "열위"이지만 HEF 이상의 증거가 있어 "근거로 쓰지 않음"으로 적었다. 팀 확인이 필요하다.

## 7. 확인 못 한 것

- B16_both_bk HEF가 자기 양자화 모델과 다르게 동작하는 원인
- NHWC ↔ NHCW 재배치가 어디서(AGX 호스트 또는 Hailo-10H 쪽) 도는지
- B16_rb09(0.9)와, 수동 bucket 없이 resources_param을 완화했을 때의 결과
- B16_rb_bk의 dev 전체 비트 일치 (6×200만 확인)
- 빌드 사이 변동

## 8. 3중 확인 기록

1. 자기 검증: B16_rb_bk와 M2rot2의 6×200 비트 일치, 에뮬레이터 대조군(B16_rb), bucket 대응 근거(층 종류·ONNX 이름·이웃), 보정 256장 assert
2. 독립 재계산: 정확도·쌍 구간·속도를 다른 경로(검토자 자체 코드)로 다시 계산 → 모두 일치
3. 독립 검토 (맥락 없는 서브에이전트) — 반영한 지적
   - (a) B16_both의 실제 차이(conv41만, `ew_add38` 분리)를 1·5-2절에 추가
   - (b) `results_chip.jsonl`의 dirty true → 커밋된 트리에서 다시 정리 (아래 9절)
   - (c) dirty 설명 모순 정정
   - (d) B16_rb_bk 200장 확인의 온도 기록
   - (e) 배치 8 차이 1.6%, 호스트 정규화 (b) 튄 값

## 9. 결과 파일의 커밋 기록

- `results_chip.jsonl`은 정리 스크립트(`g6c_summarize.py`, B16−B32 쌍 포함)를 커밋한 뒤, 깨끗한 트리에서 다시 만들었다. 모든 줄이 `dirty: false`다.
  - 처음 만든 파일은 `build.jsonl`에 기록이 덧붙은 상태에서 만들어 `dirty: true`였다. 숫자는 같다 (독립 검토가 다시 계산해 확인).
- 칩 측정 스크립트는 G6 커밋 `43a3728`의 `g6_hailo_chip.py`·`run.sh`와 sha256이 같다.
