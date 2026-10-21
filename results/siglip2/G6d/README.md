# G6d: B32 Hailo 속도 후보 한 번에 빌드·측정 + B16으로 옮길 수 있는 것 가리기 (설계 2026-10-21)

> 상태: **완료 (10/21).** 1~4절은 결과 전에 고정했다 (e9e4c40, 1-1은 Dmax2 빌드 전 2f7438f). 결과는 5절부터다.
> **한 줄 결론:** B32 성공 구성은 둘이다.
> - **G**(16비트 conv 최소): 15.66 ms, Z 대비 −5.0%, Z보다 정확도 +2.67%p 우위
> - **F**(M_both + Zoo `allocator_param`·`context_switch_param`, compile만): 16.54 ms, Z 대비 +0.3%, 출력은 M_both와 비트 동일 → +3.17%p 우위
> - 남은 +11%의 원인은 그래프 출처가 아니라 sink 마스크 연산이다. **B16에 가장 먼저 옮길 후보는 F**(compile 설정만 바꿈)다. B16_rb_bk_D는 조건 미달로 건너뛰었다.
> 브랜치 `siglip2/g6d-speed2` (`siglip2/g6c-b16-speed` d0d3b7d에서 만듦), worktree `~/oslab_g6d`. 담당: 용 (용 요청). **test는 쓰지 않는다.**
> 목적: (1) B16_rb_bk에 옮길 속도 개선 찾기 (팀은 B16 유지 가능성이 크다), (2) 보고서용 B32 부록.
> 배경: B32 Z 16.51 ms, M_both 18.38 ms (+11.35%), M_both − Z = +3.17 [+2.41, +3.95]%p (G6b). B16_rb_bk 40.51 ms (G6c). Zoo S1B16 31.5 ms.

## 1. 구성 (B32)

| 순서 | 이름 | ONNX | .alls (`alls/`) | 보정 | 만드는 방법 |
|---|---|---|---|---|---|
| 1 | **Dmax** | (M_both HAR) | `g6d_b32_Dmax.alls` = M_both + `performance_param(compiler_optimization_level=max)` | — | M_both의 optimized HAR에서 **compile만** |
| 2 | **S1** | G6 R (`g6_b32_rot_b1.onnx`, 회전만) | `g6d_b32_S1.alls` = Zoo B32 v5.4.0 .alls **글자 그대로** | Zoo 전처리 64장 | parse부터 |
| 3 | **S2** | G6 M (`g6_b32_m2rot_b1.onnx`, 회전 + sink) | `g6d_b32_S2.alls` = Zoo .alls, `{ew_add*}`만 잔차 26개 목록으로 | 64장 | parse부터 |
| 4 | **F** | (M_both HAR) | `g6d_b32_F.alls` = M_both + Zoo `clip_vit_b_16` .alls의 `allocator_param`·`context_switch_param` 두 줄 | — | M_both HAR에서 **compile만** |
| 5 | **Vrot** | Zoo 공개 B32 ONNX의 initializer 값만 회전 (`s2_g6d_vrot.py`) | `g6d_b32_Vrot.alls` = Zoo .alls 글자 그대로 | 64장 | parse부터 (Zoo yaml 노드 `onnx::Cast_0` → `1393`) |
| 6 | **G** | G6 M | `g6d_b32_G.alls` = M_both에서 `quantization_param([conv3, conv4, conv5, conv25], …)` 줄을 뺌 (16비트 conv는 패치 경로 `norm1, conv1, format_conversion1`만, 잔차 26개는 16비트) | 256장 (M_both와 같음) | parse부터 |
| 조건부 | **B16_rb_bk_D** | (B16_rb optimized HAR, G6c) | G6c `g6c_b16_rb_bk.alls` + `performance_param(compiler_optimization_level=max)` | — | compile만 (B16 상한 3시간) |

- **D 옵션의 출처:** 설치된 DFC 5.4.0 패키지 `hailo_sdk_client/sdk_backend/script_parser/commands.py`, `hailo_sdk_client/allocator/performance_params.py`의 `performance_param(compiler_optimization_level=max)`.
- **F 줄의 출처:** Zoo v5.4.0 `cfg/alls/generic/clip_vit_b_16_image_encoder.alls` (`alls/zoo_v540_clip_vit_b_16_image_encoder.alls`). 그 파일의 `resources_param`(0.9)은 넣지 않는다. M_both의 resources_param(1.0)이 그대로 남는다.
- **16비트 대응 확인 (G6 대응표):** V·R·M 그래프 모두 `conv3` = 블록 0 attention 출력 투영, `conv4·5` = 블록 0 fc1·fc2, `conv25` = 블록 5 fc2, `format_conversion1` = 패치 경로다. R 그래프는 ew_add 26개가 모두 잔차라 `{ew_add*}`를 그대로 쓴다. M 그래프는 마스크 덧셈이 섞여 있어 잔차 26개 목록(G6b와 같음)을 쓴다.
- **Vrot:**
  - 회전 방법: s2_rotate와 같은 회전 행렬(`rotq_rothad.npy`, `3a6391a2…`)로 LN 접기와 회전을 한다. ONNX 그래프 노드(종류·순서·입출력)와 initializer 이름은 그대로 두고 값만 바꾼다 (`s2_g6d_vrot.py` 머리말).
  - 만든 뒤 확인하는 것: 그래프가 원본과 같은지, ORT FP32 출력 `1393`의 cos (dev 64장). cos 최소 ≥ 0.9999가 아니면 빌드하지 않는다.
- **상한:** compile은 B32 2시간, B16 3시간. 넘으면 그 구성만 실패로 기록하고 다음으로 넘어간다.
- .alls와 Vrot ONNX(sha256)는 결과 전에 커밋하고, dev를 본 뒤 고치지 않는다. 컴파일 실패 때만 최소한으로 고치고 기록한다.

### 조건부: B16_rb_bk_D

- Dmax가 (a) M_both와 출력이 비트 단위로 같고 (b) B32 배치 1 지연을 M_both 대비 3% 이상 줄이면 B16_rb_bk_D를 만든다.
- (b)는 **결정용 측정**(Dmax와 M_both를 같은 세션에서 번갈아 배치 1 × 3세션, 같은 bench)으로 정한다. 그 값은 최종 속도 표에도 기록한다. 조건을 못 맞추면 건너뛰고 이유를 적는다.

### 1-1. 계획에서 달라진 점: Dmax compile 실패 → Dmax2 (결과 전 최소 수정)

- Dmax (`g6d_b32_Dmax.alls`)는 76초 만에 compile에 실패했다 (`logs/b32_Dmax_fail/compile.log`): `Failed to optimize bucket_0 with error: Performance Flow requires automatic resource utilization`.
  - `compiler_optimization_level=max`는 `resources_param`으로 자원 상한을 직접 정한 모델에는 쓸 수 없다.
- 최소 수정: **Dmax2** = M_both .alls에서 `resources_param` 줄만 빼고 `performance_param(compiler_optimization_level=max)`를 남긴 것 (`alls/g6d_b32_Dmax2.alls`).
  - 16비트 집합 등 양자화 줄은 M_both·M_z16(G6b, resources_param 없음)과 같다. 그래서 같은 M_both optimized HAR에서 compile만 한다.
  - 비교 기준은 **M_both**(18.38 ms)와 **M_z16**(19.21 ms, D 옵션 없는 같은 꼴) 둘 다다.
- 조건부 B16_rb_bk_D: B16_rb_bk는 resources_param + 수동 bucket이 있어야 compile되는데, D 옵션은 resources_param과 함께 쓸 수 없다. 그래서 1절의 "B16_rb_bk + D"는 그대로는 만들 수 없다. 다른 형태로 옮길지는 Dmax2 결과를 보고 따로 정한다 (용 확인).

## 2. 측정

### 2-1. 정확도 (AGX 실칩, G6 eval 코드·캐시·분류기)

- compile 설정만 바뀐 Dmax·F(·B16_rb_bk_D): dev 6×200으로 원본(M_both, B16은 M2rot2)과 비트 비교한다. 같으면 전체 eval을 생략한다. 다르면 dev 전체를 잰다.
- S1·S2·Vrot·G: dev 전체(고화질 4,923장 + CIFAR A 1,000장)
- 쌍 비교 95% 구간 (`s2_paired_ci.boot_hr`): 각 구성 − Z, 각 구성 − M_both, Vrot − V

### 2-2. 속도

- 같은 세션에서 번갈아 잰다: B32 Z, V, M_both, 새 구성 전부, B16_rb_bk (+ B16_rb_bk_D)
- 배치 1·8 × 3세션, `hailortcli parse-hef`(컨텍스트·크기), `run2 -t 10`

### 2-3. 안전 (용 규칙 + G6c 94.18 °C 반영)

- 속도는 세션·HEF마다, eval은 eval 사이마다 칩 < 64.6 °C에서 시작한다.
- **eval은 데이터셋 6개로 나눠, 사이마다 64.6 °C 미만으로 식힌다** (`run.sh EVAL_SPLIT=1`, `g6_hailo_chip.py --sets`, 기본 꺼짐). 나눠서 잰 임베딩은 합쳐 한 파일로 만든다. 출력은 결정적이라 정확도에는 영향이 없다.
- 단계(eval 데이터셋·bench HEF·run2)가 끝날 때마다 온도를 읽고 **95 °C를 넘으면 즉시 멈춘다** (`HOT_STOP=95`).
- 측정 도중에는 온도를 읽지 않는다 (추론 중 장치를 따로 열면 HailoRT 충돌 위험).

## 3. 판정 (결과 전 고정)

| 항목 | 규칙 |
|---|---|
| B32 속도 등급 | 배치 1 지연이 Z 이하 → **"Zoo보다 빠름"**, Z 대비 +5% 이내 → **"Zoo 수준"**, 그 밖 → **"느림"** |
| **B32 성공** | "Zoo 수준" 이상이면서 (구성 − Z)가 정확도 **"우위"**(고화질 Δtop-1 구간 하한 > 0)인 구성. 여러 개면 지연이 가장 짧은 것과 정확도가 가장 높은 것을 둘 다 적는다 |
| 원인 분해 (기록용) | S1 − V = 그래프 출처, S2 − S1 = sink 마스크, M_both − S2 = optimization_level·보정 장수 (속도·정확도 둘 다) |
| **B16 적용 성공** (B16_rb_bk_D를 만들었다면) | M2rot2와 출력 비트 동일 **그리고** 배치 1 지연이 B16_rb_bk 대비 3% 이상 감소 |
| 쌍 비교 | 하한 > 0 우위, 상한 < 0 열위, 그 밖 동률 |

- 이 기준은 결과를 본 뒤 바꾸지 않는다.

## 4. 출력

- 구성표(sha256·크기·컨텍스트·compile 시간), 정확도·속도 표, 판정, 원인 분해, 최고 온도와 그 시점, 계획에서 달라진 점, 확인 못 한 것
- **"B16으로 옮길 수 있는 것"** 표: 후보마다 다음을 적는다. S1·S2로 그래프 출처가 원인으로 나오면, B16 export를 어떻게 고칠지 가설을 적는다 (실행은 하지 않음).
  - (a) compile 설정만 바꾸는지
  - (b) 그래프·양자화를 바꾸는지
  - (c) B32 효과 크기
  - (d) B16에 옮길 때의 위험

## 5. 결과

### 5-0. 판정 (3절 규칙)

- **B32 성공 구성: 둘 (규칙대로 둘 다 적음)**
  - **지연이 가장 짧은 것: G** — 배치 1 15.66 ms (Z 대비 -5.0%, **"Zoo보다 빠름"**), G − Z = +2.67 [+1.92, +3.46] **우위**
  - **정확도가 가장 높은 것: F** (정확도는 dev 6×200 비트 일치로부터 M_both 값을 옮긴 **추정**, §2-1 규칙) — 배치 1 16.54 ms (Z 대비 +0.3%, **"Zoo 수준"**). 출력이 M_both와 비트 단위로 같아 F − Z = M_both − Z = +3.17 [+2.41, +3.95] **우위**
- 성공이 아닌 것
  - S1·Vrot: Zoo 수준(+1.5%·+1.6%)이지만 정확도가 동률이다.
  - S2·Dmax2·M_both: 느림 (+10.5~11.5%)
- **B16 적용 (B16_rb_bk_D):** **건너뜀.**
  - Dmax2는 (a) M_both와 비트 동일은 만족했다.
  - (b) 결정용 측정에서 M_both 대비 −0.87%로, 3% 이상 감소를 만족하지 못했다 (1-1, `chip/decision_dmax2/`).
  - 원래 Dmax(resources_param과 함께)는 compile할 수 없다는 점도 있다.
- 원인 분해 (기록용, 정확도는 쌍 비교, 속도는 배치 1 지연 차이)

| 단계 | 정확도 | 속도 (배치 1) |
|---|---|---|
| S1 − V: 그래프 출처 + 회전 (S1에는 회전이 있어 순수 그래프 출처가 아님) | +0.63 [-0.14, +1.47] 동률 | +0.21 ms |
| (기록용) S1 − Vrot: 회전을 뺀 그래프 출처 (둘 다 회전 + Zoo 레시피) | +0.13 [-0.51, +0.72] 동률 | -0.01 ms |
| (기록용) Vrot − V: Zoo 그래프에서 회전 | +0.51 [-0.37, +1.27] 동률 | +0.23 ms |
| S2 − S1: sink 마스크 | +1.48 [+0.77, +2.19] **우위** | **+1.63 ms** |
| M_both − S2: level 1·보정 256장 (16비트 집합은 같음) | +1.05 [+0.50, +1.63] **우위** | +0.01 ms |
| (기록용) F − M_both: compile 설정 (allocator·context_switch) | 비트 동일 | **-1.84 ms** |

- 해석
  - **그래프 출처는 B32·Zoo 레시피(level 0, 64장)에서 측정 가능한 차이를 만들지 않았다.** 우리 open_clip export와 Zoo ONNX는 같은 회전·같은 레시피에서 정확도(+0.13 동률)와 배치 1 속도(−0.01 ms)가 같다.
    - M_both 레시피(level 1, 256장)와 B16에서는 시험하지 않았다.
    - parse 뒤 그래프도 여전히 다르다: S1 175층 대 Vrot 223층 (`parsed_har_layer_counts.json`).
  - **배치 1에서 남은 +11%(M_both − Z = +1.89 ms)의 대부분은 sink 마스크 그래프와 함께 나타난다** (S2 − S1 = +1.63 ms). Zoo 레시피의 compile 설정(resources_param까지)으로는 이 비용이 줄지 않는다.
  - 그런데 **Zoo `clip_vit_b_16`의 `allocator_param`·`context_switch_param` 두 줄(F)**을 넣으면 출력은 그대로이고 배치 1 비용이 거의 사라진다 (F − M_both = −1.84 ms, Z 대비 +0.3%). 그래서 마스크의 고정 비용이라기보다 마스크 그래프와 compile 설정의 상호작용으로 본다.
  - 배치 8에서는 F도 Z 대비 +4.3%로, 차이가 다 없어지지는 않는다.

### 5-1. 구성표 (`build.jsonl`, `chip/parsehef_*.txt`)

| HEF | 만든 방법 | HEF 크기 B | 컨텍스트 | HEF sha256 (앞 16) | 보정 로그 | optimize / compile |
|---|---|---|---|---|---|---|
| Dmax (1차) | M_both HAR + D, compile만 | — | — | — | — | **compile 실패 76초** (`Performance Flow requires automatic resource utilization`) |
| **Dmax2** | M_both HAR + D (resources_param 뺌), compile만 | 108,232,704 | 13 | `6bc36872f79c74cc` | — | — / 76분 |
| **S1** | R ONNX `7d387f59…` + Zoo .alls | 109,973,504 | 12 | `5351ea6ff7973771` | 64 ×3 | 3분 22초 / 17분 32초 |
| **S2** | M ONNX `a0eb502b…` + Zoo .alls(잔차 목록) | 112,128,000 | 13 | `1193ca1b2a605d68` | 64 ×3 | 4분 22초 / 24분 47초 |
| **F** | M_both HAR + allocator·context_switch, compile만 | 111,583,232 | 13 | `dfd0d6baea71ee92` | — | — / 21분 45초 |
| **Vrot** | Vrot ONNX `fa0c7a18…` + Zoo .alls | 109,993,984 | 12 | `a4fc92497f3e8c7f` | 64 ×3 | 3분 25초 / 17분 27초 |
| **G** | M ONNX + M_both .alls − conv 16비트 줄 | 104,685,568 | **11** | `b6615bfc9c5f6f5c` | 256 ×4 | 33분 14초 / 20분 56초 |
| (비교) Z / V / M_both | G6·G6b | 102,645,760 / 102,649,856 / 111,955,968 | 12 / 12 / 13 | `c1612329…` / `96617700…` / `2a381857…` | | |

- M_both의 optimized HAR sha256은 `00614f55…`이고, Dmax2·F의 원본이다. 모든 HEF는 입력 UINT8 NHWC이고, 정규화가 HEF 안이다 (quant_info (1, 0)).
- **Vrot 그래프:** ONNX는 Zoo와 노드 434개의 종류·순서·입출력, initializer 이름이 같다 (`vrot.jsonl`). 그러나 **DFC parse 뒤 층은 248 → 223개**다 (`normalization` 층 86 → 61, `parsed_har_layer_counts.json`). LN의 γ=1·β=0이 되자 affine 부분이 생략된 것으로 보인다. 우리 export에서도 회전 전후가 200 → 175층(normalization 38 → 13)으로 같은 25개가 줄었다.
- 빌드 시작 기록에 `dirty=1`인 것(S2·F·Vrot·G)은 앞 커밋에 들어간 `build.jsonl`에 실행 기록이 덧붙었기 때문이다 (`git status`: `M results/siglip2/G6d/build.jsonl`만). 코드·.alls·스크립트는 커밋된 그대로다.

### 5-2. 정확도 (`results_chip.jsonl`, 실칩 dev 전체 5,923장, 커밋 `3b5f93d`, dirty false)

| HEF | 고화질 top-1 | FP32 대비 Δ [95% 구간] | cos 평균 / 최소 | CIFAR A (Δ) |
|---|---|---|---|---|
| Z | 85.74 | -4.22 [-4.95, -3.45] | 0.9366 / -0.205 | 70.8 (-4.4) |
| V | 85.75 | -4.21 [-5.00, -3.44] | 0.9373 / -0.134 | 70.3 (-4.9) |
| M_both | 88.91 | -1.05 [-1.57, -0.52] | 0.9835 / 0.242 | 74.4 (-0.8) |
| S1 | 86.39 | -3.57 [-4.31, -2.89] | 0.9528 / 0.138 | 70.8 (-4.4) |
| S2 | 87.87 | -2.09 [-2.72, -1.51] | 0.9785 / 0.279 | 73.1 (-2.1) |
| Vrot | 86.26 | -3.70 [-4.43, -2.96] | 0.9477 / 0.092 | 71.2 (-4.0) |
| G | 88.41 | -1.55 [-2.09, -1.03] | 0.9756 / 0.404 | 71.8 (-3.4) |
| Dmax2 · F | = M_both (dev 6×200, 1,200장 모두 비트 일치 → 전체 eval 생략) | | | |

| 쌍 | 차이 (고화질 %p) | 95% 구간 | 판정 | CIFAR A |
|---|---|---|---|---|
| S1 − Z | +0.65 | [-0.19, +1.46] | 동률 | +0.0 |
| S2 − Z | +2.12 | [+1.37, +2.89] | 우위 | +2.3 |
| Vrot − Z | +0.52 | [-0.27, +1.37] | 동률 | +0.4 |
| G − Z | +2.67 | [+1.92, +3.46] | 우위 | +1.0 |
| M_both − Z (G6b) | +3.17 | [+2.41, +3.95] | 우위 | +3.6 |
| S1 − M_both | -2.52 | [-3.29, -1.84] | 열위 | -3.6 |
| S2 − M_both | -1.05 | [-1.63, -0.50] | 열위 | -1.3 |
| Vrot − M_both | -2.65 | [-3.36, -1.97] | 열위 | -3.2 |
| G − M_both | -0.50 | [-1.03, +0.03] | 동률 | -2.6 |
| Vrot − V | +0.51 | [-0.37, +1.27] | 동률 | +0.9 |
| S1 − V (원인 분해: 그래프 출처 + 회전) | +0.63 | [-0.14, +1.47] | 동률 | +0.5 |
| S2 − S1 (원인 분해: sink 마스크) | +1.48 | [+0.77, +2.19] | 우위 | +2.3 |
| M_both − S2 (원인 분해: level 1·보정 256) | +1.05 | [+0.50, +1.63] | 우위 | +1.3 |
| S1 − Vrot (기록용: 회전 뺀 그래프 출처) | +0.13 | [-0.51, +0.72] | 동률 | -0.4 |

- G − M_both는 고화질 동률(하한 −1.03)이지만, CIFAR A는 −2.6%p로 G의 CIFAR 손실(−3.4)이 크다.
- S1·S2·Vrot·G의 eval은 데이터셋 6개로 나눠, 사이마다 64.6 °C 미만에서 시작했다 (`run_g6d.log`의 `cool eval_…`). `results_chip.jsonl`의 `start_temp_c`가 비어 있는 것은 라벨이 데이터셋별(`eval_<HEF>_<set>`)이기 때문이다.

### 5-3. 속도 (`results_speed.jsonl`, `chip/bench/`, HEF 10개를 같은 세션에서 번갈아, 배치 1·8 × 3세션)

- AGX Orin + Hailo-10H, HailoRT 5.3.0
- 스크립트: `g6_hailo_chip.py` `82ecd336…`, `run.sh` `2f9c9f2c…` (G6d 커밋 `b1b612a`, 옵션 `--sets`·`EVAL_SPLIT`·`HOT_STOP` 추가. bench 경로는 G6와 같음)
- 세션·HEF마다 칩 < 64.6 °C에서 시작했다 (86번, 최고 시작 64.33 °C). `HOT_STOP` 멈춤 0번
- 세 세션 사이 차이는 배치 1에서 0.43% 이내, 배치 8에서 1.6% 이내다.

| HEF | 컨텍스트 | 배치 1 ms | Z 대비 | 속도 등급 | 배치 8 ms/묶음 (Z 대비) | 배치 8 장/초 | 고화질 top-1 | 구성 − Z [95% 구간] |
|---|---|---|---|---|---|---|---|---|
| B32 Z (Zoo 공개) | 12 | 16.49 | +0.0% | — | 53.72 (+0.0%) | 148.9 | 85.74 | — |
| B32 V (G6) | 12 | 16.53 | +0.2% | Zoo 수준 | 54.00 (+0.5%) | 148.1 | 85.75 | — |
| B32 M_both (G6b) | 13 | 18.38 | +11.4% | 느림 | 61.30 (+14.1%) | 130.5 | 88.91 | +3.17 [+2.41, +3.95] 우위 |
| **B32 Dmax2** | 13 | 18.21 | +10.5% | 느림 | 67.99 (+26.6%) | 117.7 | = M_both (6×200 비트 일치) | +3.17 [+2.41, +3.95] 우위 |
| **B32 S1** | 12 | 16.74 | +1.5% | Zoo 수준 | 55.34 (+3.0%) | 144.6 | 86.39 | +0.65 [-0.19, +1.46] 동률 |
| **B32 S2** | 13 | 18.37 | +11.4% | 느림 | 61.35 (+14.2%) | 130.4 | 87.87 | +2.12 [+1.37, +2.89] 우위 |
| **B32 F** | 13 | 16.54 | +0.3% | Zoo 수준 | 56.04 (+4.3%) | 142.8 | = M_both (6×200 비트 일치) | +3.17 [+2.41, +3.95] 우위 |
| **B32 Vrot** | 12 | 16.76 | +1.6% | Zoo 수준 | 55.49 (+3.3%) | 144.2 | 86.26 | +0.52 [-0.27, +1.37] 동률 |
| **B32 G** | 11 | 15.66 | -5.0% | Zoo보다 빠름 | 48.84 (-9.1%) | 163.8 | 88.41 | +2.67 [+1.92, +3.46] 우위 |
| B16_rb_bk (G6c) | 16 | 40.48 | +145.5% | — | 177.75 (+230.9%) | 45.0 | 91.26 (= M2rot2, G6c) | — |

- 결정용 측정 (1-1, 같은 bench, M_both와 Dmax2만): 배치 1 M_both 18.376, Dmax2 18.216 ms (−0.87%). 최종 세션의 값(18.378·18.215)과 같다.

### 5-4. 온도

- **최고 87.04 °C**: 17:08:15 B16_rb_bk 배치 8 bench(세션 2) 직후. 오늘 G6d 칩 작업(14:13~17:31) 중 기록 465개 기준 (`chip/temps_top.json`)
- eval의 최고는 **74.24 °C**(S1·S2 caltech 직후)다. 데이터셋을 나눠 재서 G6c(94.18 °C)보다 크게 낮았다.
- 95 °C는 넘지 않았다.

## 6. B16으로 옮길 수 있는 것

| 후보 | (a) compile 설정만 바꾸나 (B16_rb_bk에 바로 적용) | (b) 그래프·양자화를 바꾸나 | (c) B32 효과 (배치 1, 출력) | (d) B16에 옮길 때의 위험 |
|---|---|---|---|---|
| **F: Zoo `allocator_param` + `context_switch_param`** | **예.** optimized HAR에서 compile만 | 아니오 (출력 비트 동일) | **M_both 대비 −10.0%** (18.38 → 16.54 ms), Z 대비 +0.3% | ① `context_switch_param`의 `auto_partition_to_buckets=True`·`share_buckets_partition=True`가 G6c 수동 bucket과 함께 쓰일 수 있는지 모른다. ② B16은 resources_param(1.0)과 bucket이 있어야 compile됐다 (G6c). 이 두 줄이 B16의 할당 시간 초과를 풀지 악화시킬지 모른다. **가장 유력한 후보다 — 다음 실험 제안** |
| Dmax2: `compiler_optimization_level=max` | 예 (compile만) | 아니오 (비트 동일) | 배치 1 M_both 대비 −0.9%. **배치 8은 67.99 ms로 Z 대비 +26.6%, M_both보다 약 11% 느림.** (resources_param 없는 M_z16의 G6b 세션 값 19.21 ms 대비 −5.2%, 같은 세션 비교 아님) | resources_param과 함께 쓸 수 없다. B16_rb_bk(resources_param + bucket)에는 그대로 못 붙이고, "B16 M2rot2 + D(resources_param 없음)"로만 가능하다. B32 효과가 작아 우선순위는 낮다 |
| G: 16비트 conv 최소 (패치 경로만) | 아니오 (재양자화 필요) | 예 (양자화) | Z 대비 −5.0%, M_both 대비 −14.8%. 정확도 M_both와 동률 (CIFAR −2.6) | **높음.** G6c에서 B16은 conv41 하나만 8비트로 바꿔도 실칩 HEF가 무너졌다 (B16_both_bk, 원인 미확인). B16은 블록 9 fc2가 민감한 층이었다 (G3) |
| S 계열: Zoo 레시피(level 0, 64장) | 아니오 | 예 (양자화) | M_both 대비 속도 이득은 거의 없다 (S2 −0.04%). 정확도는 −1.05 열위 | 이점 없음 |
| 그래프 출처 (export 방식) | — | 예 (그래프) | B32·Zoo 레시피에서 측정 가능한 차이 없음 (S1 − Vrot: 정확도 동률, 배치 1 −0.01 ms) | 이번 근거로는 B16 export를 고칠 이유가 보이지 않는다. 다만 M_both 레시피·B16에서는 미검증이고, parse 뒤 그래프는 다르다 (175 대 223층) |

- **S1·S2(와 Vrot) 결과로는 그래프 출처가 B32·Zoo 레시피에서 차이를 만들지 않았다.** 그래서 B16 export를 고치는 가설은 세우지 않는다. 배치 1 지연 차이는 sink 마스크 그래프와 함께 나타나고(S2 − S1 +1.63 ms), B32에서는 F의 compile 설정이 그 대부분을 가렸다.

## 7. 계획에서 달라진 점

1. **Dmax → Dmax2:** `compiler_optimization_level=max`는 resources_param과 함께 쓸 수 없어 compile이 실패했다. resources_param 줄만 뺀 Dmax2로 다시 했다 (1-1, 결과 전 커밋).
2. **B16_rb_bk_D 건너뜀:** (b) 조건 미달 (−0.87% < 3%)
3. **원인 분해 해석:** "S1 − V = 그래프 출처"에는 회전이 섞여 있다. 그래서 기록용 S1 − Vrot을 추가했다 (판정 규칙은 그대로).
4. **eval 분할:** 데이터셋 6개로 나눠 사이마다 냉각했다 (용 확인, 기본 꺼진 옵션)
5. 순서: Dmax2는 S1 optimize가 끝난 사이에 잠금을 잡아 S1 compile보다 먼저 compile됐다. 결과에는 영향이 없다.

## 8. 확인 못 한 것

- F의 두 줄을 B16_rb_bk(수동 bucket + resources_param)에 함께 쓸 수 있는지와 그 효과 (다음 실험 제안)
- F에서 두 줄 중 어느 것(allocator_param 또는 context_switch_param)이 효과를 내는지
- Vrot parse 뒤 `normalization` 층 25개가 생략된 것이 속도에 주는 영향 (Vrot과 S1의 지연은 같음)
- G가 B16에서 무너지지 않을지
- 빌드 사이 변동, Windows 쪽 부하 (해당 없음, AGX만 측정)

## 9. 3중 확인 기록

1. 자기 검증
   - Vrot: 그래프 동일성, 바꾼 initializer 175개, FP32 cos 최소 0.9999999
   - Dmax2·F: M_both와 6×200 비트 일치
   - eval 분할 결과를 합친 뒤의 모양 확인, 보정 장수 assert
2. 독립 재계산: 검토자가 `g6d_summarize.py`를 raw에서 다시 돌려 21줄 모두 일치했다. 분할 임베딩 = 데이터셋별 조각, Vrot을 무작위 입력으로 다시 확인 (cos 0.99999994).
3. 독립 검토 (맥락 없는 서브에이전트) — 반영한 지적
   - (a) "그래프 출처는 원인이 아니다" → B32·Zoo 레시피 한정으로 고치고, parse 뒤 층 수 차이 명시
   - (b) sink 마스크 비용은 compile 설정과의 상호작용, 배치 8 F +4.3%
   - (c) Dmax2 배치 8(+26.6%)과 M_z16 비교가 다른 세션 값이라는 점
   - (d) F 정확도는 추정
   - (e) parse 층 수 근거 파일 추가
