# G6d: B32 Hailo 속도 후보 한 번에 빌드·측정 + B16으로 옮길 수 있는 것 가리기 (설계 2026-10-21)

> 상태: **설계·판정 고정 (결과 전).** 1~4절은 결과 전에 커밋한다. 결과는 5절부터 붙인다.
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
(측정 후)
