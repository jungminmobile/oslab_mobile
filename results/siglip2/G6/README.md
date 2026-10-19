# G6: Hailo Zoo 모델에 M2rot을 적용해 Zoo 공개 HEF와 Hailo-10H 실칩에서 비교 (설계 2026-10-20)

> 상태: **설계·판정 고정 (결과 전).** 1~6절은 결과 전에 커밋한다. 결과는 7절부터 붙인다.
> 브랜치 `siglip2/g6-zoo-transfer` (`siglip2/g6-zoo-transfer-pre` 3211a37에서 만듦), worktree `~/oslab_g6`. 담당: 용 (용 요청). **test는 쓰지 않는다** (dev 전체만).
> 사전 확인: `results/siglip2/G6_pre/README.md` (B32는 통일 규칙 sink 가능, S1B16은 회전만).

## 1. 질문과 판정 (결과 전 고정)

- **주 주장: B32에서 M − Z가 "우위"인가.** 우리 방식(M2rot, 통일 규칙)을 적용한 HEF가 Zoo 공개 HEF보다 실칩 정확도가 높은가.
- 정확도 비교: 실칩 dev 전체, 고화질 5종 top-1의 짝 차이. 95% 구간은 `s2_paired_ci.boot_hr`(층화 짝 부트스트랩, 시드 0, 1000회)로 낸다.
  - 하한 > 0 → **우위**, 상한 < 0 → **열위**, 그 밖 → **동률**
- 쌍 비교 목록
  - B32: **M−Z**(주), M−V, M−R, R−D, D−V, V−Z, R−V
  - S1B16: R−Z, R−V, V−Z (+ S1B16 D가 있으면 R−D, D−V)
- **기여 분해 (B32):**
  - V−Z = 보정 데이터 효과 (Zoo COCO 64장 → 우리 보정 풀 64장. 같은 .alls)
  - **D−V = 레시피 효과 (묶음).** 아래 차이를 모두 묶은 값이다: 16비트 층 집합(3-2), 블록 9 fc2 16비트, 보정 64 → 256장, optimization_level 0 → 1, resources_param·bucket 없음, ONNX 출처(Zoo ONNX → 우리 timm export)
  - R−D = 회전 효과
  - M−R = sink 효과 (통일 규칙 15@6:5, 7@10:9)
- S1B16 D를 만들지 못하면 S1B16 R−V는 "회전 + 레시피 효과"로 적는다.
- **속도·크기는 판정하지 않는다.** 비율로만 보고한다. 속도 차이에는 컴파일 설정(resources_param·bucket 유무) 차이가 포함된다.
- 이 기준은 결과를 본 뒤 바꾸지 않는다. 바꿔야 하면 이유를 적고 용의 승인을 받는다.

## 2. 구성

| 모델 | 이름 | ONNX | .alls | 보정 (모두 Zoo 전처리) | 컴파일 |
|---|---|---|---|---|---|
| B32 | **Z** | — | — | — | Zoo 공개 HEF (v5.4.0 hailo10h `siglip2_b_32_256_image_encoder.hef`) |
| B32 | **V** | Zoo ONNX (`siglip2-base-patch32-256_vision_encoder.sim.onnx`, parser 노드 yaml 그대로) | Zoo v5.4.0 그대로 | 64장 | 1 |
| B32 | **D** | 우리 timm export, 회전 없음 | D256L1 이식 (3-2) | 256장 | 2 |
| B32 | **R** | 우리 timm export, rothad 회전 | D256L1 이식 | 256장 | 3 |
| B32 | **M** | 우리 timm export, rothad 회전 + M2 통일 규칙 `15@6:5, 7@10:9` | D256L1 이식 (M 그래프 층 이름) | 256장 | 4 |
| S1B16 | **Z** | — | — | — | Zoo 공개 HEF (G5에서 받은 것, `e2198ed5…`) |
| S1B16 | **V** | Zoo ONNX (`siglip_base_patch16_224_vision_encoder.onnx`) | Zoo v5.4.0 그대로 | 64장 | 5 |
| S1B16 | **R** | 우리 timm export, rothad 회전 | D256L1 이식 | 256장 | 6 |
| S1B16 | D (선택) | 우리 timm export, 회전 없음 | D256L1 이식 | 256장 | 7 (시간이 남으면) |
| S1B16 | (참고) G5c R-our | G5c HEF `e21fbb12…` (timm 전처리 보정) | | | 다시 컴파일 안 함, 참고 측정만 |

- **보정 사진:** G6_pre의 보정 256장(고화질 5종 train 보정 후보에서 시드 0, 기존 `calib_images("highres",256,0)`과 같은 사진·순서)을 Zoo 전처리(PIL BICUBIC 바로 맞춤, crop 없음)한 uint8 픽셀이다.
  - 64장 구성(V)은 그 **앞 64장**을 쓴다 (G5b 규칙).
  - 크기는 B32 256, S1B16 224. DFC에는 NHWC float32 0~255로 넣는다. 로그의 "Using dataset with N entries"가 64 또는 256인지 assert한다.
- **우리 export 모델:** open_clip `hf-hub:timm/ViT-B-32-SigLIP2-256`, `hf-hub:timm/ViT-B-16-SigLIP`의 timm trunk를 쓴다. 기존 `s2_rotate`·`s2_regcache` 코드를 그대로 쓰기 위해서다.
  - 4-1에서 Zoo ONNX와 cos 평균 ≥ 0.9999가 아니면 **멈추고 보고**한다.
  - 회전 Q는 기존 `rotq_rothad.npy`(768차원, M2rot과 같은 파일)를 쓴다.
- **M의 상수:** 회전 모델에 위 보정 256장(Zoo 전처리)을 넣어 만든다. K/V 평균은 `build_mean_cache`, u_c는 `build_uc`의 정의 그대로. 그래프는 `MaskRegCacheTrunk(form=bcast, mask_site=mlp)`에 새 옵션(4-2)을 켠 것이다.

## 3. .alls

### 3-1. V: Zoo v5.4.0 그대로 (`hailo_model_zoo/cfg/alls/generic/*.alls`, 태그 v5.4.0)

- B32: `normalization([127.5]*3,[127.5]*3)`, `optimization_level=0`, LN 분해, 보정 64장, 16비트 `[norm1, conv1, format_conversion1, conv3, conv4, conv5, conv25]`와 `{ew_add*}`, resources_param
- S1B16: 같은 꼴, 16비트 `[norm1, conv1, conv1_s2d]`, `[conv4, conv5, conv24, conv25]`, `{ew_add*}`, resources_param, bucket 3개

### 3-2. D·R·M: D256L1 원칙 이식 (결과 전 고정, 한 번 옮긴 뒤 dev 결과를 보고 고치지 않음)

D256L1(`siglip2/hailo/g4_D256L1_official_b9fc2.alls`, M2rot은 `g3r_M2rot_D256L1equiv.alls`)의 원칙:

| 항목 | D256L1 (우리 SigLIP2-B/16) | 이식 규칙 |
|---|---|---|
| optimization_level / compression_level | 1 / 0 | 같음 |
| pre_quantization | `layer_norm_decomposition, equalization=disabled, bit_decomposition_mode=uniform_precision` | 같음 |
| 보정 장수 | 256 | 256 |
| 16비트: 패치 임베딩 | `conv1` | 그 모델의 패치 임베딩 conv와, Zoo .alls가 그 경로로 함께 16비트로 둔 입력 층(`format_conversion1` 또는 `conv1_s2d`) |
| 16비트: 블록 0 fc1·fc2 | `conv4, conv5` | 그 모델에서 블록 0 MLP fc1·fc2에 해당하는 층 |
| 16비트: 블록 5 fc1·fc2 | `conv24, conv25` | 블록 5 MLP fc1·fc2 |
| 16비트: 블록 9 fc2 | `conv41` | 블록 9 MLP fc2 |
| 16비트: 잔차 흐름 덧셈 | `ew_add*` 중 잔차 덧셈 26개 (위치 임베딩 1 + 블록당 2 × 12 + 풀링 헤드 잔차 1. M 그래프의 마스크 연산 덧셈은 제외 — M2rot과 같음) | 그 그래프의 같은 26개 |
| 입력 정규화 | 없음 (호스트 정규화) | **`norm1 = normalization([127.5]*3, [127.5]*3)` 추가**, 16비트 (요청 3: Zoo와 같은 uint8 입력. Zoo .alls와 같은 줄) |
| resources_param·bucket | 없음 | 없음 |

- 층 이름의 대응은 DFC parse 결과(`s2_har_layers.py`, HAR 층 ↔ ONNX 노드)로 정한다. 대응표(원래 D256L1 층 → 새 모델 층, 근거 ONNX 노드 이름)를 6-2에 남긴다.
- 컴파일이 실패할 때만 최소한으로 고치고, 고친 내용을 8절에 적는다.

## 4. 구현 (결과 전 커밋)

1. **동일성:** open_clip timm 모델(B32·S1B16)과 Zoo ONNX(ORT)의 출력 cos. Zoo 전처리 32장(G6_pre `ident32`). 평균 ≥ 0.9999가 아니면 멈춘다.
2. **`MaskRegCacheTrunk` 옵션:** `u_starts`(sink마다 u_c 시작 블록)를 받는다. 기본값 None이면 u = K/V 시작 블록 b로, 기존 동작과 같다. 통일 규칙은 sinks `[(15,6),(7,10)]` + `u_starts [5,9]`다.
   - 회귀 확인: 기존 M2rot export(`--graph mask --mask-form bcast --mask-site mlp --rot rothad`, sinks 55@5,13@9)를 **바꾸기 전 코드와 바꾼 뒤 코드로 각각 export**한다. b1 ONNX와 dynamic ONNX 파일이 바이트 단위로 같은지 본다 (sha256).
   - 기존 배포 파일(`siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad_b1.onnx`)과의 일치도 기록한다.
3. **모델 선택:** `s2common.MODEL_ID`는 환경변수 `S2_MODEL_ID`가 있으면 그 값을 쓴다 (없으면 기존 `hf-hub:timm/ViT-B-16-SigLIP2`). export의 더미 입력 크기는 trunk의 `patch_embed.img_size`를 쓴다 (기존 모델은 224라 같다). 위 2의 회귀 확인이 이 변경도 함께 확인한다.
4. **FP32 동치 (export마다):**
   - R·D: PyTorch 회전/원래 vs Zoo ONNX(ORT), Zoo 전처리 dev 64장, cos 최소 ≥ 0.9999
   - M: PyTorch M vs HF G6_pre `hf_forward`(같은 sinks·같은 상수 정의), cos 최소 ≥ 0.9999
   - 모든 export: ORT(b1 ONNX) vs PyTorch, cos 최소 ≥ 0.9999
5. **입력 형식:** 모든 우리 HEF는 정규화를 HEF 안에 넣는다 (3-2). `hailortcli parse-hef`로 모든 HEF의 입력·출력 형식·quant_info를 6-1에 적는다. 만약 넣지 못하면 호스트 정규화 시간을 따로 재서 지연에 더한다.

## 5. 측정

### 5-1. FP32 기준 (데스크탑)

- Zoo 전처리 dev 전체 캐시: 고화질 5종 dev 전부(4,923장)와 CIFAR A 1,000장, 크기 224·256. G6_pre `cache`와 같은 사진 선택(SHA-1 대조)이다.
- 기준: HF FP32 (`google/siglip2-base-patch32-256`, `google/siglip-base-patch16-224`) `pooler_output`. 분류기는 G6_pre와 같다 (HF 텍스트 인코더, `_clean_canonicalize`, max_length 64).
  - Δtop-1과 임베딩 cos의 기준은 이 FP32다.
- 통일 규칙 M2의 FP32 top-1 (B32 15@6:5,7@10:9, G6_pre `hf_forward`)도 낸다.

### 5-2. 실칩 정확도 (AGX Orin + Hailo-10H)

- 모든 HEF를 dev 전체 5,923장으로 돌린다. 입력은 Zoo 전처리 uint8 픽셀을 float32로 바꿔 넣고, HailoRT가 quant_info로 양자화한다. 모든 HEF가 정규화를 안에 넣었으므로 입력이 같다.
- 지표: FP32 대비 Δtop-1(고화질 평균과 95% 구간, CIFAR A는 따로), 임베딩 cos(평균·최소), 데이터셋별 top-1. 계산은 G5b `g5b_devfull_summarize.py`와 같다.
- **CIFAR도 실칩 값만 쓴다** (에뮬레이터 불일치, G5b).

### 5-3. 구조·속도

- `hailortcli parse-hef`: 컨텍스트 수, 입출력 형식. HEF 크기(바이트)와 sha256.
- 지연: G5 `g5_hailo_bench.py`. 배치 1·8, 각 3세션. 세션마다 모든 HEF를 같은 순서로 번갈아 잰다 (워밍업 20, 200장). 보조로 `hailortcli run2 -t 10` FPS.
- 규칙 (G5와 같음)
  - `run.sh`를 nohup으로 돌리고 로그를 남긴다. tmux는 쓰지 않는다. 대기 명령에는 timeout을 둔다.
  - 다른 Hailo 프로세스가 0개일 때만 시작한다.
  - 세션마다 칩 온도 max(ts0, ts1)가 **쉬는 온도 + 2 °C 미만**일 때 시작한다. 쉬는 온도는 G3RQ에서 잰 62.6 °C → 문턱 64.6 °C. 시작 때 다시 잰 쉬는 온도가 62.6 °C보다 높으면 그 값 + 2 °C를 쓰고 기록한다.

### 5-4. (선택) ORT CPU 부록

시간이 남으면 데스크탑에서 B32 FP32 ONNX와 우리 ORT 경로(M2rot_ort 방식)의 INT8을 비교한다. dev 6×200 정확도와 기존 측정 규칙의 지연을 재고, **비공식**으로 표기한다.

## 6. 구성 기록 (측정 후 채움)

### 6-1. HEF 표 (sha256·크기·컨텍스트·입출력 형식)
(측정 후)

### 6-2. .alls 층 대응표 (DFC parse 뒤, optimize 전에 커밋)

도구: `g6_alls.py`. 층 이름이 아니라 그래프 구조와 원래 ONNX 노드 이름으로 정한다.
- fc1·fc2: `mlp/fc1|fc2(_n)/MatMul` conv를 위상 순서로 센 블록 번호
- 잔차 덧셈: 입력이 attention 출력 투영이나 fc2인 ew_add, 그리고 위치 임베딩 덧셈
- 근거 표: `alls/map_<구성>.csv` (역할, DFC 층, ONNX 노드)

**도구 검증 (결과 전):**
- 같은 규칙을 기존 M2rot 그래프(`siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad_b1.onnx`)에 적용하면 16비트 목록이 `siglip2/hailo/g3r_M2rot_D256L1equiv.alls`와 **정확히 같다** (conv 6개, 잔차 ew_add 26개 — `--check`).
- rothad 그래프에서는 ew_add 전체가 26개로 잔차 덧셈과 같다. 그래서 D256L1의 `{ew_add*}`와 같은 뜻이다.

| 역할 (D256L1 원래 층) | B32 D·R | B32 M | S1B16 R·D | 근거 ONNX 노드 (B32 M 예) |
|---|---|---|---|---|
| 입력 정규화 (없음 → 추가) | norm1 | norm1 | norm1 | `.alls` normalization |
| 패치 임베딩 (`conv1`) | conv1 | conv1 | conv1 | `/trunk/patch_embed/proj/Conv` |
| 입력 경로 층 (Zoo .alls 기준) | format_conversion1 | format_conversion1 | conv1_s2d (optimize 때 생김, Zoo S1B16 .alls와 같음) | `/trunk/patch_embed/Reshape;Transpose` |
| 블록 0 fc1·fc2 (`conv4, conv5`) | conv4, conv5 | conv4, conv5 | conv4, conv5 | `/trunk/blocks.0/mlp/fc1·fc2/MatMul` |
| 블록 5 fc1·fc2 (`conv24, conv25`) | conv24, conv25 | conv24, conv25 | conv24, conv25 | `/trunk/mlp/fc1·fc2/MatMul` (M 그래프의 블록 5) |
| 블록 9 fc2 (`conv41`) | conv41 | conv41 | conv41 | `/trunk/mlp/fc2_4/MatMul` (M 그래프의 블록 9) |
| 잔차 ew_add 26개 | ew_add1~26 (전체) | ew_add1~12, 14, 17, 19, 22, 24, 27, 29, 32, 34, 39, 41, 46, 48, 53 | ew_add1~26 (전체) | `/trunk/Add`, `blocks.N/Add(_1)`, M 블록의 잔차 Add |

- 만들어진 `.alls`: `alls/g6_b32_D.alls`, `g6_b32_R.alls`, `g6_b32_M.alls`, `g6_s1b16_R.alls`, `g6_s1b16_D.alls`
- Zoo 그대로: `alls/zoo_v540_b32.alls`, `zoo_v540_s1b16.alls` (S1B16은 G5c의 파일과 같다)
- B32 M에서 ew_add12는 블록 5 attention 잔차다. 통일 규칙에서는 블록 5의 K/V를 바꾸지 않으므로 보통 잔차 덧셈과 같다.

## 7. 결과
(측정 후)

## 8. 계획과 달라진 점
(측정 후)
