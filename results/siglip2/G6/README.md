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

전체 sha256과 단계별 시간·RSS는 `build.jsonl`, 입출력 형식과 컨텍스트 수는 AGX `hailortcli parse-hef`(`chip/out/parsehef_*.txt`)에서 가져왔다.
- 모든 HEF는 입력이 `UINT8 NHWC`이고 quant_info가 (scale 1, zp 0)이다. 즉 정규화가 HEF 안에 있고(픽셀을 그대로 넣음), 출력은 `UINT16 NHWC(1×1×768)`이다 (`eval_info.json`의 `quant_in`·`input_domain`).
- 그래서 호스트 정규화 시간은 따로 더하지 않는다 (README 4-5).

| HEF | 출처·빌드 | ONNX sha256 (앞 16) | HEF 크기 B (Z 대비) | 컨텍스트 | HEF sha256 (앞 16) | 보정 로그 | optimize / compile |
|---|---|---|---|---|---|---|---|
| **B32 Z** | Zoo v5.4.0 공개 | — | 102,645,760 | 12 | `c16123297d83a36c` | — | — |
| **B32 V** | Zoo ONNX + Zoo .alls | (Zoo zip `6f6f5731…`) | 102,649,856 (+0.00%) | 12 | `96617700b59e8dda` | 64 ×3 | 4분 14초 / 20분 33초 |
| **B32 D** | timm export + D256L1 이식 | `daf75de07fd529e9` | 103,997,440 (+1.3%) | 15 | `d2f5cd6c4f45159d` | 256 ×4 | 16분 23초 / 18분 42초 |
| **B32 R** | + rothad 회전 | `7d387f5941b43ece` | 111,108,096 (+8.2%) | 15 | `c2105fed76c541f9` | 256 ×4 | 16분 37초 / 19분 04초 |
| **B32 M** | + M2 통일 규칙 | `a0eb502bc6ca0f8d` | 112,975,872 (+10.1%) | 17 | `311655b72aad4dac` | 256 ×4 | 21분 22초 / 21분 42초 |
| **S1B16 Z** | Zoo v5.4.0 공개 (G5와 같은 파일) | — | 99,061,760 | 17 | `e2198ed5975da551` | — | — |
| **S1B16 V** | Zoo ONNX + Zoo .alls | (Zoo zip `5342aaa6…`) | 99,168,256 (+0.11%) | 17 | `395044e3a38190b1` | 64 ×3 | 3분 39초 / 50분 40초 |
| **S1B16 R** | timm export + rothad + D256L1 이식 | `d2dda01042883d38` | 111,230,976 (+12.3%) | 17 | `6226cb511bf41304` | 256 ×4 | 20분 55초 / 25분 30초 |
| S1B16 D (선택) | timm export + D256L1 이식 | `92b4d9f263db8547` | 99,598,336 (+0.54%) | 17 | `cb7f618106736c8e` | 256 ×4 | 20분 32초 / 23분 59초 |
| (참고) S1B16 G5c R-our | G5c 빌드 | — | 99,168,256 | 17 | `e21fbb12fc8bdb20` | 64 | — |

- 모든 빌드는 커밋 `a565049`(S1B16 D만 `aed45fd`)에서 dirty 0으로 했다. DFC 5.4.0, `--hw-arch hailo10h`이고, optimize 최대 RSS는 18.7 GB다.
- AGX에 옮긴 HEF의 sha256은 데스크탑 값과 같다 (`chip/out/run_*.log`의 `hef … sha256`).

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

### 7-0. 판정 (1절 규칙 그대로)

- **주 주장 (B32 M − Z): +3.38%p [+2.62, +4.17] → "우위".** 우리 방식(M2rot 통일 규칙 + D256L1 이식)을 적용한 B32 HEF가 Zoo 공개 HEF보다 실칩 고화질 top-1이 유의하게 높다. CIFAR A도 +3.9%p다.

| B32 기여 분해 | 차이 (고화질, %p) | 95% 구간 | 판정 | CIFAR A 차이 |
|---|---|---|---|---|
| V − Z (보정 데이터: COCO 64 → 우리 보정 풀 64) | +0.01 | [−0.72, +0.78] | 동률 | −0.5 |
| D − V (레시피 묶음: 16비트 층·블록 9 fc2·보정 256·level 1·resources/bucket 없음·ONNX 출처) | +1.74 | [+0.98, +2.55] | 우위 | +0.4 |
| R − D (회전) | +0.70 | [+0.05, +1.38] | 우위 (하한이 0에 가까움) | +1.7 |
| M − R (sink, 통일 규칙) | +0.92 | [+0.43, +1.45] | 우위 | +2.3 |
| 합계 M − Z | +3.38 | [+2.62, +4.17] | **우위** | +3.9 |
| (참고) M − V | +3.36 | [+2.61, +4.17] | 우위 | +4.4 |
| (참고) R − V | +2.44 | [+1.71, +3.16] | 우위 | +2.1 |

| S1B16 (대조군) | 차이 | 95% 구간 | 판정 | CIFAR A 차이 |
|---|---|---|---|---|
| R − Z | +0.57 | [+0.04, +1.10] | 우위 (하한이 0에 가까움) | +0.2 |
| R − V | +0.48 | [−0.03, +0.97] | 동률 | −0.7 |
| V − Z | +0.09 | [−0.42, +0.63] | 동률 | +0.9 |
| R − D (회전, 선택 D) | +0.41 | [−0.12, +0.88] | 동률 | +3.2 |
| D − V (레시피 묶음, 선택 D) | +0.07 | [−0.50, +0.65] | 동률 | −3.9 |

- S1B16 D를 넣었으므로 R − V는 "회전 + 레시피"로 묶지 않고, R − D(회전)와 D − V(레시피)로 나눠 적었다.
- 해석 (가설): B32는 우리 SigLIP2-B/16과 같은 기전(G6_pre)이다. 그래서 레시피(블록 5·9 보호)·회전·sink가 모두 쌓였다. S1B16은 Zoo HEF의 손실이 원래 작고(−1.27), sink를 쓸 수 없는 모델이라 이득이 작다.
- 구간은 사진 단위 짝 부트스트랩이다. HEF 빌드마다 생기는 양자화 무작위성(같은 레시피를 다시 빌드했을 때의 차이)은 포함하지 않는다.

### 7-1. FP32 기준 (`fp32.jsonl`, HF FP32, Zoo 전처리 dev 전체)

| 모델 | CIFAR A | Pets | Flowers | Cars | Food | Caltech | **고화질 평균** |
|---|---|---|---|---|---|---|---|
| B32 FP32 | 75.20 | 92.02 | 86.42 | 91.62 | 85.99 | 93.75 | **89.96** |
| B32 M2 FP32 (통일 규칙, 양자화 없음) | 75.30 | 91.92 | 86.42 | 91.93 | 86.09 | 93.75 | 90.02 (cos 평균 ≥ 0.99985) |
| S1B16 FP32 | 73.80 | 93.13 | 86.83 | 92.04 | 87.61 | 94.82 | **90.89** |

- 사진 수: 고화질 4,923장(Pets 990, Flowers 957, Cars 967, Food 985, Caltech 1,024)과 CIFAR A 1,000장.

### 7-2. 실칩 정확도 (`results_chip.jsonl`, AGX Hailo-10H, HailoRT 5.3.0, dev 전체 5,923장)

| HEF | 고화질 top-1 | **FP32 대비 Δ [95% 구간]** | cos 평균 / 최소 (고화질) | CIFAR A top-1 (Δ) | CIFAR cos | 시작 온도 °C |
|---|---|---|---|---|---|---|
| B32 Z | 85.74 | −4.22 [−4.95, −3.45] | 0.9366 / −0.205 | 70.8 (−4.4) | 0.881 | 62.3 |
| B32 V | 85.75 | −4.21 [−5.00, −3.44] | 0.9373 / −0.134 | 70.3 (−4.9) | 0.876 | 62.6 |
| B32 D | 87.50 | −2.46 [−3.06, −1.84] | 0.9633 / −0.054 | 70.7 (−4.5) | 0.936 | 62.2 |
| B32 R | 88.20 | −1.76 [−2.35, −1.21] | 0.9744 / 0.322 | 72.4 (−2.8) | 0.959 | 62.3 |
| **B32 M** | **89.12** | **−0.84 [−1.31, −0.37]** | **0.9840 / 0.381** | **74.7 (−0.5)** | 0.969 | 62.0 |
| S1B16 Z | 89.61 | −1.27 [−1.81, −0.72] | 0.9742 / 0.387 | 71.4 (−2.4) | 0.929 | 64.4 |
| S1B16 V | 89.70 | −1.18 [−1.67, −0.64] | 0.9765 / 0.744 | 72.3 (−1.5) | 0.922 | 62.0 |
| **S1B16 R** | **90.19** | **−0.70 [−1.09, −0.26]** | **0.9875 / 0.549** | 71.6 (−2.2) | 0.970 | 62.0 |
| S1B16 D | 89.78 | −1.11 [−1.58, −0.62] | 0.9804 / 0.704 | 68.4 (−5.4) | 0.958 | 61.9 |
| (참고) S1B16 G5c R-our | 89.70 | −1.18 [−1.67, −0.64] | 0.9765 / 0.744 | 72.3 (−1.5) | 0.922 | 64.0 |

- 모든 eval은 칩 온도 64.6 °C 미만에서 시작했다 (`chip/out/temps_eval.jsonl`). 과열 사고(8절 2)보다 먼저 모두 끝났다.
- B32 Z·V·D의 cos 최소가 음수다. 출력이 완전히 망가진 사진이 몇 장 있다는 뜻이다. R·M에서는 0.32·0.38로 줄었다.
- 데이터셋별 top-1은 `results_chip.jsonl`의 `per_set`에 있다.

### 7-3. 속도
(측정 중 — 8절 2의 사고 뒤 다시 잼)

## 8. 계획과 달라진 점

1. **S1B16 V는 G5c R-our와 같다.**
   - 요청(용 지시 1)대로 Zoo 전처리 보정 64장으로 다시 컴파일했다.
   - 그런데 그 보정 파일이 G5c의 `calib.npy`와 **바이트 단위로 같았다** (sha256 `cf73051f…`). 이 사진들에서는 timm SigLIP2 전처리(bicubic, 바로 맞춤)가 Zoo 전처리와 같은 픽셀을 낸다.
   - 그래서 HEF sha256은 다르지만(`395044e3…` vs `e21fbb12…`, 크기는 같음) 실칩 출력은 6종 모두 **비트 단위로 같다**.
   - 설계 때 "G5c는 timm 전처리라 다르다"고 적은 전제가 틀렸다. 결과에는 영향이 없다.
2. **속도 측정 중 과열 사고 (10/20 08:01~08:14)와 규칙 변경.**
   - 5-3의 "시작 때 잰 쉬는 온도가 62.6 °C보다 높으면 그 값 + 2"는 제(Claude) 설계 실수였다. 속도 측정을 eval 바로 뒤에 시작해서 약 83 °C를 쉬는 온도로 잡았고, 문턱이 약 85 °C가 되어 냉각이 사실상 꺼졌다.
   - 배치 8 세션 1에서 칩이 **103 °C**에 이르렀고, 세션 2에서 HailoRT 드라이버 오류(`HAILO_DRIVER_OPERATION_FAILED`, `/dev/h1x-0` 열기 실패)가 났다. 용이 AGX를 `sudo reboot`했다 (재부팅 뒤 48.6 °C, identify 정상).
   - 그 실행의 속도 결과(배치 1 세션 1~3, 배치 8 세션 1)는 **버린다**. AGX `out/aborted_bench_0801/`에 두었고 결과에 쓰지 않는다.
   - 바꾼 규칙 (용 보고 뒤, 속도를 다시 재기 전에 커밋 `4620c66`·`1e1c98b`·`43a3728`)
     - 문턱은 **64.6 °C 고정** (G5와 같음, 시작 온도는 기록만)
     - 90분 안에 식지 않으면 진행하지 않고 **멈춘다**
     - 세션 안에서도 **HEF마다** 64.6 °C 미만으로 식힌 뒤 잰다 (같은 순서로 번갈아 재는 것은 그대로). 첫 실행에서 배치 8 세션 하나로 88 → 103 °C가 됐기 때문이다. 이 때문에 G5(세션 안에서 냉각 없음)와 측정 조건이 다르다.
   - 정확도 eval은 사고 전에 모두 64.6 °C 미만에서 시작했고(7-2), 출력은 온도와 상관없이 결정적이다 (S1B16 V와 G5c R-our의 비트 일치로도 확인).
3. **회귀 확인에서 찾은 버그:** 변환 trunk(`MaskRegCacheTrunk`)를 export할 때 입력 크기를 읽는 곳이 틀렸다. 결과 전에 고쳤고(`eb78616`), 고친 뒤 옛 코드·새 코드·배포 파일의 sha256이 같음을 확인했다 (`regress/`).
4. **입력 경로 층:** 3-2 규칙대로 S1B16은 Zoo .alls와 같이 `conv1_s2d`(optimize 때 생김)를 16비트로 두었다. B32는 `format_conversion1`이다. 컴파일 실패는 없었고, .alls는 처음 만든 그대로 썼다.
5. **eval 실행 방식:** HEF가 빌드되는 대로 HEF마다 `run.sh`를 따로 돌렸다 (한 실행에 모두 넣지 않음). 정확도는 HEF마다 결정적이라 순서 영향이 없다. 속도는 모든 HEF를 한 실행에서 번갈아 쟀다.
6. **S1B16 D (선택):** B32 전체와 S1B16 Z·V·R의 빌드·eval이 끝난 뒤 빌드했다 (`ONLY=s1b16_D`, 커밋 `aed45fd`). 속도 측정에도 넣었다.
7. **G5c R-our:** 참고로 eval만 했다. 속도 측정에서는 뺐다 (V와 출력이 같음).
8. 모델 출처: 우리 export는 open_clip timm 모델이다. Zoo ONNX와의 cos 평균은 B32 0.999993, S1B16 1.000000이다 (`ident.jsonl`). FP32 기준과 분류기는 HF 모델로 만들었다 (G6_pre와 같음).
