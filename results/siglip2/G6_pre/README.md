# G6-pre: Hailo Zoo 두 모델에 M2rot(회전 + sink K/V 상수 캐시)이 들어맞는지 FP32 사전 확인 (2026-10-19)

> 상태: **설계·판정 고정 (결과 전).** 이 절 아래의 규칙은 측정 전에 커밋했다. 결과는 뒤 절에 붙인다.
> 범위: FP32만. 모델 변경·양자화·컴파일·실칩 측정 없음. AGX·DFC 사용 안 함 (G5c가 같은 데스크탑에서 실행 중, CPU 6스레드 제한).

## 1. 대상

| 이름 | HF 모델 | 입력 | 토큰 | Zoo ONNX (sha256은 zip) |
|---|---|---|---|---|
| **B32** (주 대상) | `google/siglip2-base-patch32-256` | 256×256 | 8×8 = 64 | `siglip2-base-patch32-256_vision_encoder.zip` `6f6f5731…` (2025-05-21) |
| **S1B16** (대조군) | `google/siglip-base-patch16-224` | 224×224 | 14×14 = 196 | `siglip_base_patch16_224_vision_encoder.zip` `5342aaa6…` (2025-03-16) |
| **REF** (기준 모델) | `google/siglip2-base-patch16-224` (= 우리 `timm/ViT-B-16-SigLIP2`인지 4-1에서 확인) | 224×224 | 196 | — |

## 2. 측정 전에 확인한 사실 (코드·설정 읽기만)

### 2-1. Zoo v5.4.0 설정 (`hailo_model_zoo/cfg/networks/*.yaml`, `cfg/base/clip.yaml`, `cfg/alls/generic/*.alls`)

| | B32 | S1B16 |
|---|---|---|
| parser 시작 노드 | `onnx::Cast_0` (→ 첫 노드 Conv 패치 임베딩) | `input.1` (→ Conv) |
| parser 끝 노드 | `1393` = MAP 헤드 출력 `Gather(head/Add, 0)` (1×768, pooler_output) | `1725` = 같은 위치 |
| ONNX 두 번째 출력 `key` | 1×64×768 (파서 범위 밖) | 1×196×768 (파서 범위 밖) |
| ONNX opset / LN | 17, `LayerNormalization` 26개 | 14, LN이 ReduceMean·Sub·Pow·Sqrt·Div로 풀려 있음(26개) |
| 정규화 위치 | **HEF 안**: `.alls` 첫 줄 `normalization([127.5]*3, [127.5]*3)` → (x−127.5)/127.5. base `clip.yaml`의 `normalization_params`(CLIP 평균·표준편차)는 `.alls`가 대신한다 (G5c README 19행과 같은 해석). Zoo ONNX 자체에는 정규화가 없다 (입력이 곧바로 Conv). | 같음 |
| 16비트 층 (.alls) | `norm1, conv1, conv1_s2d, conv4, conv5, conv24, conv25, ew_add*` | `norm1, conv1, format_conversion1, conv3, conv4, conv5, conv25, ew_add*` |

### 2-2. 전처리 비교 (요청 6)

| | 리사이즈 | crop | 보간 | 정규화 |
|---|---|---|---|---|
| Zoo `clip` 전처리 (`classification_preprocessing.clip` → `pil_resize`) | 가로세로 비율 무시하고 H×W로 바로 | 없음 | PIL **BICUBIC**, uint8 | HEF 안 (x−127.5)/127.5 |
| HF B32 / REF (`SiglipImageProcessor`) | H×W로 바로 | 없음 | `resample=2` = PIL **BILINEAR** | (x/255 − 0.5)/0.5 = 같은 값 |
| HF S1B16 | H×W로 바로 | 없음 | `resample=3` = PIL BICUBIC | 같음 |

→ B32·REF는 보간 방식이 Zoo와 다르다. **규칙대로 캐시는 세 모델 모두 Zoo 쪽 전처리**로 만든다: `PIL.Image.convert("RGB").resize((W,H), BICUBIC)` → uint8 → `x/127.5 − 1`, float32 NCHW. 나중 HEF 비교에서도 이 캐시를 쓴다 (HEF 입력은 `x·127.5+127.5`로 되돌리면 정확히 원래 uint8이다).
기존 dev 캐시(timm `ViT-B-16-SigLIP2` 전처리 224)는 4-1 동일성 확인에만 쓴다.

### 2-3. 사진 (데이터 구간)

- **분석용 보정 사진**: 기존 보정 규칙 그대로 `s2_quantize.calib_images("highres", 256, seed=0)`와 **같은 사진**(고화질 5종 train 보정 후보, dev·test와 겹치지 않음)을 같은 순서로 골라 원본에서 Zoo 전처리로 다시 만든다. 3·4번 분석은 그 **앞 128장**, 상수 캐시(5번)는 M2rot과 같은 **256장**을 쓴다.
  재현 확인: 다시 만든 224 캐시를 timm 전처리로 만들었을 때 기존 `calib_*_f32.npy`와 같은 사진인지 (사진 순서 대조) assert.
- **dev 6×200**: Pets·Flowers·Cars·Food·Caltech dev와 CIFAR A, 각 앞 200장 (기존 dev 캐시와 같은 사진·같은 순서).
- **동일성 확인용 32장**: Pets dev 앞 16장 + Caltech dev 앞 16장 (G3_why와 같은 출처).
- test는 쓰지 않는다.

### 2-4. 우리 M2rot(D256L1)에 실제로 적용된 상수 캐시 범위 (요청 3, `siglip2/s2_regcache.py`에서 확인)

M2rot = `s2_regcache.py --graph mask --mask-form bcast --mask-site mlp --rot rothad --sinks 55@5,13@9 --calib-n 256 --seed 0` (G3RQ README 11행, `MaskRegCacheTrunk`).

| 항목 | 실제 적용 |
|---|---|
| sink | `(p, b)` = (패치 55, 블록 5), (패치 13, 블록 9). 둘 다 G3_why에서 **그 블록의 MLP(fc2)가 거대 값을 써 넣는 첫 블록**이다 |
| attention K/V | 블록 l ≥ b의 self-attention에서 위치 p의 K(`k_norm` 뒤)·V를 상수로 바꿔 끼운다 (토큰 수 유지, `_replace_kv`). 활성 sink는 블록 l에서 b ≤ l인 것 |
| MAP 헤드 | 풀링 헤드 kv에서 **모든 sink 위치**의 K·V를 상수로 바꾼다 |
| MLP 입력 (M2) | 블록 l ≥ b에서 sink 행의 `norm2` 출력(affine 포함)을 `u_c[l]`로 바꾼다. `u_c[l]` = 원래 모델 블록 l `norm2` 출력의 보정 사진·비-sink 토큰 평균 |
| 잔차 | 건드리지 않는다 (`mask_site=mlp`이면 잔차 마스크 없음) |
| 상수 | 원래 FP32 모델에 보정 256장(highres, 시드 0)을 넣어 블록·헤드별 sink K·V 사진 평균 (`build_mean_cache`), `u_c` (`build_uc`) |
| 회전 | rothad는 FP32에서 수학적으로 같은 변환이라 (G3Q·G3RQ에서 확인) FP32 cos에는 영향이 없다 → 5번은 **회전 없이 M2만** 잰다 |

**새 모델로 옮기는 규칙 (결과 전 고정):**
- sink `(p, b)`: 패치 p가 블록 b 출력에서 high-norm 토큰(토큰 L2 노름 > 중앙값의 5배, G3_why 정의)인 사진이 128장 중 90% 이상이고, b는 그런 첫 블록. 그리고 p가 **어떤 블록 l > b**에서 attention 상위 2개 고정성(4절 기준) ≥ 90%여야 한다. 둘을 모두 만족하는 (p, b)를 모두 sink로 쓴다.
- 이 규칙이 REF에서 정확히 {55@5, 13@9}를 내는지 먼저 확인한다 (6절 기준 모델 검증). 다르게 나오면 REF에서 {55@5, 13@9}를 내도록 고친 규칙을 대상 모델을 보기 전에 커밋한다.
- K/V·MAP 헤드·u_c·보정 256장 처리는 위 표 그대로. HF 구조의 대응: `layer_norm1`→`norm1`, `self_attn.k_proj/v_proj`→`qkv`의 K·V (HF SigLIP에는 q/k norm 없음), `layer_norm2`→`norm2`, `post_layernorm`→`trunk.norm`, `head`(MultiheadAttentionPoolingHead)→`attn_pool`.
- 패치 번호는 위치 임베딩과 같은 행 우선(row-major) 순서다 (패치 55 = 행 3, 열 13 / 14×14).
- **판정에 쓰는 5번 값 = M2 전체 (K/V 상수 + MAP 헤드 + u_c)**. "K/V만 바꾼 것"(u_c 없음)은 기록용으로 같이 잰다.
- 구현 검증: (i) oracle(그 사진의 실제 K·V, u_c 대신 실제 norm2 출력)로 바꾸면 원래와 cos 최소 ≥ 0.99999. (ii) REF에 기존 timm 전처리 캐시와 기존 상수 파일(`regcache_55at5-13at9_highres256_s0.npz`, `regcache_uc_…npz`)을 넣었을 때 내 HF 구현의 출력이 timm `MaskRegCacheTrunk`(bcast, mlp)와 cos 최소 ≥ 0.99999. 실패하면 멈춘다.

## 3. 측정 (요청 1~6)

- 모든 HF 모델은 `attn_implementation="eager"`, FP32, `torch.set_num_threads(6)`, `OMP/MKL/OPENBLAS_NUM_THREADS=6`. 시간 측정 없음.
- **1번 가중치 동일성**: Zoo ONNX(ORT 1.25 `ORT_ENABLE_ALL`, CPU, 출력 `1393`/`1725`) vs HF `vision_model(...).pooler_output`, 2-3의 32장 (Zoo 전처리 캐시). 평균 cos < 0.9999면 **멈추고 보고**.
- **4-1 (요청 4) REF 동일성**: HF `google/siglip2-base-patch16-224` vs 우리 timm FP32(`S.load_model()` `encode_image`, 정규화 전)의 출력 cos, 같은 32장 (기존 timm 전처리 dev 캐시를 두 모델에 똑같이 넣음). 평균 cos < 0.9999면 **멈추고 보고**.
- **2번 구조표**: pre-LN 여부(잔차에 더하기 전 LN을 거치는지), LN 개수, MAP 헤드 유무, CLS 토큰 유무, 위치 임베딩 방식, 블록 수, hidden. 회전을 LN에 접는 우리 방식(`s2_rotate.check_trunk` 조건: 잔차를 읽는 곳이 LN뿐, 더하는 곳이 patch_embed·pos_embed·attn out_proj·fc2뿐)이 맞는지 코드로 확인.
- **3번 거대 활성값** (128장): 블록 출력 잔차 h (토큰 × 768)에서 블록별
  - `max|x|`, `median|x|` (그 블록 128장 전체 원소), **ratio = max|x| / median|x|** ← 판정
  - 상위 5개 채널(채널별 max|x|), 상위 5개 토큰 위치(위치별 max|x|), 기록용으로 G3_why 지표(토큰 노름 max/중앙값, high-norm 토큰 수·위치)
- **4번 sink** (128장): 블록·헤드별 attention 확률을 query 평균해 key 토큰별 질량 (사진마다). MAP 헤드도 같은 방식.
  - **고정성 = 블록 l에서 헤드 평균 질량의 상위 2개 안에 위치 p가 드는 사진 비율** ← 판정. 헤드별 값은 기록용.
  - K/V 일관성: sink 위치 K·V의 사진 평균 대비 cos (블록·헤드별 중앙값) — **기록용만** (요청 반영 1).
- **5번 상수 캐시 FP32**: 2-4 규칙으로 정한 sink에 M2를 적용한 FP32 모델 vs 원래 FP32의 임베딩 cos, dev 6×200 (1,200장). **판정 값 = 1,200장 평균 cos**. 데이터셋별 평균·최소, top-1 변화도 기록.
- **6번 FP32 top-1**: HF 텍스트 인코더(FP32, `padding="max_length", max_length=64`), 템플릿·클래스 이름은 기존 dev와 같은 것(`rn50/common.TEMPLATES`, `rn50/datasets` 템플릿), 앙상블은 `S.zeroshot_classifier`와 같은 방식(문장 정규화 → 평균 → 정규화). dev 6×200 top-1. 30분 안에 안 되면 건너뛰고 이유 기록.

## 4. 판정 (결과 전 고정)

| 항목 | 규칙 |
|---|---|
| 회전 적용 | 2번에서 pre-LN이고 LN 접기가 가능하면 **가능** |
| 거대 활성값 | 어느 블록이든 ratio ≥ 100이면 **있음**, 모든 블록 ≤ 20이면 **없음**, 그 사이 **약함** |
| sink 적용 | 2-4 규칙으로 sink가 1개 이상 나오고(고정성 ≥ 90%), 5번 M2 cos 평균 ≥ 0.999이면 **가능**. 하나라도 실패하면 **불가**로 두고 어느 조건에서 실패했는지 적는다. K/V cos(평균 대비)는 판정에 쓰지 않는다 |
| 결론 한 줄 | 회전 가능 & sink 가능 → **M2rot 전체 적용** / 회전 가능 & sink 불가 & 거대 활성값 있음·약함 → **회전만 적용** / 거대 활성값 없음 → **적용 이점 없음** (회전·sink 모두 거대 활성값 대책이므로) |

**기존 기준을 쓰지 않은 이유**: G3_why(high-norm = 노름 > 중앙값 5배)는 2-4의 sink 찾기 규칙에 그대로 썼다. RegCache G0의 K/V cos 0.95는 기록용이었고, REF도 블록 9 V가 0.39인데 M2rot이 작동하므로 판정에서 뺐다 (요청 반영 1). G1b의 FP32 cos ≥ 0.99는 요청한 0.999보다 느슨해서 요청 값을 쓴다.

## 5. 기준 모델 검증 규칙 (요청 반영 2, 결과 전 고정)

1. **REF를 먼저 잰다** (3·4·5번, 2-4 sink 규칙 확인). 대상 모델(B32·S1B16)의 3~5번 결과는 그 전에 보지 않는다. 1번 동일성 확인은 가중치 확인이라 먼저 해도 된다.
2. REF가 통과하지 못하는 문턱은 **REF 값으로 바꾼다**:
   - 거대 활성값 "있음" 문턱: REF 최대 ratio < 100이면 → REF 최대 ratio (소수 첫째 자리 내림)로
   - sink 고정성 문턱: REF의 55@5·13@9가 90% 미만이면 → 둘 중 낮은 값으로
   - 상수 캐시 cos 문턱: REF M2 cos 평균 < 0.999이면 → REF 값 (소수 넷째 자리 내림)으로
   - sink 찾기 규칙이 REF에서 {55@5, 13@9}를 정확히 내지 않으면 → 2-4대로 규칙을 고친다
3. 바꾼 값과 이유를 6절에 적고 **대상 모델을 측정하기 전에 커밋**한다. 바꿀 것이 없어도 "바꾸지 않음"을 커밋한다.

## 6. 기준 모델(REF) 결과와 문턱 조정

(측정 후 기록)
