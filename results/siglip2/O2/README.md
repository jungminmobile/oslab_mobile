# O2: B16 ORT 모델에 통일 sink 규칙 적용 — B16·B32 ORT 설계 맞추기 (설계 2026-10-22)

> **용 요청. 판정 기준은 결과를 보기 전에 고정한다.** dev만 쓴다 (test 사용 안 함). CPU만 쓴다 (DFC·AGX 안 씀).
> 이 PC(연구실 데스크탑 WSL) 측정은 **비공식**이다.

## 0. 요약 (결과 뒤 채움)

## 1. 설계 대조: B16 O-AB(`db2c1fdd…`) vs B32 O-AB(`7a01044c…`)
실제 로그·파일·코드로 확인했다 (근거 열).

| 항목 | B16 O-AB | B32 O-AB | 같은가 | 근거 |
|---|---|---|---|---|
| 원본 모델 | open_clip timm `hf-hub:timm/ViT-B-16-SigLIP2` (224, 패치 16, 토큰 196) | `hf-hub:timm/ViT-B-32-SigLIP2-256` (256, 패치 32, 토큰 64) | 모델 고유 | `s2_g6.MODEL_IDS` |
| MLP GELU | 정확한 GELU (ONNX `Erf`) | tanh 근사 (ONNX `Tanh`) | 모델 고유 | O1 6-2 노드 수 |
| 회전 | rothad, Q sha256 `3a6391a2…` (`apply_rot`) | rothad, Q `3a6391a2…` (`apply_rot`) | **같음** | G3RQ `cache.jsonl`·G6b `export.jsonl` |
| sink 패치 | 55, 13 | 15, 7 | 모델 고유 | G6_pre |
| **K/V 상수 시작 블록** | **5, 9** | **6, 10** | **다름** | G3RQ `export.jsonl` `sinks 55@5,13@9` / `s2_g6.UNIFIED` |
| u_c(MLP 입력) 시작 블록 | 5, 9 (`u_starts` 지정 안 함 → K/V 시작과 같게) | 5, 9 | 숫자는 같음, **규칙은 다름** (B16: u_c = K/V 시작, B32: K/V = u_c + 1) | `MaskRegCacheTrunk.__init__` |
| MAP 헤드 | `pool` 키: 모든 sink의 K·V를 보정 평균 상수로 | 같음 | **같음** (같은 클래스) | `MaskRegCacheTrunk` |
| M2 그래프 | `MaskRegCacheTrunk` 제자리 마스크 `bcast` + MLP 입력 u_c (`mask_site mlp`, 잔차 그대로) | 같은 클래스·같은 형식 | **같음** | G3RQ `export.jsonl` options / `s2_g6.m2_trunk` |
| M2 상수 계산 | `build_mean_cache` + `build_uc`, 회전한 모델, 배치 16, 256장 | 같은 함수, 같은 순서 | **같음** | `s2_regcache` cache·ucache / `s2_g6.m2_trunk` |
| M2 상수 보정 사진 | `calib_images("highres", 256, 0)` (timm 전처리 224), 픽셀 sha256 앞 16자 `774f2967d9866178` | Zoo 전처리 256 `cache_zoo256/calib256_s0_x.npy` (같은 사진·같은 순서), `f296caa848de38a0` | **같은 사진, 크기만 모델에 맞춤**. 224에서는 timm 캐시 = Zoo 224 캐시 (픽셀 차이 0, 이번에 확인) | 이번 확인 |
| 상수 저장 | npz 파일 (`regcache_…_rothad_highres256_s0.npz` `c205e3a0…`, `regcache_uc_…` `00d8ee3d…`) 후 로드 | export 중 메모리에서 바로 | 값은 같은 계산 (float32 저장) | |
| PyTorch 스레드 | 4 (`s2_regcache --threads 4`) | 6 (`s2_g6`) | 사소한 차이 (부동소수 합 순서) | |
| ONNX export | `torch.onnx.export` opset 17, dynamo 끔, 배치 가변 (`s2_regcache export`) | 같은 호출 (`s2_g6 export --cfg m2rot --dynamic`) | **같음** | |
| FP32 M2rot 파일 | `siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx` `16ef696e…` (커밋 `f64c9ae`) | `siglip2_image_fp32_g6b_b32_m2rot.onnx` `23236123…` (커밋 `e7cd97c`) | | |
| **FP32 동치 확인 방법** | (a) 회전한 M2 vs 회전 없는 M2 cos 최소 0.9999998 (64장), ORT vs PyTorch (CIFAR 64장) | 독립 HF M2 구현 vs PyTorch (dev 64장) 0.9999999988, ORT vs PyTorch | **다름** | G3RQ `m2rotcheck.jsonl` / G6b `export.jsonl` |
| 양자화 옵션 | U8 + reduce_range, 채널별 S8 가중치, percentile 99.999 비대칭, highres 256장 시드 0, hc16, op MatMul, `quant_pre_process`, QDQ | 같음 | **같음** | O1 `quant/quantize_log.jsonl` |
| 양자화 보정 사진 | `calib_images` (위와 같은 사진, 224) | `--calib-npy cache_zoo256/calib256_s0_x.npy` (같은 사진, 256) | **같은 사진** | 위 |
| O-B | `s2_ortfuse --skip-ln --fold-shape` (SkipLN 24, bias 23, Reshape 28) | 같음 | **같음** | O1 `ortfuse_log.jsonl` |
| O-A | `--no-matmul-out-q` | 같음 | **같음** | O1 |
| 평가 FP32 기준 | Release `siglip2_image_fp32.onnx` `333cad08…` (`s2_export_onnx`) | `g6_b32_plain_dyn` `be19a8ad…` (`s2_g6 export --cfg plain`, undedup) | export 코드 다름 (기준 파일용, 모델 자체는 원본) | G6b |
| 평가 코드 | `s2_g6b_ort.py`, Zoo 224 dev 캐시, HF 텍스트 분류기 | 같은 코드, Zoo 256 | **같음** | O1 |

**다른 항목 (모델 고유가 아닌 것):** ① K/V 시작 블록(5·9 vs 6·10)과 u_c와의 관계, ② FP32 동치 확인 방법, ③ PyTorch 스레드 수(사소), ④ 평가용 FP32 기준 파일의 export 코드(모델 설계와 무관). 보정 사진·회전·M2 형식·MAP 헤드·양자화·O-A·O-B는 같다.

## 2. B16 통일 규칙 모델 (이번에 만든 것)
- `s2_g6.py`의 `UNIFIED`에 `b16: 55@6:5, 13@10:9`(G6_pre 13-2 REF 통일 규칙)을 넣고 `export --model b16 --cfg m2rot --dynamic`을 허용했다. B32 경로는 그대로다 (회귀: B32를 다시 export해 sha256 `23236123…` 같음, `regress/export_sha256.txt`).
- M2 상수: Zoo 224 보정 256장(= 기존 B16 보정 사진과 픽셀 동일) 평균, 회전한 모델에서. 기존 B16·B32와 같은 방식.
- 결과: `siglip2_image_fp32_o2_b16_m2rotu.onnx` (`$SIGLIP2_ARTIFACTS/onnx/`, `~/siglip2_work/g6/onnx/g6_b16_m2rot_dyn.onnx`와 같은 파일), sha256 `341a4027…`, 커밋 `9e588c8`.
- 양자화 (커밋 `c949da2`): 기존 M2rot_ort 설정 → `siglip2_image_int8_o2_b16_m2rotu_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx` (`e4cdee15…`, "통일 기존설정"); O-AB 설정(`s2_ortfuse` → `--no-matmul-out-q`) → `…_o2_b16_m2rotu_o1b_…_rr_moq0_all.onnx` (`f5108c7b…`, "통일 O-AB").

### FP32 동치 (oracle) — 해석과 결과
- 계획에서 정한 해석: 독립 HF 구현(`s2_g6pre.hf_forward`, 같은 sink·같은 상수 정의)과 PyTorch 통일 M2rot의 cos 최소 ≥ 0.99999 (dev 64장).
- **HF SigLIP2 설정 그대로(tanh GELU)면 0.9999725로 기준 아래**다. 원인: timm B16은 정확한 GELU(erf), HF 설정은 tanh 근사다. M2 없는 원래 모델끼리도 0.9999777이다 (G6_pre `m2xchk`도 같은 값 0.99998). B32는 timm도 tanh라 이 차이가 없다.
- **HF MLP 활성함수를 정확한 GELU로 바꾼 oracle: cos 최소 0.99999999999 → 통과** (`export/oracle.jsonl`). 원래 모델끼리도 0.9999999999999. 즉 M2 구현은 독립 구현과 같고, 0.99997은 GELU 근사 차이뿐이다.
- ORT vs PyTorch: 0.99999999998.
- 통일 FP32 ONNX vs Release FP32: cos 최소 0.99982 / 평균 0.99996. vs 기존 M2rot FP32: 최소 0.99938 / 평균 0.99995 (dev 64장). dev 전체 값은 6절.

## 3. 측정
- 정확도: dev 전체(고화질 5종 4,923장 + CIFAR A 1,000장), O1과 같은 `s2_g6b_ort.py`. 모델 7개: Release FP32, 기존 M2rot FP32, 통일 FP32, 기존 M2rot_ort(`e84c3727…`), 기존 O-AB(`db2c1fdd…`), 통일 기존설정, 통일 O-AB.
- 쌍 비교 95% 구간 (`boot_hr`): 통일 O-AB − 기존 O-AB, 통일 기존설정 − 기존 M2rot_ort, 각 − Release FP32. CIFAR A 따로.
- 지연: 기존 O-AB와 통일 O-AB를 번갈아, 스레드 1·2·4 × 3세션 (`s2_g6b_bench.py --warmup 20 --n 200 --repeats 3`, O1과 같은 게이트). 세션별 평균의 중앙값. peak 메모리(`s2_model_stats.py`)와 크기도 적는다.

## 4. 판정 (결과 전 고정, 용 지시 그대로 + 계획 보고 때 정한 해석)
- **"통일 규칙 채택"**: 아래를 모두 만족
  1. (통일 O-AB − 기존 O-AB) 고화질 Δtop-1이 "동률" 이상 = 95% 구간 상한 ≥ 0
  2. CIFAR A 차이(통일 O-AB − 기존 O-AB) ≥ −1.0%p
  3. 지연 차이가 t1·t2·t4 **각각** ±3% 이내 (3세션 중앙값)
  → 이 경우 두 모델의 ORT 설계가 같아진다.
- 열위면 B16은 기존 규칙을 유지하고, "B16은 옛 규칙, B32는 통일 규칙"과 그 이유를 적는다.

## 5. 구현 (커밋)
- `9e588c8` `s2_g6.py` B16 통일 규칙 (`UNIFIED["b16"]`, `UNIFIED_TAG`, `m2_trunk(key)`), `o2_export.sh`
- `c949da2` `o2_oracle.py` 결과, `o2_quant.sh`
- `244e7bd` `o2_models.sh`, `o2_run.sh`

## 6. 결과
(측정 뒤 채움)
