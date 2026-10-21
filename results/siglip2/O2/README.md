# O2: B16 ORT 모델에 통일 sink 규칙 적용 — B16·B32 ORT 설계 맞추기 (설계 2026-10-22)

> **용 요청. 판정 기준은 결과를 보기 전에 고정한다.** dev만 쓴다 (test 사용 안 함). CPU만 쓴다 (DFC·AGX 안 씀).
> 이 PC(연구실 데스크탑 WSL) 측정은 **비공식**이다.

## 0. 요약

**판정: 통일 규칙 채택** (4절 세 조건 모두 만족). B16 ORT도 B32와 같은 통일 sink 규칙(K/V는 입력이 high-norm인 블록부터, u_c는 high-norm이 되는 블록 MLP부터)을 쓸 수 있다. 바꾸면 두 모델의 ORT 설계가 같아진다 (1절의 다른 항목 ①이 없어지고, 남는 차이는 확인 방법·스레드 수 같은 만드는 과정의 차이뿐).

| 구성 (B16, dev 전체, 비공식) | 지연 t1 / t2 / t4 ms (3세션 중앙값) | FP32 대비 고화질 Δ [95%] | CIFAR A (FP32 대비) | 크기 MiB |
|---|---|---|---|---|
| 기존 O-AB (`db2c1fdd…`, 55@5,13@9) | 81.18 / 44.39 / 25.71 | −0.16 [−0.55, +0.19] | 74.8 (−2.1) | 91.65 |
| **통일 O-AB** (`f5108c7b…`, 55@6:5,13@10:9) | 80.60 / 43.87 / 25.32 (**−0.7 / −1.2 / −1.5%**) | −0.24 [−0.63, +0.10] | 75.3 (−1.6) | 91.63 |
| 짝 비교: 통일 O-AB − 기존 O-AB | | **−0.08 [−0.26, +0.09]** (동률) | **+0.5** | |

- 조건 ① 상한 +0.09 ≥ 0 → 통과, ② CIFAR A +0.5 ≥ −1.0 → 통과, ③ 지연 −0.7 / −1.2 / −1.5% (모두 ±3% 안) → 통과.
- 통일 기존설정 − 기존 M2rot_ort = +0.00 [−0.24, +0.25] (CIFAR A −0.5): O-AB가 아닌 설정에서도 동률이다.
- 통일 FP32는 Release FP32와 거의 같다: +0.06 [−0.02, +0.16], cos 평균 0.99998 / 최소 0.9989 (dev 전체). 기존 M2rot FP32는 +0.10 [+0.00, +0.22], cos 최소 0.9969.
- **교체 여부는 팀이 정한다.** 교체하면 예선 B16 ORT 후보가 `db2c1fdd…`(Release `siglip2-oab-20261022`)에서 `f5108c7b…`로 바뀐다. 새 파일은 아직 Release에 올리지 않았다. B16 Hailo HEF(M2rot_D256L1, 55@5,13@9 규칙)는 이번에 건드리지 않았으므로, 교체하면 B16은 "ORT 통일 규칙 / Hailo 옛 규칙"이 된다 (9절).

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

### 6-1. 회귀·동일성
- `s2_g6.py` 변경 뒤 B32 m2rot을 다시 export: sha256 `23236123…` 같음 (`regress/export_sha256.txt`, 커밋 `9e588c8`).
- 정확도 실행의 Release FP32·기존 M2rot_ort·기존 O-AB 임베딩은 O1 저장 임베딩과 비트 단위로 같다 (`regress/emb_vs_o1.txt`).
- 통일 FP32의 O-B 정리(`s2_ortfuse`): SkipLN 24(bias 23), Reshape 28, FP32 동치 64장 cos 최소 0.99999988, 최대 차이 9.5e-6 (`ortfuse/ortfuse_log.jsonl`).

### 6-2. 정확도 (07:24~07:56, `o2_run.sh`, 커밋 `244e7bd` dirty 아님, `accuracy/ort_accuracy.jsonl`)
- 시작 전 다른 세션 지연 측정 0개 → 대기 없음 (`eval_wait.log`).

| 구성 | CIFAR A | Pets | Flowers | Cars | Food | Caltech | 고화질 top-1 |
|---|---|---|---|---|---|---|---|
| Release FP32 (`333cad08…`) | 76.90 | 93.33 | 86.73 | 93.80 | 89.44 | 95.02 | 91.66 |
| 기존 M2rot FP32 (`16ef696e…`) | 76.90 | 93.54 | 86.73 | 94.11 | 89.44 | 95.02 | 91.77 |
| 통일 FP32 (`341a4027…`) | 76.90 | 93.33 | 86.83 | 93.90 | 89.44 | 95.12 | 91.72 |
| 기존 M2rot_ort (`e84c3727…`) | 72.60 | 92.93 | 86.31 | 93.80 | 89.04 | 95.12 | 91.44 |
| 기존 O-AB (`db2c1fdd…`) | 74.80 | 93.23 | 86.31 | 94.00 | 88.83 | 95.12 | 91.50 |
| 통일 기존설정 (`e4cdee15…`) | 72.10 | 93.03 | 86.42 | 93.90 | 89.04 | 94.82 | 91.44 |
| 통일 O-AB (`f5108c7b…`) | 75.30 | 92.83 | 86.21 | 94.31 | 88.73 | 95.02 | 91.42 |

| 짝 비교 (A − B) | 고화질 Δ [95%] | CIFAR A Δ | cos 평균 / 최소 (고화질) | cos (CIFAR A) |
|---|---|---|---|---|
| **통일 O-AB − 기존 O-AB** | **−0.08 [−0.26, +0.09]** | **+0.5** | 0.9987 / 0.9873 | 0.9982 |
| 통일 기존설정 − 기존 M2rot_ort | +0.00 [−0.24, +0.25] | −0.5 | 0.9949 / 0.9808 | 0.9938 |
| 기존 M2rot FP32 − Release FP32 | +0.10 [+0.00, +0.22] | +0.0 | 0.9999 / 0.9969 | 0.9995 |
| 통일 FP32 − Release FP32 | +0.06 [−0.02, +0.16] | +0.0 | 1.0000 / 0.9989 | 0.9998 |
| 기존 M2rot_ort − Release FP32 | −0.23 [−0.64, +0.18] | −4.3 | 0.9757 / 0.8534 | 0.9131 |
| 기존 O-AB − Release FP32 | −0.16 [−0.55, +0.19] | −2.1 | 0.9869 / 0.9149 | 0.9501 |
| 통일 기존설정 − Release FP32 | −0.22 [−0.62, +0.17] | −4.8 | 0.9756 / 0.8582 | 0.9126 |
| 통일 O-AB − Release FP32 | −0.24 [−0.63, +0.10] | −1.6 | 0.9869 / 0.9135 | 0.9501 |

- 통일 FP32는 Release FP32에 기존 M2rot FP32보다 더 가깝다 (cos 최소 0.9989 vs 0.9969). K/V를 한 블록 늦게(입력이 이미 high-norm인 블록부터) 바꾸기 때문으로 본다 (G6_pre 2단계와 같은 방향, 원인 확인 안 함).

### 6-3. 지연 (08:01~08:21, `o2_ort_speed.sh` = O1 스크립트에서 결과 폴더만 바꿈, 커밋 `8c834b4` dirty 아님, `speed/runs.jsonl` 9줄, **비공식**)
- 게이트 (`speed/gate.log`): 대기 4회, 모두 직전 실행이 올린 부하 (다른 프로세스 0). Windows 쪽 부하는 확인 못 함.

| 스레드 | 기존 O-AB 세션별 ms | 통일 O-AB 세션별 ms | 중앙값 기존 / 통일 | 차이 |
|---|---|---|---|---|
| 1 | 81.29, 80.99, 81.18 | 80.86, 80.59, 80.60 | 81.18 / 80.60 | −0.71% |
| 2 | 44.39, 44.46, 44.28 | 43.87, 43.83, 43.89 | 44.39 / 43.87 | −1.17% |
| 4 | 25.71, 26.55, 25.60 | 25.32, 25.80, 25.30 | 25.71 / 25.32 | −1.52% |

- 모든 세션에서 통일 쪽이 조금 빠르다. 그래프 구조가 같고 K/V 상수를 쓰는 블록이 하나 적어서(블록 5에 K/V 마스크 연산이 없어짐)로 보지만, 연산 수·프로파일로 확인하지 않았다.
- 기존 O-AB의 절대값(t4 25.71 ms)은 O1(26.21 ms)과 다르다. 실험끼리 절대값은 비교하지 않는다.

### 6-4. 크기·peak 메모리 (08:26, `s2_model_stats.py` 4스레드, 모델마다 새 프로세스, 1회, `model_stats/model_stats.jsonl`)
| 구성 | 크기 MiB | peak 추론 MiB |
|---|---|---|
| 기존 O-AB | 91.65 | 216.1 |
| 통일 O-AB | 91.63 | 240.8 |
- 1회 측정이다. 같은 기존 O-AB 파일이 O1에서는 234.5 MiB였다 (이번 216.1). 측정 사이 흔들림이 ±10% 정도라 이번 차이(+11%)는 판정할 수 없다. 판정 항목도 아니다.

## 7. 판정 (4절 그대로)
| 조건 | 값 | 결과 |
|---|---|---|
| ① 통일 O-AB − 기존 O-AB 고화질 상한 ≥ 0 | −0.08 [−0.26, **+0.09**] | 통과 |
| ② CIFAR A 차이 ≥ −1.0%p | +0.5 | 통과 |
| ③ 지연 차이 t1·t2·t4 각각 ±3% 이내 | −0.71 / −1.17 / −1.52% | 통과 |

**→ "통일 규칙 채택".** B16 ORT를 통일 규칙으로 만들면 B16·B32의 ORT 설계(회전·sink 규칙·M2 형식·MAP 헤드·보정 사진·양자화·O-A·O-B)가 같아진다. 실제 예선 후보 교체(`db2c1fdd…` → `f5108c7b…`)는 팀이 정한다 (정민 노트북 공식 측정 필요, test 재사용 문제는 O1 9절과 같음).

## 8. 계획에서 달라진 점
1. **oracle 해석 (사후 보완):** 계획의 "독립 HF 구현 cos ≥ 0.99999"는 B16에서 GELU 근사 차이(timm erf vs HF tanh) 때문에 원래 모델끼리도 넘을 수 없다 (0.99998, G6_pre와 같음). 그래서 HF MLP 활성함수를 timm과 같은 정확한 GELU로 바꾼 비교를 oracle로 썼다 (0.99999999999). HF 설정 그대로의 값(0.9999725)도 같이 적었다. 이 판단은 결과(정확도)를 보기 전, export 직후에 했다.
2. 정확도 실행에 기존 M2rot FP32(`16ef696e…`)를 추가했다 (FP32끼리 비교용, 판정에는 안 씀).
3. peak 메모리·크기는 요청에 없었지만 적었다 (판정에 안 씀).

## 9. 확인 못 한 것 / 팀이 정할 것
- **Hailo B16:** 이번은 ORT만이다. B16 Hailo HEF는 옛 규칙(55@5,13@9, M2rot_D256L1)이다. B16 ORT를 통일 규칙으로 바꾸면 B16의 ORT와 Hailo 규칙이 달라진다. Hailo도 통일 규칙으로 다시 만들지는 팀이 정한다 (DFC·실칩 필요, 이번 범위 밖).
- 통일 O-AB가 조금 빠른 원인 (연산 수·프로파일 안 봄), peak 메모리 차이 (1회).
- 통일 O-AB 파일은 Release에 올리지 않았다.
- test는 쓰지 않았다.

## 10. 재현
```bash
bash results/siglip2/O2/o2_export.sh        # B32 회귀 + B16 통일 FP32 export
(cd siglip2 && RN50_RESULTS=../results/siglip2/O2/export python ../results/siglip2/O2/o2_oracle.py)
bash results/siglip2/O2/o2_quant.sh         # 기존 설정·O-AB 양자화
bash results/siglip2/O2/o2_run.sh           # 정확도 → 지연 3세션 → 메모리
```
