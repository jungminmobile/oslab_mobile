# O1: ORT CPU 지연 줄이기 — B16 M2rot_ort·B32 INT8에 O-A·O-B·O-AB (설계 2026-10-22)

> **용 요청. 결과를 보기 전에 고정한다.** 판정 기준은 결과를 본 뒤 바꾸지 않는다. **dev만 쓴다 (test 사용 안 함).**
> 이 PC(연구실 데스크탑 WSL) 측정은 **비공식**이다. 동결(10/28) 전 마지막 ORT 개선 실험이다.

## 0. 메인 모델 결정용 요약 (결과 뒤 채움)

(6절 결과를 본 뒤 채운다: 두 모델의 자기 FP32 대비 ORT 배속 t1·t2·t4와 FP32 대비 고화질 손실, 기준값과 최선 후보를 나란히)

## 1. 질문과 배경
- 정확도는 그대로 두고 ORT CPU 지연만 줄이는 두 후보를 **B16 M2rot_ort**(`e84c3727…`)와 **B32 INT8**(G6b, `3a46ae8a…`)에 같은 방식으로 적용한다. 판정은 모델별로 따로 낸다.
- G8 프로파일(B16, 4스레드): 동적 양자화보다 5.3 ms 느린 원인은 가중치 MatMul 둘레의 독립 Q/DQ 3.5 ms, 합쳐지지 않은 bias Add 1.0 ms, Transpose 0.5 ms. 정수 행렬곱 자체는 같다(18.9 ms).
- 양자화 로그: SkipLayerNorm 융합 25개 중 1개, Transpose 53, Squeeze 38, Gather 28.

## 2. 후보 (모두 기본 꺼짐 옵션)
| 이름 | 만드는 법 |
|---|---|
| 기준 | B16: `siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx` (`e84c3727…`). B32: `siglip2_image_int8_g6b_b32_m2rot_…_rr_all.onnx` (`3a46ae8a…`). 다시 만들지 않는다 |
| **O-A** | 같은 FP32·보정 사진·설정 + `s2_quantize.py --no-matmul-out-q` (extra_options `OpTypesToExcludeOutputQuantization=["MatMul"]`): MatMul 출력에 Q/DQ를 붙이지 않아 ORT가 DQ→MatMul을 `MatMulIntegerToFloat`로 합치게 한다 |
| **O-B** | 양자화 전 FP32 그래프 정리 `s2_ortfuse.py --skip-ln --fold-shape` (아래) → 같은 설정으로 양자화 |
| **O-AB** | O-B FP32 + `--no-matmul-out-q` |
| FP32 | B16 Release `siglip2_image_fp32.onnx` (`333cad08…`), B32 `siglip2_image_fp32_g6b_b32.onnx` (`be19a8ad…`, 회전 없음) — G6b와 같다 |

공통 양자화 옵션 (기준과 같음): `--calib-source highres --calib-n 256 --seed 0 --method percentile --percentile 99.999 --act-type u8 --reduce-range --op-types MatMul --hist-chunk 16 --tag all`. B32는 `--calib-npy ~/siglip2_work/g6pre/cache_zoo256/calib256_s0_x.npy` (G6b와 같은 사진의 256 캐시).

### O-B 원인과 변환 (ORT 최적화 그래프를 저장해 확인)
- **SkipLayerNorm이 안 합쳐지는 원인:** 잔차 Add의 출력이 LayerNorm과 **다음 잔차 Add 두 곳**에서 쓰인다. ORT C++ `SkipLayerNormFusion`은 Add 출력의 소비자가 하나일 때만 합치므로, 소비자가 하나뿐인 마지막 `/trunk/norm`만 합쳐진다. 회전·M2 때문이 아니다: Release FP32(`333cad08…`)도 똑같이 25개 중 1개만 합쳐진다.
- **변환 (1) `--skip-ln`:** ORT python 융합기(`onnxruntime.transformers`)의 `FusionSkipLayerNormalization`은 소비자가 여럿이어도 SkipLayerNormalization의 4번째 출력(`input_skip_bias_sum`, Add의 합)으로 처리한다. 이어서 `FusionBiasSkipLayerNormalization`으로 앞 MatMul(proj·fc2)의 bias Add(1차원 상수)를 SkipLayerNorm의 bias 입력으로 흡수한다. 수학적으로 같은 연산이다(덧셈 순서만 커널 안으로).
  - 남는 LayerNorm 2개: 블록 0 norm1(입력이 위치 임베딩 상수와의 Add), MAP 헤드 norm(상수 probe와의 Add). 융합기가 상수 입력 Add는 합치지 않는다.
- **변환 (2) `--fold-shape`:** Reshape의 모양 입력이 Shape→Gather→Unsqueeze→Concat으로 계산된다(배치 가변 export). 기호 모양 추론으로 출력 모양이 `[batch, 고정값…]`이고 입력 0번 축도 `batch`일 때만 상수 `[0, 고정값…]`으로 바꾼다(Reshape의 0 = 입력 축 크기 그대로, allowzero=0). 쓰이지 않게 된 모양 계산 노드를 지운다.
- **손대지 않은 것:** Split→Squeeze→Transpose(q·k·v 분리와 head 축 배치)는 attention에 필요한 실제 데이터 이동이라 같은 수의 연산으로만 바꿀 수 있어 그대로 둔다. Transpose 53·Squeeze 38은 그대로다.
- **FP32 동치 기준 (결과 전 고정):** 같은 입력 64장(pets_dev, Zoo 전처리 캐시), 배치 1과 배치 4에서 출력 cos 최소 ≥ 0.99999. 최대 절대 차이를 기록한다. 실패하면 양자화하지 않는다 (`o1_quant_b.sh`가 assert).
- O-B는 ORT 전용 그래프다. Hailo 쪽(B16 M2rot_D256L1·B32 HEF)은 건드리지 않는다.

### 회귀 확인 (옵션을 끄면 이전과 같아야 함)
- 옵션을 모두 끈 `s2_quantize.py`로 별도 폴더(`$SIGLIP2_ARTIFACTS/o1_qregress`, FP32는 심볼릭 링크)에서 다시 만들어 sha256 비교.
- `s2_profile_ort.py --zoo-size`(기본 0 = 기존 동작)는 입력 캐시만 바꾸는 옵션이다.

## 3. 측정
- **정확도:** dev 전체 (G6 Zoo 전처리 캐시: 고화질 5종 4,923장 + CIFAR A 1,000장), G6b와 같은 eval 코드 `s2_g6b_ort.py` (ORT `ORT_ENABLE_ALL`, 정수 커널, HF 분류기). 쌍 비교 95% 구간(`boot_hr`, 층화 짝 부트스트랩): 각 후보 − 기준, 각 후보 − FP32. 출력 cos(기준 대비)도 적는다.
- **지연:** G6b `g6b_ort_speed.sh`를 그대로 복사한 `o1_ort_speed.sh` + `s2_g6b_bench.py --warmup 20 --n 200 --repeats 3`, 배치 1, 스레드 1·2·4 × 3세션, 10개 모델(두 모델 × FP32·기준·O-A·O-B·O-AB)을 같은 세션에서 번갈아, 세션별 평균의 중앙값.
  - 게이트: 측정 직전 WSL 안 다른 python·hailo·DFC 0개 + 1분 평균 부하 < 2, 아니면 5분마다 다시. Windows 쪽 부하는 확인 못 함.
- **최대 메모리:** `s2_model_stats.py` (4스레드, 모델마다 새 프로세스, 1회). **크기:** ONNX 파일 바이트.
- **프로파일:** `s2_profile_ort.py --threads 4 --zoo-size <224|256>` (워밍업 10 + 30장), 모델별 기준과 가장 빠른 후보(4스레드 중앙값).
- 모든 결과 행에 커밋 해시·dirty, CPU·VNNI·ORT 버전.

## 4. 판정 (결과 전 고정, 모델별로 따로)
- **채택 후보:** 아래를 모두 만족
  1. 스레드 1·2·4 중앙값이 **모두** 기준 대비 −3% 이상 (단축률 (기준 − 후보)/기준 ≥ 3%)
  2. (후보 − 기준) 고화질 Δtop-1이 "동률" 이상 = 95% 구간 상한 ≥ 0 (구간이 0을 포함하거나 전부 0 위)
  3. CIFAR A 차이(후보 − 기준) ≥ −1.0%p
- 여러 개면 4스레드 지연이 가장 짧은 것.
- 채택 후보가 있으면 **"예선 모델 교체 제안"**으로 적되, 교체는 팀이 정한다 (정민 노트북 공식 측정과 test 1회가 필요하다).

## 5. 구현 (커밋)
- `8fbd752` `s2_quantize.py --no-matmul-out-q` (O-A)
- `3972f43` `s2_ortfuse.py` (O-B)
- `4841e6e` `s2_profile_ort.py --zoo-size`, 스크립트 `o1_models.sh`(모델 목록)·`o1_eval.sh`·`o1_ort_speed.sh`·`o1_profile.sh`·`o1_run.sh`

## 6. 결과
(측정 뒤 채움)
