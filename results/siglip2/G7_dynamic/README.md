# G7: ORT 동적 양자화 기준선 vs M2rot_ort (설계 2026-10-19)

> **용 요청. 결과를 보기 전에 고정한다.** 기준선 측정이라 **채택 판정은 없다** (기록용). **dev만 쓴다 (test 사용 금지).**
> 이 PC(연구실 데스크탑 WSL) 측정은 **비공식**이다.

## 1. 배경
- 심사위원이 가장 먼저 떠올릴 기준선은 "ORT `quantize_dynamic` 한 줄"이다. SigLIP2에서 이것을 잰 기록이 없다 (4비트 MatMulNBits 실험의 동적 활성값만 있음).
- Hailo는 정적 양자화만 되므로 동적 양자화는 **예선 후보가 아니다.** 보고서의 비교 기준선으로만 쓴다.

## 2. 구성 (모두 onnxruntime 1.25.0 `quantize_dynamic`, `op_types_to_quantize=["MatMul"]`)
| 이름 | 입력 FP32 ONNX | 설정 | 만드는 명령 (`siglip2/`) |
|---|---|---|---|
| **D-default** | Release `siglip2_image_fp32.onnx` (`333cad08…`) | ORT 기본값: weight_type=QInt8, per_channel=False, reduce_range=False | `s2_quantize.py --dynamic --op-types MatMul --per-channel 0 --dump-ops` |
| **D-pc** | 같음 | per_channel=True, reduce_range=True (가중치 설정을 M2rot_ort와 맞춤) | `s2_quantize.py --dynamic --op-types MatMul --per-channel 1 --reduce-range --dump-ops` |
| **D-m2rot** | M2rot FP32 `siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx` (`16ef696e…`) | D-pc와 같음 | 위 + `--src siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx` |
| M2rot_ort (비교 대상) | M2rot FP32 | 정적 INT8 QDQ: MatMul 전부, 활성값 U8 텐서별(percentile 99.999, highres 256장 시드 0, hc16), 가중치 S8 채널별 + reduce_range | G3RQ-ORT (`../G3RQ_m2rot/ort/`), 91.7 MiB, sha256 `e84c3727…` |
| FP32 | Release `siglip2_image_fp32.onnx` (`333cad08…`) | | |

- 짝 비교로 나눠 보려는 것: **D-m2rot − D-pc = 그래프 변환(회전 + M2)의 효과**(동적 양자화 위에서), **M2rot_ort − D-m2rot = 같은 그래프에서 정적 보정 대 동적 활성값**.
- **전처리 차이 (해석할 때 주의):** 정적 경로(`M2rot_ort`, R0)는 `quant_pre_process`(shape 추론 + ORT 기본 그래프 최적화 + 기호 shape 정리)를 거친 `*_pre.onnx`를 양자화한다. 동적 경로(`--dynamic`)는 "한 줄" 기준선이므로 **FP32 ONNX 원본을 그대로** 넣고 `quant_pre_process`를 하지 않는다. 그래서 두 경로는 양자화 전 그래프 모양(예: 합쳐진 노드, MatMul 수)이 다를 수 있다. 실행 시 ORT가 같은 최적화(`ORT_ENABLE_ALL`)를 다시 하므로 최종 그래프는 비슷할 것으로 보지만, 아래 4절 연산 개수로 실제 차이를 확인해 적는다.
- 활성값 형식: 동적 양자화의 활성값은 ORT에서 항상 U8(실행 중 텐서별 min/max, `DynamicQuantizeLinear`)이다. 가중치는 세 D 모두 S8.

## 3. 구현과 코드 동일성
- `siglip2/s2_quantize.py`에 **기본값이 꺼진 `--dynamic`** 추가 (커밋 `4bed4b7`). 보정·제외 옵션과 함께 쓰면 assert로 막는다. `dump_ops`의 정수 커널 목록에 `DynamicQuantizeMatMul`를 추가했다 (정적 모델에는 이 연산이 없어 기존 기록은 같다).
- 옵션을 끈 기본 경로로 R0를 다시 만들면 이 PC의 기존 재생성 파일과 sha256이 같아야 한다 (`410d7483…`).

## 4. 잴 것 (판정 없음, 기록용)
1. **크기** (MiB, FP32 대비 %), sha256. 각 모델의 ORT 최적화 그래프(`ORT_ENABLE_ALL`) 연산 개수 — `MatMulInteger`·`DynamicQuantizeLinear`·`DynamicQuantizeMatMul`·`MatMulIntegerToFloat` 등 정수 커널이 실제로 쓰이는지, float `MatMul`이 몇 개 남는지.
2. **정확도:** dev 스크리닝 6×200(`s2_screen.py`) → dev 전체(고화질 4,923 + CIFAR A 1,000, `s2_paired_ci.py`, 정수 커널 4스레드).
   - 고화질 Δtop-1(FP32 대비)과 95% 구간, 고화질 cos, 데이터셋별 top-1, CIFAR A Δ.
   - 짝 비교 (층화 짝 부트스트랩 시드 0, 1000회): **M2rot_ort − D-default, M2rot_ort − D-pc, M2rot_ort − D-m2rot, D-m2rot − D-pc.**
   - M2rot_ort는 같은 실행에서 다시 돌리고, G3RQ-ORT의 저장 예측(`../G3RQ_m2rot/ort/full/pred_…_rr_all.npz`, 커밋 `bfa62ef`)과 같은지 대조해 적는다.
3. **지연:** `s2_bench_speed.py --warmup 20 --n 200 --repeats 3`, 배치 1, 스레드 1·2·4 × 3세션, **FP32·M2rot_ort·D-default·D-pc·D-m2rot 5개**를 같은 세션에서 번갈아, 세션별 평균의 중앙값. FP32 대비 감소율과 M2rot_ort 대비 차이.
4. **peak 메모리:** `s2_model_stats.py` (모델마다 새 프로세스).

## 5. 측정 조건 (다른 작업과 겹침)
- 이 PC에서 다른 세션의 DFC 컴파일(G5c)이 돌 수 있다. DFC는 건드리지 않는다 (`dfc.lock` 사용 안 함).
- 양자화·정확도 평가는 DFC와 동시에 해도 되며 `nice -n 10`, ORT 스레드 4개.
- **지연·메모리는 게이트를 통과할 때만 시작한다:** WSL 안 다른 python·hailo 프로세스 0개 + 1분 평균 부하 < 2. 세션·스레드마다 측정 직전 확인, 안 맞으면 5분마다 다시 확인. 대기 횟수와 시각을 `speed/gate.log`에 남긴다. 정확도 평가가 끝난 뒤에 시작한다 (내 평가 프로세스도 게이트에 걸리므로). **Windows 쪽 부하는 확인하지 못한다** (interop 꺼짐).
- 모든 결과 행에 커밋 해시·dirty, CPU·VNNI·ORT 버전을 남긴다. onnx는 커밋하지 않고 경로·sha256만 적는다.

## 6. 진행 기록
### 0. 코드 동일성 — 통과 (10/19 21:4x, 커밋 `4bed4b7` dirty 아님, `identity/quantize_log.jsonl`)
- `--dynamic`을 끈 기본 경로로 R0 재생성 (태그 `idcheck_g7`): sha256 **`410d7483…`** = 이 PC의 기존 재생성 파일(`…_idcheck_ort.onnx`)과 같다.

### 1. 양자화와 연산 개수 (커밋 `19751e1` dirty 아님, `quant/quantize_log.jsonl`, `quant/ops_reference.json`)
| 이름 | 파일 (`$SIGLIP2_ARTIFACTS/onnx/`) | sha256 | 크기 MiB (FP32 354.5 대비) |
|---|---|---|---|
| D-default | `siglip2_image_int8_dynamic_ws8_pt_ops-MatMul.onnx` | `dc2d8708d6a9b697821da7a86a455db1958c38c468e2a0d6170016b9ee5bacd0` | 91.33 (−74.2%) |
| D-pc | `siglip2_image_int8_dynamic_ws8_pc_ops-MatMul_rr.onnx` | `564c3be9fe007b81b58b99bc286204d00bea00a074e8e268019eefad2a9ffa7c` | 91.75 (−74.1%) |
| D-m2rot | `siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_dynamic_ws8_pc_ops-MatMul_rr.onnx` | `afd20978285df24834e62ea7a03c688a5cabc178e445f6ed41f5347f5ba7a8e6` | 91.68 (−74.1%) |
| M2rot_ort (참고) | `siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx` | `e84c3727…` (확인함) | 91.72 (−74.1%) |
- 입력 FP32 sha256 확인: `333cad08…`, M2rot FP32 `16ef696e…`.

ORT 최적화 그래프 (`ORT_ENABLE_ALL`, 4스레드):
| 이름 | 정수로 계산 | 변환 노드 | float으로 남은 계산 |
|---|---|---|---|
| D-default·D-pc·D-m2rot (셋 다 같음) | `DynamicQuantizeMatMul` **53** | (`DynamicQuantizeMatMul` 안에 합쳐짐) | `MatMul` 13 + `FusedMatMul` 13 (attention QKᵀ·AV), `Softmax` 13, `LayerNormalization` 25 |
| M2rot_ort | `QLinearMatMul` **79**, `QLinearSoftmax` 13 | `QuantizeLinear` 79, `DequantizeLinear` 66 | `LayerNormalization` 25 |
- 동적 양자화는 **가중치가 상수인 MatMul 53개만** 정수로 바꾼다 (ORT `quantize_dynamic`의 MatMul 처리는 두 번째 입력이 상수일 때만). attention의 활성값×활성값 MatMul 26개와 Softmax는 float이다. M2rot_ort는 79개 전부와 Softmax까지 정수다.
- 양자화 전 저장 그래프(`MatMulInteger` 53 + `DynamicQuantizeLinear` 53 + `Cast`·`Mul`)는 실행 시 ORT가 `DynamicQuantizeMatMul` 하나로 합친다. 정수 커널이 실제로 쓰인다.
- **전처리 차이 확인:** FP32 원본은 노드 791개, `quant_pre_process` 뒤(`_pre.onnx`) 510개 (M2rot: 947 → 603). MatMul 수는 79개(상수 가중치 53개)로 같다. 동적 모델의 최적화 그래프에는 원본에서 온 `Shape`·`Gather`·`Squeeze`·`Unsqueeze`·`Concat` 등 모양 계산 노드가 남아 있다 (D-default: Shape 14, Gather 28, Squeeze 38 등). 이 차이는 지연에 영향을 줄 수 있다 (7절에서 해석).

### 2. dev 스크리닝 6×200 (커밋 `3bbcdc3` dirty 아님, `screen/screen.jsonl`, 정수 커널 4스레드)
| 구성 | 고화질 cos | 고화질 Δtop-1 | CIFAR A Δ | CIFAR A cos |
|---|---|---|---|---|
| M2rot_ort | 0.9756 | +0.3 | −2.5 | 0.912 |
| D-default | 0.7606 | −41.3 | −49.0 | 0.801 |
| D-pc | 0.7574 | −40.9 | −50.0 | 0.813 |
| D-m2rot | 0.9748 | −0.8 | −35.5 | 0.855 |
- M2rot_ort 값은 G3RQ-ORT b단계 기록(0.9756, +0.3, −2.5)과 같다.

### 3. dev 전체 짝 비교 (22:00~22:20, 커밋 `6a6038f` dirty 아님, `full/paired_ci.jsonl`, 정수 커널 ORT 1.25.0 4스레드, `nice -n 10`, DFC 컴파일과 동시)
| 구성 | CIFAR A | Pets | Flowers | Cars | Food | Caltech | 고화질 Δtop-1 (FP32 대비) [95%] | 고화질 cos | CIFAR A Δ |
|---|---|---|---|---|---|---|---|---|---|
| FP32 (`333cad08…`) | 76.90 | 93.33 | 86.73 | 93.80 | 89.44 | 95.02 | 기준 (91.66%) | 1 | 기준 |
| **M2rot_ort** | 72.60 | 92.93 | 86.31 | 93.80 | 89.04 | 95.12 | **−0.23 [−0.64, +0.18]** | 0.9757 | −4.3 |
| D-default | 22.60 | 69.49 | 34.27 | 50.47 | 40.91 | 67.19 | **−39.20 [−40.55, −37.86]** | 0.7634 | −54.3 |
| D-pc | 22.00 | 71.11 | 35.53 | 50.67 | 38.78 | 66.70 | **−39.11 [−40.46, −37.79]** | 0.7612 | −54.9 |
| D-m2rot | 49.70 | 93.13 | 87.15 | 93.80 | 89.14 | 92.97 | **−0.43 [−0.88, +0.03]** | 0.9802 | −27.2 |

짝 비교 (고화질, B − A, 층화 짝 부트스트랩 시드 0·1000회):
| A → B | 차이 %p [95%] |
|---|---|
| D-default → M2rot_ort | **+38.97 [+37.63, +40.38]** |
| D-pc → M2rot_ort | **+38.88 [+37.50, +40.22]** |
| D-m2rot → M2rot_ort | **+0.20 [−0.25, +0.65]** (구분 안 됨) |
| D-pc → D-m2rot (그래프 변환 효과) | **+38.68 [+37.31, +40.09]** |
| D-default → D-pc (가중치 채널별+rr) | +0.09 [−0.82, +0.95] (구분 안 됨) |
- `paired_ci.jsonl`의 `pair: [a, b]`는 `hr_diff_b_minus_a`(b − a)로 저장된다. M2rot_ort가 a인 행은 부호를 뒤집어 "M2rot_ort − D"로 적었다.
- **M2rot_ort 예측 대조:** 이번 실행의 M2rot_ort 예측이 G3RQ-ORT 저장 예측(`bfa62ef`)과 6종 모두 예측·정답 여부가 같고 cos 차이 0 (`full/m2rot_ort_pred_check.txt`). 같은 사진·같은 스크립트로 재현됨.
- 참고: 한 실행 안에서 스크린 단계는 `3bbcdc3`, dev 전체 단계는 `6a6038f`로 기록됐다. 그 사이 커밋은 지연 스크립트 추가뿐이라 평가 코드는 같다.

### 4. 지연 (10/19 22:20 ~ 10/20 00:01, `scripts/g7_speed.sh`, 커밋 `64fa3f1` dirty 아님, `speed/runs.jsonl` 9줄, **비공식**)
- 방법: `s2_bench_speed.py --warmup 20 --n 200 --repeats 3`, 배치 1, 다섯 모델을 같은 세션에서 번갈아, 스레드 1·2·4 × 3세션, 세션별 평균(ms/장)의 중앙값.
- 게이트 (`speed/gate.log`): 측정 직전 WSL 안 다른 python·hailo 0개 + 1분 평균 부하 < 2, 안 맞으면 5분마다 다시 확인. **대기 13회** (22:20 1회, 22:37 1회, 22:49~23:32 9회, 23:39 1회, 23:56 1회). DFC 컴파일(G5c)은 22:20 이전에 끝났고, 이후 대기는 주로 다른 세션의 G6 작업(`s2_g6pre.py`)과 측정 직후 부하 때문이었다. **Windows 쪽 부하는 확인하지 못했다** (interop 꺼짐).
- **측정 도중 끼어든 작업 (기록):** ① s=2 t=1 (22:44~22:49): 다른 세션의 Hugging Face 다운로드(`snapshot_download`, 22:48:41 시작)가 마지막 약 30초와 겹침. ② s=2 t=2 (22:54~22:57): 다른 세션의 `s2_g6pre.py ident`(22:57:45 시작)가 마지막 약 6초와 겹침. 두 측정값은 세션 1보다 0.6~2.3% 느린 정도라 그대로 썼다 (다섯 모델이 같은 방향이라 모델 사이 비교에는 영향이 작다).
- 세션 3은 다섯 모델 모두 세션 1보다 약 4~10% 빨랐다 (같은 방향이라 기계 상태 차이로 보임). 모델끼리 비교는 같은 세션 안에서 번갈아 쟀으므로 유지된다.

| 구성 | t1 ms | t2 ms | t4 ms | FP32 대비 (t1 / t2 / t4) | M2rot_ort 대비 (t1 / t2 / t4) |
|---|---|---|---|---|---|
| FP32 | 260.16 | 136.18 | 72.53 | 기준 | |
| **M2rot_ort** | 99.81 | 55.85 | 31.89 | −61.6 / −59.0 / −56.0% | 기준 |
| D-default | 87.65 | 47.89 | 25.88 | −66.3 / −64.8 / −64.3% | −12.16 / −7.96 / −6.01 ms (−12.2 / −14.3 / −18.8%) |
| D-pc | 89.23 | 48.93 | 26.38 | −65.7 / −64.1 / −63.6% | −10.58 / −6.92 / −5.51 ms (−10.6 / −12.4 / −17.3%) |
| D-m2rot | 91.14 | 49.97 | 27.27 | −65.0 / −63.3 / −62.4% | −8.67 / −5.88 / −4.62 ms (−8.7 / −10.5 / −14.5%) |
- 세션별 값: `speed/speed_*.json`. 참고: 같은 PC의 G3RQ-ORT 측정(10/19 오전)보다 FP32·M2rot_ort가 모두 약 8% 느리다 (FP32 t4 72.53 vs 67.04, M2rot_ort 31.89 vs 29.51). 실험끼리 절대값을 비교하지 않는다.

### 5. peak 메모리 (10/20 00:01, `s2_model_stats.py` 4스레드, 모델마다 새 프로세스, 1회, 커밋 `64fa3f1` dirty 아님, `model_stats/model_stats.jsonl`)
| 구성 | peak (추론 중) MiB | FP32 대비 | M2rot_ort 대비 |
|---|---|---|---|
| FP32 | 525.4 | 기준 | |
| **M2rot_ort** | 243.6 | −53.6% | 기준 |
| D-default | 186.8 | −64.4% | −56.8 (−23.3%) |
| D-pc | 185.3 | −64.7% | −58.3 (−23.9%) |
| D-m2rot | 187.2 | −64.4% | −56.4 (−23.2%) |
- 참고: G3RQ-ORT의 같은 측정(FP32 656.2, M2rot_ort 232.4)과 절대값이 다르다 (특히 FP32). 1회 측정이라 실행마다 흔들리는 것으로 보고, 이 실험 안에서만 비교한다.

## 7. 결과 요약 (판정 없음, 기록용)
환경: Intel Core Ultra 9 285K (avx2 O, avx512f X, avx_vnni O), WSL Ubuntu-22.04, onnxruntime 1.25.0, 정수 커널. **비공식** (이 PC).

| 구성 | 크기 MiB (FP32 대비) | 지연 t1 / t2 / t4 ms (FP32 대비) | peak MiB (FP32 대비) | 고화질 Δtop-1 FP32 대비 [95%] | M2rot_ort − 이 구성 [95%] | 고화질 cos | CIFAR A Δ |
|---|---|---|---|---|---|---|---|
| FP32 | 354.5 | 260.16 / 136.18 / 72.53 | 525.4 | 기준 (91.66%) | | 1 | 기준 (76.90) |
| **M2rot_ort** (정적) | 91.7 (−74.1%) | 99.81 / 55.85 / 31.89 (−61.6 / −59.0 / −56.0%) | 243.6 (−53.6%) | **−0.23 [−0.64, +0.18]** | | 0.9757 | −4.3 |
| D-default | 91.3 (−74.2%) | 87.65 / 47.89 / 25.88 (−66.3 / −64.8 / −64.3%) | 186.8 (−64.4%) | **−39.20 [−40.55, −37.86]** | **+38.97 [+37.63, +40.38]** | 0.7634 | −54.3 |
| D-pc | 91.8 (−74.1%) | 89.23 / 48.93 / 26.38 (−65.7 / −64.1 / −63.6%) | 185.3 (−64.7%) | **−39.11 [−40.46, −37.79]** | **+38.88 [+37.50, +40.22]** | 0.7612 | −54.9 |
| D-m2rot | 91.7 (−74.1%) | 91.14 / 49.97 / 27.27 (−65.0 / −63.3 / −62.4%) | 187.2 (−64.4%) | **−0.43 [−0.88, +0.03]** | +0.20 [−0.25, +0.65] (구분 안 됨) | 0.9802 | −27.2 |

읽는 법:
1. **"한 줄" 동적 양자화(D-default)는 SigLIP2에서 무너진다:** 고화질 −39.2%p, CIFAR A −54.3. 가중치를 채널별 + reduce_range로 바꿔도(D-pc) 그대로다 (D-pc − D-default +0.09 [−0.82, +0.95]). → 원인은 가중치 쪽이 아니라 활성값 쪽이다.
2. **같은 동적 양자화도 M2rot 그래프 위에서는 고화질이 거의 회복된다:** D-m2rot − D-pc = **+38.68 [+37.31, +40.09]%p**. 고화질에서 M2rot_ort와 구분되지 않고(+0.20 [−0.25, +0.65]), cos는 오히려 더 높다(0.9802 vs 0.9757).
3. **그러나 CIFAR A는 D-m2rot도 −27.2**로 크게 떨어진다 (M2rot_ort −4.3). 정적 보정(M2rot_ort)이 동적보다 확실히 나은 곳은 CIFAR A다. 그 대가로 M2rot_ort는 동적보다 지연이 8.7~18.8% 길고 peak 메모리가 약 23% 크다 (크기는 같음).
4. 크기는 네 INT8 모두 91.3~91.8 MiB로 같다 (가중치 MatMul이 모두 INT8이기 때문).

원인 가설 (확인 안 함, 다음 실험 후보):
- (1·2의 원인) **동적 양자화는 활성값 눈금을 사진마다 텐서 전체의 min/max로 정한다.** SigLIP2는 일부 토큰(sink 토큰)·채널에 거대 활성값이 있어(G3R·G3Q 진단), 그 값 하나가 눈금을 정하면 나머지 토큰이 몇 단계로 뭉개진다. 정적 경로는 percentile 99.999로 이 값을 잘라내고(R0는 fc2 블록 5·9를 FP32로 남김), M2(sink 토큰 레지스터 캐시) + 회전은 거대 값 자체를 없애거나 퍼뜨린다 → D-m2rot의 회복과 맞는다. 확인 방법: D-default의 MatMul별 입력 max/중앙값 비, 또는 블록 5·9 fc2만 빼고 동적 양자화.
- (3의 원인) CIFAR A(32px를 키운 사진)에서는 M2가 다루지 않는 위치·블록에 큰 활성값이 생길 수 있다 (M2 캐시는 고화질 보정 평균). D-m2rot의 CIFAR A cos 0.855 vs 고화질 0.98. 정적 경로는 percentile로 잘라 덜 민감하다. 확인 안 함.
- (지연) 동적 모델은 attention QKᵀ·AV 26개와 Softmax를 float으로 계산하는데도 M2rot_ort보다 빠르다. 가설: M2rot_ort의 QuantizeLinear 79·DequantizeLinear 66 변환 노드 비용이 활성값×활성값 정수화 이득(토큰 196개라 작음)보다 크다. 또는 `DynamicQuantizeMatMul` 커널이 `QLinearMatMul`보다 빠르다. 그래프 차이(M2·회전)는 D-m2rot도 빠르므로 주원인이 아니다. 확인 안 함 (`s2_profile_ort.py`로 확인 가능).
- 전처리 차이(2절): 동적 모델 그래프에 모양 계산 노드가 더 남아 있지만, 그래도 더 빠르므로 지연 결론을 뒤집는 요인은 아니다.

보고서용 한 줄: "ORT `quantize_dynamic` 한 줄은 SigLIP2에서 고화질 top-1을 −39.2%p 잃는다. 우리 정적 INT8(M2rot_ort)은 같은 크기에서 −0.23%p이며, 그래프 변환(M2 + 회전)이 손실 대부분(+38.7%p)을 막고, 정적 보정이 저해상도(CIFAR A)에서 추가로 +23%p를 지킨다. 대신 동적보다 CPU 지연이 9~19% 길다."
- dev만 썼다 (test 사용 안 함). Hailo는 정적 양자화만 되므로 동적 구성은 예선 후보가 아니다.
