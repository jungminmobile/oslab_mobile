# G8: 가중치 MatMul만 정적 INT8 (S-const) — M2rot_ort를 동적만큼 빠르게 (설계 2026-10-20)

> **용 요청. 결과를 보기 전에 고정한다.** 결과를 본 뒤에는 판정 기준을 바꾸지 않는다. **dev만 쓴다 (test 사용 금지).**
> 이 PC(연구실 데스크탑 WSL) 측정은 **비공식**이다.

## 1. 질문과 배경 (G7, `../G7_dynamic/README.md`)
- 정적 M2rot_ort(MatMul 79 + Softmax 13 정수)는 동적 D-m2rot(가중치 MatMul 53개만 정수)보다 지연이 9~19% 길다 (t1/t2/t4 99.8/55.9/31.9 vs 91.1/50.0/27.3 ms).
- 가설: 원인은 attention 행렬곱(QKᵀ·AV 26개)과 Softmax 정수화, 그리고 그 주변 Q/DQ 노드(Q 79·DQ 66) 비용이다.
- 질문: 가중치 MatMul만 정적으로 양자화하면 정확도는 지키면서 동적만큼 빨라지는가?

## 2. 구성
| 이름 | 만드는 법 |
|---|---|
| **S-all** (기준) | 기존 M2rot_ort 그대로 (`siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx`, sha256 `e84c3727…`). 다시 만들지 않는다 |
| **S-const** | M2rot FP32 (`siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx`, `16ef696e…`)에서 S-all과 같은 옵션(U8 + reduce_range, 채널별, percentile 99.999, highres 256장 시드 0, hc16, op MatMul, `quant_pre_process` 경로) + **`--const-b-only`** |
| 대조 (근거용) | 같은 옵션 + `--ort-const-b-only` (extra_options에 `MatMulConstBOnly=True`만) |
| FP32 | Release `siglip2_image_fp32.onnx` (`333cad08…`) |
| D-m2rot | G7 파일 (`siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_dynamic_ws8_pc_ops-MatMul_rr.onnx`, `afd20978…`) |

### `--const-b-only` 구현 (설계 변경, 용 승인 10/20)
- ORT 1.25에서 `MatMulConstBOnly`는 `onnx_quantizer.py`(QOperator 형식)와 `operators/matmul.py`의 `QOpMatMul.should_quantize`에서만 읽힌다. 우리가 쓰는 **QDQ 형식(`QDQQuantizer` → `QDQMatMul`)은 이 값을 읽지 않는다** → extra_options에 넣어도 S-all과 같은 모델이 나올 것으로 본다 (대조 파일로 확인).
- 그래서 `--const-b-only`는 **B(`input[1]`)가 initializer가 아닌 MatMul을 `nodes_to_exclude`에 자동으로 넣는다.** ORT `find_initializer_in_path`도 B가 바로 initializer인지만 보므로 같은 규칙이다. 판단은 `quant_pre_process` 뒤 그래프(`*_pre.onnx`)에서 한다.
- 안전장치 (코드 assert): ① 제외된 MatMul이 정확히 26개(QKᵀ 13 + AV 13 = 블록 12개 + MAP 헤드 1개) ② 제외된 MatMul의 B가 그래프 입력과 무관한 상수 경로(Transpose를 거친 가중치 등)면 잘못 빠진 것이므로 멈춤.
- **M2의 상수 K/V 캐시가 Concat으로 붙는 QKᵀ·AV는 B가 initializer가 아니라(캐시 상수 + 토큰 활성값의 Concat) 제외되는 것이 맞다.** B가 토큰 활성값에 의존하는 활성값×활성값 행렬곱이기 때문이다.
- AV가 빠지면 Softmax 출력에 Q/DQ가 붙지 않으므로 Softmax도 float으로 남는다.

### 코드 동일성
- 옵션을 끈 기본 경로로 M2rot_ort를 다시 만들면 기존 sha256 `e84c3727…`와 같아야 한다.

## 3. 먼저 확인할 것 (게이트, 실패하면 멈추고 보고)
1. S-const의 ORT 최적화 그래프에서 정수 MatMul이 가중치 MatMul 53개뿐이고 QKᵀ·AV·Softmax가 float인지 연산 개수로 확인. Q·DQ 노드 수를 S-all과 비교. 자동 제외된 MatMul이 26개인지 확인하고 노드 이름 목록을 이 README에 남긴다 (26이 아니면 코드가 멈춤 → 보고).
2. 가중치 MatMul 53개의 가중치 INT8 값·scale·zero point와 활성값(입력·출력) scale·zero point가 S-all과 **비트 단위로 같은지** 확인. 다르면 이유를 찾아 보고.

## 4. 판정 기준 (결과 전 고정)
- **① dev 전체에서 S-const − S-all 고화질 Δtop-1 95% 구간 하한 ≥ −0.5%p** (`s2_paired_ci.py`, 층화 짝 부트스트랩 시드 0, 1000회).
- **② 측정한 모든 스레드(1·2·4)에서 S-const가 S-all보다 3% 이상 빠름**: (S-all − S-const) / S-all ≥ 3%, 3세션 중앙값 기준.
- 둘 다 만족 → **"예선 교체 후보"로 팀에 제안** (결정은 팀). 하나라도 실패 → 기록만.
- 기록 (판정에 안 씀): 고화질 cos, 데이터셋별 top-1, CIFAR A Δ, peak 메모리, 크기, D-m2rot와의 지연 차이.

## 5. 측정
1. 정확도: dev 스크리닝 6×200 → dev 전체(고화질 4,923 + CIFAR A 1,000), 정수 커널 4스레드. S-all은 같은 실행에서 다시 돌리고 저장 예측(`../G7_dynamic/full/pred_…_rr_all.npz`)과 같은지 대조.
2. 지연: `s2_bench_speed.py --warmup 20 --n 200 --repeats 3`, 배치 1, 스레드 1·2·4 × 3세션, FP32·S-all·S-const·D-m2rot를 같은 세션에서 번갈아, 세션별 평균의 중앙값.
3. peak 메모리: `s2_model_stats.py` (모델마다 새 프로세스).
4. 시간 분해: `s2_profile_ort.py --threads 4`로 S-all·S-const·D-m2rot의 연산 종류별 시간(int_matmul, float_matmul, Q/DQ, Softmax, 기타). 프로파일 자체 오버헤드가 있어 비율로 비교한다.

## 6. 다른 세션과 겹침 (용 10/20)
- **양자화·정확도 평가:** 시작 전(단계마다) 다른 세션이 지연 측정 중인지 확인한다: `s2_bench_speed`·`s2_model_stats`·`s2_profile_ort`·`hailortcli` 프로세스가 하나라도 있으면 5분 간격으로 기다리고 기록한다 (`scripts/g8_measure_busy.sh`). 측정이 아닌 일(G6 사전 확인, 다운로드, 평가)이면 그대로 진행하되 `nice -n 10`, ORT 4스레드.
- **지연·메모리·프로파일:** 게이트(WSL 안 다른 python·hailo 0개 + 1분 평균 부하 < 2)를 통과할 때만, 5분 간격으로 대기·기록. Windows 쪽 부하는 확인하지 못한다. 측정 중에는 레포 파일을 고치지 않는다 (dirty 방지).
- 모든 결과 행에 커밋 해시·dirty, CPU·VNNI·ORT 버전. onnx는 커밋하지 않고 경로·sha256만.

## 7. Hailo 쪽 메모 (기록만, 이번 실험은 Hailo를 건드리지 않음)
- 팀 규칙상 ORT에서 FP32로 남긴 층은 Hailo에서 `a16_w16`으로 옮긴다. S-const가 채택되면 (가) attention 행렬곱을 Hailo에서도 16비트로 바꿀지(본선 지연이 늘 수 있음), (나) Hailo는 지금 구성(M2rot_D256L1)을 그대로 두고 "CPU 속도 때문에 ORT에서만 float"로 기록할지는 **팀이 정한다.**

## 8. 진행 기록
### 0. 코드 동일성 — 통과 (10/20 01:0x, `identity/quantize_log.jsonl`)
- 옵션을 끈 기본 경로로 M2rot_ort 재생성: 커밋 `449b1a1`(태그 `idcheck_g8`)과 이름 확인 순서를 고친 뒤 `d74c130`(태그 `idcheck_g8b`) 둘 다 sha256 **`e84c3727…`** = 기존 M2rot_ort. dirty 아님.
- 고친 것 (`d74c130`): 처음에는 파일 이름 중복 확인이 `_cbo` 표시를 붙이기 전에 실행돼, S-all 이름과 겹쳐 "이미 있음"으로 멈췄다 (덮어쓰지 않음). 두 옵션일 때만 표시를 붙인 뒤 확인하도록 바꿨다. 옵션을 끈 경로는 그대로다.

### 1. 양자화 (커밋 `d74c130` dirty 아님, `quant/quantize_log.jsonl`)
| 이름 | 파일 (`$SIGLIP2_ARTIFACTS/onnx/`) | sha256 | MiB |
|---|---|---|---|
| S-const | `siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_cbo_all.onnx` | `f1b64c42a291e0bcb98b741ba574996dbe0734b15391b9d509da584f7f3ede93` | 91.68 |
| 대조 (`--ort-const-b-only`) | `…_hc16_rr_ortcbo_all.onnx` | **`e84c3727…` = S-all과 같음** | 91.72 |
- **대조 결과: extra_options `MatMulConstBOnly=True`는 ORT 1.25 QDQ 형식에서 실제로 무시된다** (결과 파일이 S-all과 비트 단위로 같음, 최적화 그래프도 QLinearMatMul 79 + QLinearSoftmax 13).

### 2. 게이트 — 둘 다 통과 (커밋 `fa7642d` dirty 아님, `gate/gate.jsonl`, `gate/const_b_only_map.txt`)
**게이트 1: 연산 개수와 제외 목록**
- 자동 제외 **26개 = QKᵀ 13 + AV 13** (다음 연산이 Softmax면 QKᵀ, 아니면 AV). 블록 12개 + MAP 헤드 1개:
  - 블록 0~4 (원래 이름): `/trunk/blocks.{0..4}/attn/MatMul` (QKᵀ), `/trunk/blocks.{0..4}/attn/MatMul_1` (AV)
  - 블록 5~11 (M2가 다시 쓴 attention, 앞의 가중치 MatMul `/trunk/qkv/MatMul`·`/trunk/qkv_1/MatMul` … `/trunk/qkv_6/MatMul`): `/trunk/MatMul`·`_1`, `_2`·`_3`, `_4`·`_5`, `_6`·`_7`, `_8`·`_9`, `_10`·`_11`, `_12`·`_13` (짝수 QKᵀ, 홀수 AV)
  - MAP 헤드 (앞의 가중치 MatMul `/trunk/q/MatMul`·`/trunk/kv/MatMul`): `/trunk/MatMul_14` (QKᵀ), `/trunk/MatMul_15` (AV)
- 잘못 빠진 MatMul 없음: 제외된 26개의 B는 모두 토큰 활성값에 의존한다 (코드 assert 통과). B의 바로 앞 연산은 QKᵀ = Transpose(K), AV = Squeeze(블록 0~4) / Add(M2 블록·헤드: 상수 캐시와 합치는 경로)다. M2의 상수 K/V 캐시가 Concat으로 붙는 QKᵀ·AV는 B가 initializer가 아니므로 제외되는 것이 맞다.
- 가중치 MatMul 53개 중 B가 Transpose를 거쳐 들어가 initializer가 아닌 경우는 없다 (`_pre.onnx`에서 상수 가중치 MatMul 53개 모두 B가 바로 initializer).
- ORT 최적화 그래프 (`ORT_ENABLE_ALL`, 4스레드):

| | 정수 계산 | Q | DQ | float으로 남은 계산 |
|---|---|---|---|---|
| S-all | QLinearMatMul 79, QLinearSoftmax 13 | 79 | 66 | LayerNormalization 25 |
| **S-const** | **QLinearMatMul 53** | **53** | **66** | MatMul 13 + FusedMatMul 13 (QKᵀ·AV), **Softmax 13**, LayerNormalization 25 |
| D-m2rot (G7) | DynamicQuantizeMatMul 53 | 0 | 0 | MatMul 13 + FusedMatMul 13, Softmax 13, LayerNormalization 25 |

**게이트 2: 가중치 MatMul 53개 비트 비교** — 가중치 INT8 값·scale·zero point, 입력 활성값 scale·zero point, 출력 활성값 scale·zero point 모두 S-all과 **비트 단위로 같다** (`bit_identical_weight_matmul: true`). 바뀐 것은 attention 쪽뿐이다.

### 3. 정확도 (01:33~01:53, `scripts/g8_eval.sh`, 정수 커널 4스레드 `nice -n 10`, 커밋 `757fb2d` dirty 아님)
- 다른 세션 측정 확인 (`eval_wait.log`): 스크리닝·dev 전체 시작 전 모두 측정 프로세스 0개 → 대기 없음.
- 스크리닝 6×200 (`screen/screen.jsonl`): S-all 고화질 cos 0.9756 / Δtop-1 +0.3 / CIFAR A −2.5, **S-const 0.9785 / −0.3 / −4.0**, D-m2rot 0.9748 / −0.8 / −35.5.

| 구성 (dev 전체, `full/paired_ci.jsonl`) | CIFAR A | Pets | Flowers | Cars | Food | Caltech | 고화질 Δtop-1 (FP32 대비) [95%] | 고화질 cos | CIFAR A Δ |
|---|---|---|---|---|---|---|---|---|---|
| FP32 | 76.90 | 93.33 | 86.73 | 93.80 | 89.44 | 95.02 | 기준 (91.66%) | 1 | 기준 |
| S-all | 72.60 | 92.93 | 86.31 | 93.80 | 89.04 | 95.12 | −0.23 [−0.64, +0.18] | 0.9757 | −4.3 |
| **S-const** | 72.00 | 92.32 | 86.62 | 93.59 | 88.43 | 95.02 | **−0.47 [−0.88, −0.08]** | 0.9787 | −4.9 |
| D-m2rot | 49.70 | 93.13 | 87.15 | 93.80 | 89.14 | 92.97 | −0.43 [−0.88, +0.03] | 0.9802 | −27.2 |

| 짝 비교 (고화질, A → B는 B − A) | 차이 %p [95%] |
|---|---|
| **S-all → S-const** | **−0.24 [−0.58, +0.10]** |
| S-const → D-m2rot | +0.04 [−0.39, +0.51] (구분 안 됨) |
| S-all → D-m2rot | −0.20 [−0.65, +0.25] |
- S-all 예측은 G7 저장 예측(`../G7_dynamic/full/pred_…_rr_all.npz`)과 모든 배열이 비트 단위로 같다 (`full/s_all_pred_check.txt`).
- 기록: S-const는 cos가 S-all보다 높은데(0.9787 vs 0.9757) top-1은 낮다. 차이 −0.24는 구간이 0을 포함해 S-all과 구분되지 않지만, 하한 −0.58이 기준 −0.5보다 낮다.

### 4. 지연 (01:54~02:44, `scripts/g8_speed.sh`, 커밋 `b9e1ced` dirty 아님, `speed/runs.jsonl` 9줄, **비공식**)
- 게이트 (`speed/gate.log`): 대기 **5회** (01:54, 02:09, 02:24, 02:37 지연 / 02:44 메모리 — 모두 직전 측정이 올린 부하, 다른 프로세스 0). 모든 측정의 직전·직후 다른 프로세스 0. Windows 쪽 부하는 확인하지 못했다.

| 구성 | t1 ms | t2 ms | t4 ms | FP32 대비 (t1 / t2 / t4) | S-all 대비 (t1 / t2 / t4) |
|---|---|---|---|---|---|
| FP32 | 238.97 | 127.56 | 68.65 | 기준 | |
| S-all | 89.76 | 51.38 | 30.30 | −62.4 / −59.7 / −55.9% | 기준 |
| **S-const** | 89.92 | 51.09 | 30.23 | −62.4 / −59.9 / −56.0% | **+0.16 / −0.29 / −0.07 ms (+0.2 / −0.6 / −0.2%)** |
| D-m2rot | 83.77 | 46.52 | 26.17 | −64.9 / −63.5 / −61.9% | −5.99 / −4.86 / −4.13 ms (−6.7 / −9.5 / −13.6%) |
- 세션별 S-const의 S-all 대비 단축률: t1 −0.2·−0.1·−0.1%, t2 +0.6·+0.5·+0.3%, t4 +0.1·−0.1·+0.5% → 어느 세션에서도 3%에 못 미친다.
- S-const − D-m2rot: +6.15 / +4.57 / +4.06 ms (+7.3 / +9.8 / +15.5%).
- 참고: 같은 PC의 G7 측정(10/19 밤)보다 절대값이 약 5~10% 빠르다 (FP32 t1 238.97 vs 260.16). 실험끼리 절대값을 비교하지 않는다.

### 5. peak 메모리 (02:49, `s2_model_stats.py` 4스레드, 모델마다 새 프로세스, 1회, 커밋 `b9e1ced`, `model_stats/model_stats.jsonl`)
| 구성 | peak (추론 중) MiB | FP32 대비 | S-all 대비 |
|---|---|---|---|
| FP32 | 466.9 | 기준 | |
| S-all | 239.9 | −48.6% | 기준 |
| S-const | 219.5 | −53.0% | −20.4 (−8.5%) |
| D-m2rot | 184.8 | −60.4% | −55.1 (−23.0%) |
- FP32 절대값은 G7(525.4)·G3RQ(656.2)와 또 다르다. 1회 측정이라 실험 안에서만 비교한다.

### 6. 시간 분해 (02:49, `s2_profile_ort.py --threads 4`, 워밍업 10 + 30장(pets_dev 앞 40장), 커밋 `b9e1ced`, `profile/profile.jsonl`)
연산 커널 시간 ms/장 (프로파일러 오버헤드가 있어 절대값은 지연 측정과 다르다, 비교용):

| 구성 | 정수 MatMul | float MatMul | Q/DQ | Softmax | Add | Transpose | BiasGelu·Gelu | 나머지 | **커널 합** | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| S-all | 19.82 (79개) | – | 3.73 (Q 1.24 + DQ 2.49) | 2.17 (QLinearSoftmax) | 1.68 | 1.22 | 1.66 | 3.52 | **33.8** | 35.6 |
| S-const | 18.89 (53개) | 2.81 (FusedMatMul 1.45 + MatMul 1.36) | 3.48 (Q 0.88 + DQ 2.60) | 0.55 | 1.72 | 1.39 | 1.68 | 3.30 | **33.8** | 35.5 |
| D-m2rot | 18.98 (DynamicQuantizeMatMul 53개) | 2.75 | 0 | 0.52 | 0.73 | 0.87 | 1.68 | 2.95 | **28.5** | 29.7 |

## 9. 결과 요약과 판정 (4절 기준 그대로)
환경: Intel Core Ultra 9 285K (avx2 O, avx512f X, avx_vnni O), WSL Ubuntu-22.04, onnxruntime 1.25.0, 정수 커널. **비공식** (이 PC).

| 구성 | 크기 MiB | 지연 t1 / t2 / t4 ms (S-all 대비) | peak MiB | 고화질 Δtop-1 FP32 대비 [95%] | S-const − S-all [95%] | 고화질 cos | CIFAR A Δ |
|---|---|---|---|---|---|---|---|
| FP32 | 354.5 | 238.97 / 127.56 / 68.65 | 466.9 | 기준 (91.66%) | | 1 | 기준 (76.90) |
| S-all (M2rot_ort) | 91.7 (−74.1%) | 89.76 / 51.38 / 30.30 | 239.9 | −0.23 [−0.64, +0.18] | | 0.9757 | −4.3 |
| **S-const** | 91.7 (−74.1%) | 89.92 / 51.09 / 30.23 (+0.2 / −0.6 / −0.2%) | 219.5 | −0.47 [−0.88, −0.08] | **−0.24 [−0.58, +0.10]** | 0.9787 | −4.9 |
| D-m2rot | 91.7 (−74.1%) | 83.77 / 46.52 / 26.17 (−6.7 / −9.5 / −13.6%) | 184.8 | −0.43 [−0.88, +0.03] | | 0.9802 | −27.2 |

- **① S-const − S-all 하한 −0.58 < −0.5 → 실패.**
- **② S-all 대비 단축 t1 −0.2% / t2 +0.6% / t4 +0.2% (모두 3% 미만) → 실패.**
- **판정: ①② 모두 실패 → 기록만 (예선 교체 후보 아님).** M2rot_ort(S-all)를 그대로 둔다. Hailo는 건드리지 않았다 (7절 메모는 채택되지 않아 해당 없음).

### 왜 빨라지지 않았나 (6절 시간 분해)
1. **attention 정수화는 느린 원인이 아니었다 (가설 기각).** S-all → S-const에서 없어진 것(QKᵀ·AV 정수 26개 약 0.9 ms + QLinearSoftmax 2.17 ms + Q 26개 약 0.36 ms ≈ 3.5 ms)과 생긴 것(float FusedMatMul·MatMul 2.81 ms + Softmax 0.55 ms ≈ 3.4 ms)이 거의 같다. 토큰 196개라 attention 행렬곱이 작아서 정수든 float이든 비용이 비슷하다.
2. **동적과의 차이 5.3 ms는 가중치 MatMul 둘레에서 나온다.** 정수 행렬곱 시간은 S-const 18.89 vs D-m2rot 18.98 ms로 같다. 차이는 ① 독립 Q/DQ 노드 3.48 ms (Q 53 + DQ 66, 동적은 `DynamicQuantizeMatMul` 안에 합쳐져 0), ② Add 1.72 vs 0.73 ms (정적 QDQ 그래프는 QLinearMatMul 뒤 DQ → bias Add가 따로 남고, 동적은 bias가 커널에 합쳐짐), ③ Transpose 1.39 vs 0.87 ms. 합 약 5.1 ms로 차이 대부분을 설명한다.
3. 메모리는 S-const가 S-all보다 8.5% 작다 (attention 쪽 Q 텐서·QLinearSoftmax 작업 공간이 없어진 것으로 보임, 확인 안 함).

### 다음 후보 (확인 안 함, 팀·용이 정할 일)
- 가중치 MatMul의 **출력 양자화를 끄는** 구성 (ORT extra_options `OpTypesToExcludeOutputQuantization=["MatMul"]`): QLinearMatMul 대신 `MatMulIntegerToFloat`(DQ·bias 합침)로 바뀌면 동적처럼 독립 DQ·Add가 줄 수 있다. 입력 활성값은 그대로 정적 눈금이라 CIFAR A 정확도는 정적 쪽을 유지할 가능성이 있다. QDQ 형식에서 이 옵션이 실제로 읽히는지부터 이번처럼 대조 파일로 확인해야 한다.
- dev만 썼다 (test 사용 안 함).
