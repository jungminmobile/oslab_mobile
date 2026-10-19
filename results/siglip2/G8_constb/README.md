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
(결과 뒤에 채움)
