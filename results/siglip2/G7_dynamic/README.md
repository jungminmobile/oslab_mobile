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
(결과 뒤에 채움)
