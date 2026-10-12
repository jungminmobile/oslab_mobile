# G1 ② 손실 감 잡기: ORT INT8 전부 (MinMax), 정수 커널 (2026-10-13)

ROADMAP G1 마지막 항목: "ONNX Runtime INT8 전부(MinMax), 정수 커널, dev 1,000장".
**결론: SigLIP2는 텐서별 8비트 MinMax로 전부 양자화하면 완전히 무너진다 (고화질 평균 top-1 91.7 → 1.4, cos 0.445).**
원인은 블록 5부터 잔차 흐름·MLP fc2 입력에 생기는 활성값 이상치(최대값이 99.9% 분위수의 40~286배)로 보인다(아래 진단).
이 결과는 G3의 출발점이다. 보고서 수치가 아니라 방향 확인용이다.

## 실행 명령
```bash
cd siglip2
bash scripts/g1_pipeline.sh          # [4/5] 양자화, [5/5] 평가 (아래 표의 원본)
RN50_RESULTS=../results/siglip2/G1_int8_minmax_all/diag "$SIGLIP2_PYTHON" s2_diag_int8_collapse.py --n 200            # 1차 진단
RN50_RESULTS=../results/siglip2/G1_int8_minmax_all/diag "$SIGLIP2_PYTHON" s2_diag_int8_collapse.py --n 200 --round 2  # 2차 진단
```

## 조건
| 항목 | 값 |
|---|---|
| 모델 | `siglip2_image_int8_minmax_highres256_s0_s8_all.onnx` (89.6 MiB, FP32 354.5 MiB 대비 **−74.7%**), sha256 `e6587716…040bc121` |
| 양자화 | ORT 1.25.0 정적 PTQ, QDQ, 가중치 INT8 대칭 채널별, 활성값 INT8 비대칭 텐서별(S8S8, D-023), MinMax, ORT 기본 대상 연산 전부 |
| 보정 | 고화질 5종 train(dev 제외) 보정 후보 311장 중 256장, 시드 0. 보정기가 실제로 가져간 사진 256장 확인 |
| 평가 | dev 6종, **정수 커널**(`--exec int`, ORT_ENABLE_ALL, D-008), 4스레드, 이 PC(i7-1165G7, AVX512-VNNI) |
| 기록 | `runs.jsonl`·`quantize_log.jsonl` 두 번째 줄 (커밋 `ab52093`, dirty=false). `evaluate.jsonl` 마지막 12줄 |

## 결과 (dev, 정수 커널)
| dev | n | FP32 top-1 | INT8 top-1 | Δ (%p) | INT8 cos 평균 | cos 하위 1% |
|---|---|---|---|---|---|---|
| CIFAR A | 1000 | 76.90 | 1.60 | −75.30 | 0.4783 | 0.3681 |
| Pets | 990 | 93.33 | 2.32 | −91.01 | 0.4437 | 0.3633 |
| Flowers | 957 | 86.73 | 1.04 | −85.69 | 0.4546 | 0.3584 |
| Cars | 967 | 93.80 | 0.62 | −93.18 | 0.4228 | 0.3044 |
| Food | 985 | 89.44 | 1.42 | −88.02 | 0.4502 | 0.3513 |
| Caltech | 1024 | 95.02 | 1.46 | −93.56 | 0.4534 | 0.3411 |
| **고화질 5종 평균** | | **91.66** | **1.37** | **−90.29** | **0.445** | |

정수 커널 그래프 (`quantize_log.jsonl`의 `ort_optimized_graph`): 정수 계산 `QLinearMatMul` 66, `QLinearSoftmax` 13 / float으로 남은 계산 `MatMul` 13, `LayerNormalization` 26, `Erf` 13, patch `Conv` 1 / 변환 노드 Q 187·DQ 398.

## 원인 진단 (`diag/diag.jsonl`, CIFAR dev A 앞 200장, 커밋 `ab52093`)
가설은 실행 전에 정했다 (`s2_diag_int8_collapse.py` 머리말).

| 가설 | 확인 | 결과 | 판정 |
|---|---|---|---|
| H2 정수 커널 문제 | 같은 QDQ 모델을 fake-quant(디버깅 전용)로 실행 | 정수 cos 0.475 / fake cos 0.476, top-1 1.0 / 1.5 | **기각** — 양자화 오차 자체 |
| H3 연산 유형 | MatMul만 양자화 | top-1 13.0, cos 0.762 | MatMul만으로도 크게 손상 |
| | QDQRegistry 연산만(MatMul·LayerNorm 등, Add·Mul·Softmax 제외)* | top-1 1.0, cos 0.512 | LayerNorm 경계 양자화가 손실을 크게 키움 |
| | 전부 (기본) | top-1 1.0, cos 0.475 | |
| H1 활성값 이상치 | PyTorch hook, Pets dev 32장, 텐서별 max/p99.9 | 아래 표 | **강하게 지지** |

\* 1차 진단의 `V_no_add`와 `V_no_softmax`는 대상 연산 목록을 `QDQRegistry`와의 교집합으로 만드는 실수로, ORT 기본 목록에 있는 Add·Mul·Softmax 등이 둘 다 처음부터 빠져 **같은 모델**이 되었다. 위 표에서는 하나의 행으로 해석했다. 2차 진단에서 기본 목록 기준으로 다시 나눴다.

활성값 이상치 (max |x| / 99.9% 분위수):
| 블록 | 잔차 흐름(블록 출력) max | 비율 | fc2 입력(GELU 출력) max | 비율 |
|---|---|---|---|---|
| 0~4 | 8.6~12.5 | 3.7~6.2 | 5.4~18.3 | 3.5~6.0 |
| 5 | 249.5 | 65.6 | 156.6 | **124.0** |
| 6~8 | 255.9~256.7 | 55~68 | 6.5~29.9 | 3.3~9.0 |
| 9 | 609.7 | 50.3 | **1752.0** | **286.3** |
| 10~11 | 708~712 | 39~50 | 206~332 | 15~35 |

해석: 텐서별 8비트에서 범위가 ±712면 눈금 한 칸이 약 5.6이라, 대부분 값(99.9%가 4~18 이하)이 0~3칸으로 뭉개진다. ViT의 "massive activation"(잔차 흐름의 소수 채널·토큰이 매우 큰 값)과 같은 모양이다.
**RN50 트랙(앞쪽 stem·layer1이 민감)과 대조적으로, SigLIP2는 뒤쪽 블록(5~11)의 잔차 흐름·fc2 입력·LayerNorm 경계가 민감 위치 후보다.** 블록 단위·연산 유형 민감도는 G3에서 정식으로 잰다.

## 2차 진단 (위치 좁히기, `diag/diag.jsonl` `round: 2` 줄, 커밋 `c7bfd35`, CIFAR dev A 앞 200장, FP32 top-1 74.5)
| 변형 | 양자화 대상 | top-1 | cos 평균 | cos 하위 1% |
|---|---|---|---|---|
| 전부 (기본) | ORT 기본 목록 전부 | 1.0 | 0.475 | 0.387 |
| 기본 − LayerNorm | Add·Softmax·Mul·MatMul 등 (LayerNorm만 float) | 1.0 | 0.483 | 0.392 |
| MatMul + LayerNorm | | 1.0 | 0.510 | 0.384 |
| MatMul만 (1차) | | 13.0 | 0.762 | 0.639 |
| MatMul − 모든 fc2 | fc2 13개 float | **64.5** | **0.867** | 0.783 |
| MatMul − 블록 5~11 fc2 | fc2 7개 float | 57.5 | **0.870** | 0.787 |

해석:
1. **잔차 흐름을 8비트로 자르면 그것만으로 붕괴**한다: LayerNorm 입력(= 잔차 흐름) 양자화를 더하면 0.76 → 0.51, 잔차 합(Add) 양자화가 있으면 LayerNorm을 빼도 0.48.
2. **MatMul 손실은 블록 5~11의 fc2(입력 = 이상치가 큰 GELU 출력)에 몰려 있다**: 그 7개만 float으로 두면 0.76 → 0.87로, fc2 13개 전부를 뺀 것(0.867)과 같다.
3. 그래도 cos 0.87(top-1 −10~17%p, 200장)이라 나머지 MatMul·보정 방식도 개선이 필요하다 → G3 (보정 방식 비교, 블록·연산 유형 민감도).
200장 진단이라 top-1 차이(57.5 vs 64.5)는 표본 잡음 범위일 수 있다. 위치 판단은 cos로 한다.

## 3중 확인
1. 자기 검증: 보정 사진 256장 assert, `CalibMaxIntermediateOutputs` 미사용 assert. 같은 파이프라인 두 번 실행(`fe295d5` / `ab52093`) → dev 6종 12개 수치 모두 같음. 같은 조건으로 다시 양자화 → sha256 같음(`requant_sha256.txt`)
2. 독립 재계산: 요약 표를 `rn50/summarize.py`(다른 코드)로 raw `evaluate.jsonl`에서 다시 계산 → 고화질 Δtop-1 −90.29, cos 0.445 일치. 정수 커널 vs fake-quant 비교(H2)도 독립 경로 역할
3. 독립 검토: 코드 검토 서브에이전트 1회(지적 6건 반영, 커밋 `add8dc1`). 이 결과 폴더는 PR에서 다시 검토 예정

## 다음 (G3로 넘김)
- 보정 방식(Percentile·Entropy)은 이상치를 잘라 눈금을 촘촘히 하지만, massive activation은 의미 있는 값일 수 있어 잘라내면 손실 가능 → 비교 필요
- 혼합 정밀도: 이상치 텐서(블록 5~11 잔차 흐름, fc2 입력, LayerNorm 경계)를 FP32로 두는 구성. Hailo에서는 `a16_w16`에 해당 (공식 SigLIP `.alls`도 `ew_add*`(잔차 합)를 a16_w16으로 둠, D-018)
- patch `Conv`가 정수 커널로 융합되지 않은 이유 확인
