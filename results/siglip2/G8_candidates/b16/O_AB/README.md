# O-AB 전체 측정 (2026-10-22, 쭈리 PC i7-1165G7, 비공식)

- 대상: 준용 O1 결과 Release `siglip2-oab-20261022` (pre-release). **다시 양자화하지 않고 파일 그대로** 씀.
  - B16 O-AB `siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx` sha256 `db2c1fdd…a77e`
  - B32 O-AB `siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx` sha256 `7a01044c…dbf6`
  - O-A: 가중치 MatMul 출력을 8비트로 다시 반올림하는 단계를 끔 (DQ 66 → 13). O-B: Add+LayerNorm → SkipLayerNormalization(com.microsoft) 24개.
- 비교: 현재 예선 모델 M2rot_ort (`e84c3727…`), B32는 우리 B32 M2rot(통일 규칙 `15@6:5, 7@10:9`).
- 코드 커밋 `4b23052`, 모든 단계 시작·끝 dirty 0. 정수 커널, ORT 1.25.0, dev만 사용 (**test 안 씀**).
- 스크립트: scratchpad `run_oab_full.sh`, `run_rerun24.sh` (명령은 아래 각 절). 공용 잠금 안, 속도 단계는 시작 전 확인(CPU 30초 < 25%, B/16 FP32 4T 중앙값 < 200 ms).

## 1. 정확도 — dev 전체 (`dev_full/paired_ci.jsonl`, `../../b32/O_AB/dev_full/paired_ci.jsonl`)

```bash
RN50_RESULTS=…/b16/O_AB/dev_full python s2_paired_ci.py --models <B16 O-AB> <M2rot_ort>
S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256 SIGLIP2_ARTIFACTS=…/siglip2_b32 RN50_RESULTS=…/b32/O_AB/dev_full python s2_paired_ci.py --models <B32 O-AB> <B32 M2rot>
```

| 모델 | 고화질 5종 Δtop-1 [95%] | cos | CIFAR A Δ | Pets / Flowers / Cars / Food / Caltech Δ |
|---|---|---|---|---|
| B16 M2rot_ort | −0.23 [−0.64, +0.18] | 0.9757 | −4.3 | −0.40 / −0.42 / 0.00 / −0.40 / +0.10 |
| **B16 O-AB** | **−0.18 [−0.55, +0.18]** | **0.9869** | **−2.3** | 0.00 / −0.21 / +0.20 / −0.81 / −0.10 |
| B16 O-AB − M2rot (짝) | **+0.04 [−0.26, +0.35] 동률** | | | |
| B32 M2rot | −0.79 [−1.26, −0.34] | 0.9773 | −0.2 | |
| B32 O-AB | −0.27 [−0.62, +0.09] | 0.9893 | −0.5 | |
| B32 O-AB − M2rot (짝) | +0.52 [+0.11, +0.95] 유의 | | | |

- B16 M2rot_ort 값은 10/19 dev 전체(`G6_m2rot_ort/dev_full`)와 소수점까지 같음 (재현).
- **B32 주의**: B32 O-AB는 준용 `g6b_b32_m2rot` FP32 그래프에서 만든 것이라 우리 B32 M2rot과 바탕 그래프가 다를 수 있다. +0.52는 O-AB 효과와 바탕 차이가 섞였을 수 있음 → 준용에게 바탕 확인 요청.

## 2. 지연 — 8모델 같은 실행 번갈아 (`../../latency_this_pc/all8_20261022*`)

- 1차 `all8_20261022` (10:40~11:54, 시작 확인 cpu 20%·probe 195.7 ms): 1세션 2·4스레드가 다른 세션보다 FP32 기준 17~70% 느림.
- 재측정 `all8_20261022_rerun_t24` (12:31~13:11, 2·4스레드 3세션). **실행 사고**: 같은 스크립트가 두 번 실행되어 두 번째가 13:11에 같은 폴더로 다시 측정하며 `cross_latency.json`을 덮어씀 → 13:19 중단, 덮인 파일은 `cross_latency_DUPLICATE_partial_1312.json`. 첫 실행 값은 `latency_run.log`의 요약·세션 줄에서 `cross_latency_recovered.json`으로 복원 (세션 줄에서 다시 낸 중앙값이 로그 요약과 정확히 같음). 세션 안 반복별 값은 잃음.
- **이상치 규칙(측정 중 정함)**: 같은 스레드의 모든 세션 중앙값보다 B/16 FP32가 10% 넘게 느린 세션은 뺀다 → 1차 1세션 2T·4T, 재측정 3세션 4T 제외. 1T는 1차 3세션, 2T는 5세션, 4T는 4세션.

| ms (중앙값) | 1T | 2T | 4T | FP32 대비 1/2/4T |
|---|---|---|---|---|
| B16 FP32 | 383.6 | 216.7 | 165.8 | |
| B16 M2rot_ort | 154.4 | 104.5 | 79.2 | −59.8 / −53.1 / −51.5% |
| **B16 O-AB** | **145.3** | **88.4** | **68.0** | **−62.1 / −60.5 / −57.9%** |
| B32 FP32 | 132.8 | 74.0 | 61.5 | |
| B32 M2rot | 51.4 | 36.4 | 31.4 | −61.3 / −53.2 / −49.4% |
| B32 O-AB | 47.9 | 31.3 | 27.0 | −63.8 / −57.8 / −56.9% |
| RN50 FP32 | 107.3 | 63.0 | 47.5 | |
| RN50 B2 | 46.1 | 28.4 | 20.9 | −58.0 / −56.3 / −56.0% |

(비율은 세션마다 같은 실행 안 비율을 낸 뒤 중앙값)

- **B16 O-AB vs M2rot_ort**: 1T −9.2% (세션 −9.8/−2.0/−9.2), 2T −15.6% (−15.4/−15.8/−16.1/−15.6/−14.7), 4T −13.8% (−13.7/−14.0/−10.6/−14.6).
- B32 O-AB vs M2rot: 1T −5.2%, 2T −11.1% (한 세션 −1.9), 4T −14.3%.

## 3. 메모리 (1차 `--mem`, 모델마다 새 프로세스, 4스레드, 1회)

| MiB | peak_wset | 추론 중 샘플 최대 |
|---|---|---|
| B16 FP32 / M2rot / O-AB | 428.9 / 204.9 / **202.1** | 같음 |
| B32 FP32 / M2rot / O-AB | 453.7 / 195.3 / 193.0 | 428.5 / 182.1 / 178.7 |
| RN50 FP32 / B2 | 316.2 / 113.0 | 220.4 / 113.0 |

- E12-1(정민 코드, 3회 중앙값): M2rot C0 227.2 → C3 206.0 (−9.3%), O-AB C0 224.8 → C3 205.6 (−8.5%). 측정 정의가 달라 절대값은 위 표와 다르다.

## 4. 연산별 시간 (`profile/`, `profile_n200/`)

- 30장 1회(`profile/`)에서는 O-AB가 빠르지 않게 나옴 (1T 162.7 vs 153.4 ms). **200장 × 3회(`profile_n200/`)에서는 6번 모두 O-AB가 빠름**: 1T −3.2 / −10.8 / −6.7%, 4T −11.1 / −9.0 / −21.1% → 30장 결과는 표본 흔들림.
- 줄어든 곳 (1T, 3회 범위): QLinearSoftmax 16.4~18.0 ms → float Softmax 3.3~3.8 ms (O-A로 attention 행렬곱 출력이 float이 됨), qdq 10.5~11.3 → 6.0~7.1 ms, LayerNorm+Add(약 14~15) → SkipLayerNorm+Add(약 12~15).
- 늘어난 곳: 정수 행렬곱 1T +3~10 ms, 4T 0~+4 ms (출력 형식이 바뀐 커널).

## 5. 모델 통계 (`model_stats/`)
- 파라미터 92,884,224 (FP32와 같음), GMACs 17.73 (같음). 파일 91.65 MiB (M2rot 91.72).

## 6. 남은 것 (제출 모델로 바꿀 때)
- test 1회 (프로토콜 예외, 쭈리 승인 필요), 다른 PC(VNNI 없는 x86, ARM) 비트 일치, com.microsoft 연산 → 채점 PC가 onnxruntime인지 학술부 확인, 보고서 2장·9장·Release 갱신, B32 바탕 그래프 확인.
