# G3b: 제자리 마스크(bcast) HAR로 16비트 층 줄이기 (2026-10-18, 연구실 데스크탑 WSL Ubuntu-22.04, DFC 5.4.0, CPU)

> **사전 계획 밖 실행 (탐색적 결과).** 설계 4절은 G3b를 "G2·G3a 통과 시"에만 하도록 정했고, D-041도 처음에는 "파싱되면 멈춤"이었다. G2가 실패한 뒤 용의 승인으로 G3b를 4회 실행했다. 판정 기준(cos ≥ 0.952)은 바꾸지 않았다. 결론에 쓰기 전에 팀(쭈리·준혁) 확인이 필요하다 (siglip2/CLAUDE.md: 최종 결정권자).

## 결론: **실패** (README 5절 G3b 기준 "D256L1 이상의 스크리닝 cos 0.952" 미달). 컴파일(HEF)은 하지 않았다.

| 구성 | `.alls` | optimize | 스크리닝 (dev 앞 200장, SDK_QUANTIZED) | 판정 |
|---|---|---|---|---|
| A 전부 a8_w8 | `siglip2/hailo/g3r_A256L1_all8.alls` | rc=0, 27분 | 5종 cos 0.14~0.27, top-1 0~1.5% (붕괴, 4종 뒤 중단) | 실패 |
| B D256L1 − sink 층 (`{ew_add*}` 16비트) | `g3r_B256L1_nosink16.alls` | **rc=1, 53초**: `Unsupported layers for the target mercury` precision_change 8개 (a16_w16_a8, signed) | — | 실패 |
| B2 D256L1 − sink 층 (잔차 ew_add 26개만 16비트) | `g3r_B2256L1_nosink16res.alls` | rc=0, 56분, 출력 SNR 3.5 dB | **부분(2종)**: CIFAR A cos 0.730 / top-1 4.0, Pets 0.778 / 3.5 (2종 뒤 중단, jsonl 없음 — 고화질 5종 평균 아님) | 실패 |

- 공통: 최적화 수준 1, 보정 256장 시드 0(`calib_highres256_s0_nhwc.npy` `91f551e3…`), `layer_norm_decomposition`, equalization 끔 — D256L1과 같음. HAR `fbf722b5…`(`../g3a_maskbc`).
- 중단 이유: A·B2 모두 결과가 이미 분명했고, B2 스크리닝 중 WSL 메모리 available이 1 GB까지 떨어져(이 스크리닝 RSS 18 GB + 회전 실험 DFC 스크리닝 12 GB) 회전 실험 작업까지 죽을 위험이 있어 멈췄다. 부분 로그 `screen_*_partial.log`.
- B 실패 원인: `{ew_add*}`가 마스크 K·V 경로의 덧셈 24개까지 16비트로 만들어, 그 출력이 8비트 attention 행렬곱으로 가는 16→8비트 부호 있는 정밀도 변환이 생겼다 (hailo10h 미지원). B2는 원래 그래프의 ew_add 26개(위치 임베딩 1 + 잔차 24 + 풀링 1)에 해당하는 층만 16비트로 지정했다 (`../../../siglip2/hailo/layer_map_siglip2_rcmaskbc.csv`에서 ONNX 입력으로 구분).

## 진단 (붕괴 원인)
| 확인 | 결과 |
|---|---|
| 파싱된 HAR float 에뮬레이션 (SDK_NATIVE, Pets 16장) | cos 0.99993 (최소 0.99983) → 파서 변환은 맞다 |
| A optimize 후 float (SDK_FP_OPTIMIZED) | cos 0.99993 → 최적화 전처리(LN 분해 등)도 맞다 |
| A 양자화 (SDK_QUANTIZED) | cos 0.247 → 오차는 전부 양자화에서 |
| 참고: 원래 모델 DFC 기본 8비트 (팀 G1) | 고화질 cos 0.121 → 전부 8비트 붕괴는 원래 모델도 같음 |
| B2 층별 보정 범위 (`limvals_B2_top.txt`) | **`conv_var_inv_layer_normalization` 11·13·15·…·25의 범위가 0~1000 (11은 2000)** = 1/√ε. sink 토큰의 잔차를 0으로 만들어 그 토큰의 LayerNorm 분산이 0 → 분해된 1/√(분산+ε) 층에 거대 값이 생긴다 |

- 해석 (**가설**, 원래 D256L1 HAR과 직접 비교는 못 함 — 이 PC에 없음): 마스크 그래프는 sink의 거대 값을 없앤 대신 "0 벡터의 LayerNorm"이라는 **새 이상치**를 만들었다. ORT에서는 LayerNorm이 float라 문제가 없었고(INT8이 R1과 동률), Hailo는 LN을 분해해 8비트로 양자화하므로 그 층의 눈금이 망가진다.
- 대책 후보 (실행 안 함, 용 결정 필요): sink 토큰의 잔차를 0 대신 **보정 평균 같은 평범한 상수 벡터 h_c**로 바꿔 끼운다 (h' = h·(1−m) + m·h_c). sink의 K·V는 여전히 캐시로 바꿔 끼우므로 다른 토큰에는 영향이 없어 R1과 동치가 유지된다. 그 토큰의 MLP가 다시 거대 값을 쓰는지는 확인해야 한다.

## 실행
```bash
bash siglip2/hailo/g3r_g3b_wsl.sh A256L1_all8 B256L1_nosink16      # 커밋 f64cc86
bash siglip2/hailo/g3r_g3b_wsl.sh B2256L1_nosink16res              # 커밋 b1b3336
python siglip2/hailo/s2_dfc_native_check.py --har <har> --ctx native|fp_optimized|quantized --k 16
```
- 기록: `run_info.txt`, `optimize_*.log`, `har_sha256.txt`, `screen.jsonl`(A 4종 중단 전 기록 없음 — 프로세스 종료로 jsonl 미작성, 부분 로그만).

## 마지막 시도: sink 잔차를 h_c로 채우기 (D-041 연장, 용 승인 10/18 — G3b 마지막 시도, 실패하면 G3R은 "ORT 동률·Hailo 실패"로 정리)

### 고정한 정의와 기준 (실행 전)
- **h_c 정의:** 원래 FP32 모델에서 **보정 구간(train 고화질 256장, 시드 0, `calib_images("highres", 256, 0)`)**으로, 블록 l (5 ≤ l ≤ 11) 입력 잔차를
  **sink가 아닌 토큰**(패치 55·13을 뺀 194개) 전체와 사진 256장에 대해 평균한 768차원 벡터 h_c[l]. 마지막 LayerNorm(`trunk.norm`) 입력도 같은 방식으로 h_c[final].
  블록 l에서 활성인 sink 위치(블록 5~8: 55, 블록 9~11: 55·13)에 같은 h_c[l]을 넣는다 (평균에서 sink를 모두 빼므로 위치와 무관하게 같은 값). dev는 쓰지 않는다.
- **그래프:** 블록 l ≥ 5 입력에서 h' = h·(1−m) + m ⊙ h_c[l], 마지막 LN 전에 h' = h·(1−m) + m ⊙ h_c[final]. K·V는 지금처럼 bcast 캐시로 바꿔 끼운다 → 다른 토큰에는 영향이 없어 R1과 동치.
- **optimize 전 확인 (하나라도 실패하면 optimize 하지 않고 보고):**
  - (a) h_c 마스크 vs R1 FP32 임베딩 cos 최소 ≥ 0.99999 (dev 64장, Pets·Caltech 32장씩)
  - (b) 블록 5~11 어디서도 sink 위치가 거대 값을 다시 만들지 않음: sink 행의 잔차(블록 출력)·fc2 입력 max|·|가 일반 토큰 max|·|의 2배 이내 (dev 64장)
  - (c) FP32로 LayerNorm 분해 값 1/√(분산+ε)를 직접 계산해, sink 행 값이 일반 토큰의 [최솟값, 최댓값] 안 (블록 5~11의 norm1·norm2, 마지막 norm, dev 64장)
- **그다음:** B2 구성(잔차 ew_add 26개만 16비트)으로 optimize → 스크리닝(dev 6종 × 200장). 판정 README 5절 그대로 (고화질 cos ≥ 0.952). optimize 뒤 LN 분해 층 limvals를 기록한다.
- **메모리:** optimize와 에뮬레이터 스크리닝은 모두 `flock ~/siglip2_work/dfc.lock` 안에서 돈다 (회전 세션도 같은 잠금).

### 결과 (10/18, 커밋 `68b7af5`, dirty 아님, `../g3b_hc/`): **사전 확인 (c) 실패 → optimize 하지 않음. G3b 최종 실패**
- h_c: `regcache_hc_55at5-13at9_highres256_s0.npz` (256장, `hcache.jsonl`). 노름 b5 2.79 … b11 14.07, final 27.12.
- 기본값(`--hfill zero`)으로 다시 export한 bcast 파일 sha256이 이전과 같다 (`62754e37…`·`40a64c00…`). selftest: h_c 채움 vs mean 7.7e-7.

| 확인 (dev 64장) | h_c 채움 | (비교) 0 채움 | 판정 (h_c) |
|---|---|---|---|
| (a) vs R1 FP32 cos 최소 | 0.9999998 | 0.9999998 | 통과 |
| (b) sink 행 max / 일반 토큰 max (잔차·fc2 입력, 블록 5~11 최대) | 1.05 (`fc2in_10`) | 1.09 | 통과 |
| (c) sink 행 1/√(분산+ε) ⊂ 일반 토큰 [min, max] | **15개 LN 중 12개 범위 밖**: 최대 11.72 (블록 6 norm1) vs 일반 최대 8.16. 범위 밖 배율 1.02~1.9배 | 15개 중 9개 범위 밖 (norm1 7개·final은 1000.0, n2_8 1개) | **실패** |

- 층별 값은 `../g3b_hc/hccheck.jsonl`. h_c로 바꿔 1/√ 값이 1000 → 최대 11.7로 줄었지만, 일반 토큰 범위 안에 들지 않았다.
- 해석 (**가설**): h_c는 많은 토큰의 평균이라 채널 사이 값 차이가 작아져 분산이 어느 개별 토큰보다도 작다 → 1/√분산이 일반 토큰보다 크다.
- 고정한 규칙(사전 확인 하나라도 실패하면 optimize 안 함, 이번이 마지막 시도)에 따라 여기서 끝낸다.

## G3R 최종 정리 (10/18, 용 결정: 마지막 시도 실패 시 멈춤 — 4번째 시도 결과는 아래 추가)
- **ORT:** sink 제거 그래프(R1, 또는 같은 계산의 제자리 마스크 bcast)는 FP32와 동치이고, fc2 예외 없이 MatMul 79개 전부 INT8로 R0와 동률 정확도(−0.06 [−0.44, +0.33]), 크기 −12.8%, 4스레드 지연 −4.1%(비공식). 다만 G2 비열등 기준(하한 ≥ −0.3)을 못 넘어 **채택하지 않는다**.
- **Hailo:** 그래프는 파싱되지만(G3a 통과), sink 토큰을 0이나 평균으로 채우면 Hailo가 LayerNorm을 분해해 8비트로 양자화하는 1/√(분산+ε) 층에 새 이상치가 생겨 16비트 층을 줄인 구성이 무너진다 (cos 0.73~0.78 < 0.952). **G3R은 Hailo에 쓰지 않는다** (Hailo 최종은 D-042 rothad_D256L1, 10/18 쭈리 승인 — G3R과 별개).
- 보고서용 한 줄: "고정 위치 sink 토큰을 상수 K·V로 대체하면 ORT 정수 커널에서는 fc2 예외 없이 전부 8비트가 가능했지만(정확도 동률), NPU의 LayerNorm 분해 양자화와는 충돌했다."

## 4번째(마지막) 시도: 분산 맞춤 채움 벡터 h_c* (용 직접 요청, 10/18 — D-041)
### 고정한 정의와 기준 (실행 전)
- h_c*[l] = μ_l + (h_c[l] − μ_l) · √(v_target[l] / v(h_c[l])). μ_l·v(h_c[l])는 h_c[l](3번째 시도의 보정 평균 벡터, 768차원)의 채널 평균·채널 분산(편향 분산, LayerNorm과 같음).
- v_target[l] = 보정 구간(train 고화질 256장, 시드 0)에서 원래 FP32 모델의 그 LN 입력(블록 l 입력 = norm1 입력, 마지막은 `trunk.norm` 입력)의 **sink가 아닌 토큰**(194개 × 256장) 토큰별 채널 분산의 **중앙값**. dev는 쓰지 않는다.
  → var(h_c*[l]) = v_target[l]이 되어, sink 행의 1/√(분산+ε)가 일반 토큰의 중앙값 수준이 된다.
- 그래프: 3번째 시도와 같은 h + m ⊙ (h_c* − h). K·V는 bcast 캐시 그대로 → R1과 동치.
- 사전 확인 (a)(b)(c): 3번째 시도와 같은 기준. 셋 다 통과하면 B2 구성(잔차 ew_add 26개만 16비트) optimize → 스크리닝(dev 6종 × 200장), 판정 고화질 cos ≥ 0.952.
  optimize 뒤 LN 분해 층 limvals를 원래 모델 D256L1 HAR(`~/siglip2_work/hailo_g3q/opt_D_D256L1.har`, 이 PC에서 회전 세션이 팀 `.alls`로 만든 것, 이 PC 스크리닝 hr_cos 0.9666)과 나란히 비교한다.
- **(c)가 또 실패하거나 스크리닝이 기준 미달이면 G3R Hailo는 끝.** .alls에서 LN 분해 층을 16비트로 두는 방법은 하지 않는다.
- DFC(optimize·스크리닝)는 `flock ~/siglip2_work/dfc.lock` 안에서, 시작 전 `free -g` available ≥ 15 GB 확인.

### 결과 (10/18 16:18~17:22, `../g3b_hcv/`): **스크리닝 기준 미달 → G3b 최종 실패, G3R Hailo 종료**
| 단계 | 결과 |
|---|---|
| h_c* (`hcvcache.jsonl`) | var(h_c*) = v_target: 블록 5 0.0085 → 0.051 … 블록 11 0.258 → 1.018, final 0.958 → 3.154 |
| 사전 확인 (커밋 `2a00801`, `hccheck.jsonl`) | (a) 0.9999999 통과, (b) 최대 1.46배 통과, (c) sink 1/√ 최대 4.54 ⊂ 일반 [최소, 8.16], 15개 LN 모두 통과 |
| export·파싱 | ORT vs PyTorch 0.9999999, 354.53 MiB, DFC 파싱 rc=0, native cos 0.99993 |
| optimize (B2와 같은 규칙, `g3r_B2hcv256L1_nosink16res.alls`, flock 안) | rc=0, 50분, **출력 SNR 3.61 dB** (원래 D256L1 16.05 dB, 0 채움 B2 3.54 dB) |
| **스크리닝 (dev 6종 × 200장, 커밋 `64d7af3`, dirty 아님)** | **고화질 cos 0.7327, Δtop-1 −87.6%p** (CIFAR A 0.749·Pets 0.777·Flowers 0.802·Cars 0.663·Food 0.705·Caltech 0.716) → **0.952 미달, 실패** |

**LN 분해 층 limvals 비교** (`limvals_ln_compare.txt`, |limval| 최대: 원래 D256L1 / 0 채움 B2 / h_c* B2)
- `conv_var_inv`(1/√(분산+ε)): 블록 5 이후 LN(11·13·15·…·25)이 0 채움에서 1000(LN11 2000)이었고, **h_c*에서는 원래 D256L1과 같거나 작다** (LN11 7.24/2000/7.24, LN13 17.54/1000/8.80, LN25 1.18/1000/1.20). → 1000짜리는 사라졌다.
- `square1`: 원래 D256L1은 블록 6 이후 8만~91만(sink 거대 값의 제곱), h_c*는 5천~6만으로 오히려 작다.

**그런데도 무너진 이유 (가설, 확인 안 함 — 시도 횟수 끝)**
- 채움 덧셈 h + m⊙(h_c* − h) 8개(ew_add12·17·22·27·32·39·46·53, `fill_layers.txt`)가 **잔차 흐름 전체**를 지나는데 8비트다.
  범위가 같은 지점의 16비트 잔차 ew_add와 같다 (ew_add16 = ew_add17 = 66.14, ew_add52 = ew_add53 = 192.01).
  원래 D256L1에서 16비트로 유지되던 잔차 흐름이 블록 5부터 매 블록 8비트로 깎인다. 0 채움(h·(1−m) 곱, ew_mult)도 같은 구조였다.
- 즉 "sink 행을 바꿔 끼우는 연산이 잔차 흐름 위에 놓인다"는 그래프 구조 자체가 Hailo의 16비트 잔차 구성과 맞지 않는다. 채움 연산까지 16비트로 두는 것은 또 다른 시도라 하지 않았다 (D-041 마지막 시도 규칙).

## G3R 최종 정리 (갱신, 10/18 17:30)
- Hailo 시도 4회(0 채움 A·B·B2, h_c 채움은 사전 확인 실패, h_c* 채움) 모두 실패. **G3R Hailo 종료, G3R은 Hailo에 쓰지 않는다** (Hailo 최종 = D-042 rothad_D256L1). ORT는 G2 미채택(동률이나 비열등 기준 미달).
- 보고서용: "고정 위치 sink 토큰을 상수 K·V로 대체하면 ORT 정수 커널에서는 fc2 예외 없이 전부 8비트가 가능했다(정확도 동률). Hailo에서는 sink 행을 바꿔 끼우는 연산이 LayerNorm 분해(0 채움) 또는 16비트 잔차 흐름(평균 채움)과 충돌해 무너졌다."

## 5번째(진짜 마지막) 시도: 진단 먼저, 처방은 진단이 가리킬 때만 (용 직접 요청, 10/18 — D-041)
### 0단계 판정 규칙 (실행 전 고정)
- 검증할 가설(용): 0 채움(SNR 3.54 dB)과 h_c* 채움(3.61 dB)은 LN 분해 이상치가 전혀 다른데도(1000 vs 정상) 결과가 같다 → LN은 주원인이 아니다.
  두 시도의 공통점은 **잔차 흐름 위의 마스크 연산**(0 채움: h·(1−m) Mul, h_c*: h + m⊙(h_c* − h) Add)이 8비트로 남아 16비트 잔차를 블록마다 8비트로 깎은 것이다.
- **1단계 진단 결과, SNR을 가장 많이 깎는 층들이 "잔차 위의 마스크 연산"이면 → 2단계 처방(M2) 진행** (용에게 다시 묻지 않음).
- **다른 층(K·V 마스크 연산, 캐시 K·V와의 attention 행렬곱, LN 분해 등)이 주원인이면 → 처방하지 않고 원인만 기록, G3R Hailo 종료.**
- 2단계 판정: 스크리닝 고화질 cos ≥ 0.952 (기존 기준 그대로). 참고: 원래 모델 D256L1의 같은 PC 스크리닝 hr_cos 0.9666 (`~/siglip2_work/hailo_g3q/screen_D_D256L1.log`, 회전 세션).
- 1단계 진단 대상: h_c* B2 optimize HAR(`~/siglip2_work/g3b_20261018_161830/opt_B2hcv256L1_nosink16res.har`), 비교: 원래 D256L1 HAR(`~/siglip2_work/hailo_g3q/opt_D_D256L1.har`, 읽기만).
  ① 층별 정밀도(a8/a16) 표와 "잔차 경로 위 a8 층" 목록 ② `hailo analyze-noise`(보정 구간 사진, dev 아님) SNR 하락 상위 10 ③ 같은 역할 층끼리 limvals 비교(R1 쪽이 2배 이상 넓은 층).
### 2단계 처방 M2 (진단이 잔차 마스크를 가리킬 때만)
- 잔차 마스크를 모두 없애고, sink 시작 블록부터 매 블록 sink 행의 MLP 입력(norm2 출력, affine 포함)을 상수 u_c로 바꿔 끼운다: z' = z + m⊙(u_c − z).
  u_c[l] = 보정 구간(train 고화질 256장, 시드 0)에서 **원래 FP32 모델**의 블록 l norm2 출력을 sink가 아닌 토큰(194개 × 256장)에 대해 평균한 벡터. dev는 쓰지 않는다. K·V 캐시 교체와 풀링 헤드는 그대로.
- 사전 확인 (하나라도 실패하면 optimize 하지 않고 종료):
  (a) M2 vs R1 FP32 cos 최소 ≥ 0.99999 (dev 64장)
  (b) 블록 5~11 모든 블록에서 sink 행의 잔차 max ≤ 일반 토큰 max × 2
  (c) 새로 생긴 모든 연산(마스크 Sub·Mul·Add)의 입력·출력 범위(max, 99.9% 분위수)가 원래 모델의 같은 위치 텐서 범위의 2배 이내 + 잔차 경로 위에 새 연산이 없음(그래프로 확인)
  (d) LN 분해 1/√(분산+ε)의 sink 행 값이 일반 토큰 범위 안
- 그다음 parse → B2와 같은 규칙의 .alls(잔차 ew_add만 16비트, 이 HAR 층 이름으로 다시 매칭, 매칭 표 저장) → optimize → **출력 SNR이 10 dB 미만이면 스크리닝 생략하고 실패** → 스크리닝.
- 새 기능은 기본 꺼진 옵션(`--mask-site mlp`)으로 넣고, 기본 export sha256이 이전과 같은지 확인한다. DFC는 모두 flock 안, free ≥ 15 GB.

### 1단계 진단 결과 (10/18, `../diag5/`) → **판정: 주원인 = 잔차 위 마스크(채움) 연산 → 2단계 M2 진행** (0단계 규칙대로, 묻지 않음)
| 진단 | 결과 | 파일 |
|---|---|---|
| ① 층별 정밀도 | 잔차 경로 위 층 중 **입력 8비트 층**: 원래 D256L1 **0개** / 0 채움 22개(h·(1−m) Mul이 `a8_w8_a8` → 잔차가 통째로 8비트 + 8→16 변환) / h_c* **8개 = 채움 덧셈 전부**(`a8_w8_a16`) | `residual_path_*.csv`, `residual_a8_summary.txt` |
| ② 층별 잡음 | `hailo analyze-noise`는 원래 D256L1에서도 DFC 내부 오류(차원 768 vs 780)로 실패 → optimize가 HAR에 저장한 Layer Noise Analysis(누적 snr·층 자체 partial_snr)로 대체 | `layer_snr_*.csv`, `analyze_noise_*_fail.log` |
| ② 누적 SNR 추적 (h_c*) | 블록 4 출력 15.6 dB(원래 16.3) → 블록 5 attention 뒤 15.7 → **블록 5 MLP 뒤 11.4 (−4.4)** → **블록 6 채움 덧셈 5.95 (−5.4, 단일 최대)** → 블록 7 채움 −2.1 … 채움 덧셈 합계 **−9.5 dB** → 최종 3.6 (원래 16.05) | `snr_trace_block5.txt`, `snr_trace_fill.txt` |
| ② 부원인 | 블록 5 fc1(conv24)·fc2(conv25)가 B2에서 8비트: partial 6.1·10.8 dB (원래 D256L1은 16비트로 38·46 dB) → 블록 5 MLP에서 −4.4 dB. **M2로는 고쳐지지 않는다** (B2 규칙 고정) | 같음 |
| ② 비주원인 | `reduce_sum_matmul*`(K′의 헤드별 합, 행렬곱 zero point 보정용으로 DFC가 새로 만든 8비트 층): 텐서 partial 1.3~2.5 dB로 가장 낮지만, 블록 5 Q·K′ᵀ 누적 SNR 13.1 dB = 원래 13.3 dB → 출력에 번지지 않음. 원래 모델에서는 이 합을 qkv conv가 직접 계산(780채널) | `reduce_sum_compare.txt` |
| ③ limvals 같은 역할 비교 | 같은 이름 층 235개 중 h_c*가 원래보다 2배 이상 넓은 층 **0개** (38개는 2배 이상 좁음). K′ 범위 9.0 vs 원래 K 11.9 | `limvals_role_compare.txt` |
- 해석: 용의 가설(잔차 위 8비트 마스크 연산)이 맞다. 0 채움과 h_c*의 결과가 같았던 것은 둘 다 잔차를 8비트로 지나게 했고, 블록 5 MLP 8비트도 공통이기 때문이다. LN 분해는 주원인이 아니다.
