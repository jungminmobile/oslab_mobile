# G3b: 제자리 마스크(bcast) HAR로 16비트 층 줄이기 (2026-10-18, 연구실 데스크탑 WSL Ubuntu-22.04, DFC 5.4.0, CPU)

## 결론: **실패** (README 5절 G3b 기준 "D256L1 이상의 스크리닝 cos 0.952" 미달). 컴파일(HEF)은 하지 않았다.

| 구성 | `.alls` | optimize | 스크리닝 (dev 앞 200장, SDK_QUANTIZED) | 판정 |
|---|---|---|---|---|
| A 전부 a8_w8 | `siglip2/hailo/g3r_A256L1_all8.alls` | rc=0, 27분 | 5종 cos 0.14~0.27, top-1 0~1.5% (붕괴, 4종 뒤 중단) | 실패 |
| B D256L1 − sink 층 (`{ew_add*}` 16비트) | `g3r_B256L1_nosink16.alls` | **rc=1, 53초**: `Unsupported layers for the target mercury` precision_change 8개 (a16_w16_a8, signed) | — | 실패 |
| B2 D256L1 − sink 층 (잔차 ew_add 26개만 16비트) | `g3r_B2256L1_nosink16res.alls` | rc=0, 56분, 출력 SNR 3.5 dB | CIFAR A cos 0.730 / top-1 4.0, Pets 0.778 / 3.5 (2종 뒤 중단) | 실패 |

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
| (c) sink 행 1/√(분산+ε) ⊂ 일반 토큰 [min, max] | **15개 LN 중 12개 범위 밖**: 최대 11.72 (블록 6 norm1) vs 일반 최대 8.16. 범위 밖 배율 1.02~1.9배 | 15개 중 8개 범위 밖, norm1·final은 1000.0 | **실패** |

- 층별 값은 `../g3b_hc/hccheck.jsonl`. h_c로 바꿔 1/√ 값이 1000 → 최대 11.7로 줄었지만, 일반 토큰 범위 안에 들지 않았다.
- 해석 (**가설**): h_c는 많은 토큰의 평균이라 채널 사이 값 차이가 작아져 분산이 어느 개별 토큰보다도 작다 → 1/√분산이 일반 토큰보다 크다.
- 고정한 규칙(사전 확인 하나라도 실패하면 optimize 안 함, 이번이 마지막 시도)에 따라 여기서 끝낸다.

## G3R 최종 정리 (10/18, 용 결정: 마지막 시도 실패 시 멈춤)
- **ORT:** sink 제거 그래프(R1, 또는 같은 계산의 제자리 마스크 bcast)는 FP32와 동치이고, fc2 예외 없이 MatMul 79개 전부 INT8로 R0와 동률 정확도(−0.06 [−0.44, +0.33]), 크기 −12.8%, 4스레드 지연 −4.1%(비공식). 다만 G2 비열등 기준(하한 ≥ −0.3)을 못 넘어 **채택하지 않는다**.
- **Hailo:** 그래프는 파싱되지만(G3a 통과), sink 토큰을 0이나 평균으로 채우면 Hailo가 LayerNorm을 분해해 8비트로 양자화하는 1/√(분산+ε) 층에 새 이상치가 생겨 16비트 층을 줄인 구성이 무너진다 (cos 0.73~0.78 < 0.952). **Hailo 구성은 현재 D256L1 그대로 둔다.**
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
