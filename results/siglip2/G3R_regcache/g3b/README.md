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
