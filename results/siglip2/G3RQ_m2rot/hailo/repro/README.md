# 빌드 재현성: rothad_D256L1·M2rot_D256L1 2차 빌드로 D-043 판정 (설계 2026-10-19 — 판정 기준은 팀(준혁)이 결과 전에 정함)

## 1. 2차 빌드
- rothad_D256L1과 M2rot_D256L1을 **1차와 같은 `.alls`·같은 보정 사진·같은 명령**으로 각각 다시 optimize → compile (같은 PC, DFC 5.4.0, `CUDA_VISIBLE_DEVICES=-1`, `flock ~/siglip2_work/dfc.lock` 안, 시작 전 `free -g` 여유 ≥ 15 GB).
  - rothad: HAR `~/siglip2_work/hailo_g3q/siglip2_rothad_b16_224.har`(`303a8828…`), `.alls` `siglip2/hailo/g4_D256L1_official_b9fc2.alls`, 1차 HEF `7243b76c…` (G3Q-H 6절, 브랜치 `siglip2/g3q-rotate`)
  - M2rot: HAR `~/siglip2_work/g3b_20261018_215006/siglip2_m2rot.har`, `.alls` `g3r_M2rot_D256L1equiv.alls`, 1차 HEF는 `../compile/files.txt`
  - 보정 `calib_highres256_s0_nhwc.npy` (sha256 `91f551e3…`)

## 2. 판정 (팀 결정, 결과 전에 고정 — D-043에 같은 내용)
- **비교:** AGX Hailo-10H 실칩 dev 전체(5,923장)에서 **"M2rot 2차 빌드 − rothad 2차 빌드"** 고화질 5종 Δtop-1, 층화 짝 부트스트랩 95% 구간 (시드 0, 1000회, 1차와 같은 방법 `s2_hailo_chip_eval.py`).
  - **하한 > 0 → D-043 확정: Hailo 최종을 M2rot으로 교체**
  - **0을 포함 → rothad 유지 (D-042 그대로)**
  - **상한 < 0 → rothad 유지**
- 측정 방법은 1차(`../README.md` 1~5절)와 같다: `s2_hailo_chip.py eval --x-npy` (FLOAT32 입출력), FP32 기준·clf는 데스크탑 dev 캐시, 세션 시작 온도 max(ts0, ts1) < 64.6 °C (쉬는 온도 + 2 °C, 5절 사후 변경 그대로).

## 3. 기록용 (판정에 쓰지 않음)
- HEF sha256이 1차와 같은지 (두 구성 각각).
- 1차 ↔ 2차 빌드끼리의 실칩 짝 차이 (rothad끼리, M2rot끼리) — 1차 결과 +0.85 (rothad − D256L1, G3Q-H 9절), +0.53 (M2rot − rothad, `../README.md` 6절)과 나란히 표로.
- rothad HAR SDK_NATIVE(float) 에뮬레이션 vs FP32 ONNX 임베딩 동치 (cos 최소, dev 64장).
- 지연: 2차 빌드 둘로 다시 잰다 (배치 1, 워밍업 20, 200회, 쉬는 온도 + 2 °C 미만에서 시작, 3세션 번갈아, 3세션 중앙값).

## 4. 보고서용 메모
- **회전은 수학적 동치**다: 회전 FP32 ONNX vs 원래 FP32 임베딩 cos 최소 0.99999988 (dev 64장, G3Q 8절).
- **M2는 상수 캐시 근사**다: 원래 대비 cos 최소 0.9994 (`../../../G3R_regcache/g3a_mask/maskcheck.jsonl`), FP32 dev 전체 Δtop-1 +0.10 [0.00, +0.22]%p (G3R R1 FP32, `../../../G3R_regcache/fp32_full`).

## 5. 결과 (10/19 03:38~05:54)
### 2차 빌드 (데스크탑, 커밋 `8342996` dirty 0, `run_info.txt`)
| 구성 | 입력 (1차와 같음) | optimize | opt HAR sha256 (1차) | 출력 SNR (1차) | compile | HEF 크기·컨텍스트 (1차) | HEF sha256 (1차) |
|---|---|---|---|---|---|---|---|
| rothad | HAR `303a8828…`, `.alls` `96696f76…`, 보정 `91f551e3…` | rc=0, 20분 48초 | `7c68d009…` (`a51af7af…`) | 16.63 dB (16.63) | rc=0, 25분 | 107,540,480 B·17 (같음) | **`705c3a81…`** (`7243b76c…`) |
| M2rot | HAR `ab7b324a…`, `.alls` `485adb36…`, 보정 같음 | rc=0, 27분 39초 | `ca1d0dbc…` (`7aa2abe6…`) | 18.03 dB (18.03) | rc=0, 28분 54초 | 110,436,352 B·21 (같음) | **`040a3ae1…`** (`94fff097…`) |
- **HEF sha256은 1차와 다르다.** 바이트 비교(`cmp -l`): rothad 326바이트, M2rot 174바이트만 다르고, 모두 파일 앞부분(13번째 바이트부터, M2rot는 앞 175 KB 안)에 있다. 헤더·메타데이터(빌드 시각 등)로 보이지만 형식은 확인하지 않았다. opt HAR sha256도 다르다 (HAR 안 메타데이터로 추정, 확인 안 함).
- SDK_NATIVE 동치 (rothad 파싱 HAR, dev 64장, `native_check.jsonl`): **cos 최소 0.99999988**.

### 실칩 (AGX Hailo-10H, HailoRT 5.3.0, 05:23~05:54, `chip/`, 커밋 `e0dff66`의 `chip/run.sh`)
- 시작 전 AGX 사용자 0, 다른 Hailo 작업 0. HEF 양쪽 sha256 확인(`hef2_sha256.txt`). 모든 세션 시작 온도 < 64.6 °C (eval 62.6·64.2, 지연 64.3·64.6·64.5, run2 64.5·64.6 °C, `chip/temps.jsonl`).
- **2차 빌드 실칩 임베딩이 1차 빌드와 비트 단위로 같다**: `emb_rothad2` sha256 `e5a6de9d…` = 1차 rothad, `emb_M2rot2` `0527f695…` = 1차 M2rot (`chip/emb_sha256.txt`, 1차 `../chip_full/emb_sha256.txt`). rothad 1차를 다른 날 두 번 잰 임베딩(G3Q·G3RQ)도 비트 단위로 같았다 → 칩 실행은 결정적이다.

| 비교 (실칩 dev 전체, 고화질 5종, 시드 0·1000회) | Δtop-1 [95%] | cos 차 | 예측 일치 | CIFAR A 차 |
|---|---|---|---|---|
| **판정: M2rot 2차 − rothad 2차** | **+0.53 [+0.03, +1.02]** | +0.0096 | 95.7% | +2.3 |
| (기록) rothad 2차 − rothad 1차 | 0.00 [0.00, 0.00] | 0 | 100% | 0 |
| (기록) M2rot 2차 − M2rot 1차 | 0.00 [0.00, 0.00] | 0 | 100% | 0 |
| (참고, 1차) M2rot − rothad (`../README.md` 6절) | +0.53 [+0.03, +1.02] | +0.0096 | | +2.3 |
| (참고, 1차) rothad − D256L1 (G3Q-H 9절) | +0.85 [+0.27, +1.45] | +0.0112 | | +1.8 |
- 파일: `chip_eval_vs_*.json` (`s2_hailo_chip_eval.py --k 0`, 분석 폴더 `~/siglip2_work/hailo_repro/analysis/`).

### 지연 (2차 빌드, 배치 1, 워밍업 20, 200회, 세션마다 번갈아, `chip/bench_s*.json`)
| 세션 | rothad 2차 중앙값 ms | M2rot 2차 중앙값 ms |
|---|---|---|
| 1 / 2 / 3 | 46.171 / 46.171 / 46.171 | 47.974 / 47.951 / 47.868 |
| **3세션 중앙값** | **46.171** (1차 46.178) | **47.951 (+3.86%)** (1차 47.855, +3.63%) |
- `run2 -t 10`: rothad 2차 22.28 FPS, M2rot 2차 21.58 FPS (1차와 같음).

### 판정 (2절 규칙 그대로)
- **M2rot 2차 − rothad 2차 = +0.53 [+0.03, +1.02]%p, 하한 > 0 → D-043 확정: Hailo 최종을 M2rot_D256L1로 교체.**
- 기록: 빌드 간 차이는 0 (실칩 출력이 비트 단위로 같음). HEF 파일은 앞부분 수백 바이트만 다르다. 빌드 차이는 결론에 영향이 없다.
- CPU 미세조정(QFT) 결과는 이 판정에 쓰지 않는다 (D-043).
