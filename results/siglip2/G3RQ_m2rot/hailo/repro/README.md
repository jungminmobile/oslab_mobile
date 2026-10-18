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

## 5. 결과
(실행 뒤 덧붙인다.)
