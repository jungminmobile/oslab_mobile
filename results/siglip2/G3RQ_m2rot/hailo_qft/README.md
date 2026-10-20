# G3RQ-QFT: Hailo 최종(M2rot_D256L1, D-043)에 CPU 미세조정 (설계 2026-10-19 — 1~3절은 결과 전에 고정)

> **별도 실험이다. 결과로 D-043을 바꾸지 않는다.** 채택 여부는 따로 정한다 (용 10/19).

## 1. 구성
- 입력 HAR `~/siglip2_work/g3b_20261018_215006/siglip2_m2rot.har` (sha256 `ab7b324a…`), 보정 `calib_highres256_s0_nhwc.npy` 256장 그대로 (`91f551e3…`, D-035).
- `.alls`: `siglip2/hailo/g3rq_m2rot_qft.alls` = `g3r_M2rot_D256L1equiv.alls`(D-043 구성) + `post_quantization_optimization(finetune, policy=enabled, dataset_size=256, epochs=4, batch_size=2)`. 그 밖의 미세조정 설정은 DFC 5.4.0 기본값 (학습률 2e-4/8×2, l2rel, 출력 층 손실, cosine restarts, warmup 1 epoch — `../../G3Q_rotate/hailo_qft/README.md` 2절).
- `CUDA_VISIBLE_DEVICES=-1` (GPU 경로 불가, 같은 README 5절). `flock ~/siglip2_work/dfc.lock` 안, 시작 전 `free -g` 여유 ≥ **20 GB** (CPU 시험에서 최대 RSS 25 GB).
- 시간: CPU 시험 환산 약 40분. **optimize가 2시간을 넘으면 멈추고 보고** (`timeout 2h`).

## 2. 판정 규칙 (결과 전에 고정)
- 스크리닝: `s2_dfc_screen.py --k 200 --bs 8` (dev 6×200, SDK_QUANTIZED, CPU). 기준 M2rot_D256L1 같은 PC 스크리닝 고화질 cos **0.9866** (`../hailo/screen.jsonl`).
- **고화질 cos ≥ 0.9866 + 0.003 = 0.9896 → "개선 후보"** → compile → AGX 실칩 dev 전체 짝 비교 (M2rot_D256L1 대비, 층화 짝 부트스트랩 시드 0·1000회) + 배치 1 지연 (세션 시작 온도 쉬는 온도 + 2 °C = 64.6 °C 미만, 3세션 번갈아). **하한 > 0이면 팀에 제안.**
- 0.9896 미만이면 "개선 없음"으로 기록하고 끝.
- 비교 기준 HEF: D-043 확정 HEF (M2rot 2차 빌드 `040a3ae1…`, 1차와 출력 같음).

## 3. 기록
- 손실 곡선(epoch별 distill loss), 출력 SNR (미세조정 없음 18.03 dB), 시간·최대 RSS.

## 4. 결과 (10/19 05:56~06:51, 커밋 `9d77155` dirty 0, `run_info.txt`·`screen.jsonl`)
- optimize rc=0, **45분 1초** (바이어스 보정 18분 46초, 미세조정 256장 × 4 epoch = 512스텝, 17분 4초), **최대 RSS 30.3 GB** — CPU 시험의 25 GB보다 크다. 시작 시 메모리 여유가 29 GB여서 한계에 가까웠다 (다음에는 여유 ≥ 32 GB를 기준으로).
- 출력 SNR **17.45 dB** (미세조정 없음 18.03 dB).
- distill loss (l2rel, 출력 `ew_add57`, epoch별 누적 평균): 첫 스텝 0.121 → epoch 1 끝 0.172 → 2: 0.225 → 3: 0.198 → 4: 0.170. **줄지 않았다** (기본 학습률 2e-4/8×2 = 5e-5, cosine restarts 1 epoch 주기).

| 구성 (dev 6×200, SDK_QUANTIZED) | 고화질 cos | Δtop-1 | CIFAR A (top-1, cos) | Pets | Flowers | Cars | Food | Caltech |
|---|---|---|---|---|---|---|---|---|
| M2rot_D256L1 (미세조정 없음, `../hailo/screen.jsonl`) | **0.9866** | −0.2 | | | | | | |
| M2rot_D256L1 + QFT | 0.9785 | −0.3 | 74.0, 0.935 | 92.0 | 85.0 | 93.0 | 88.0 | 95.0 |

- **판정: 0.9785 < 0.9896 → 개선 없음.** 오히려 cos −0.0081, SNR −0.6 dB로 나빠졌다. compile·실칩은 하지 않았다.
- 해석 (가설, 확인 안 함): 보정 256장으로 4 epoch를 돌리는 동안 손실이 줄지 않았으므로, 기본 학습률·스케줄이 이 모델(이미 바이어스 보정으로 맞춘 상태)에는 맞지 않거나 256장이 너무 적다 (DFC 기본 dataset_size 1024). 학습률을 낮추거나 데이터를 늘리는 것은 데이터 구성 규칙(D-035)과 이번 실험 범위 밖이다.
- D-043은 그대로다 (이 실험은 별도 기록).
