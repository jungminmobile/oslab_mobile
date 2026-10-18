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
