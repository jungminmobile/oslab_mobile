# G6 최종 test 1회 평가 — SigLIP2 ORT 최종 (2026-10-16 22:45 ~ 10-17 07:43)

**결론: 최종 모델(fc2L2 U8 + reduce_range)은 test 고화질 5종 평균에서 FP32 90.34% → 89.51%, Δ −0.83 [−1.01, −0.66]%p다.** dev(−0.78 [−1.26, −0.31])와 거의 같다. CIFAR-100 −4.71, ImageNetV2 −1.95%p (별도 표기, D-017). test는 이 1회만 썼다.

## 방법
- 모델: FP32 베이스라인(`siglip2_image_fp32.onnx`, sha256 `333cad08…`) + 최종(`siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx`, `5eee7839…`, D-030)
- `siglip2/s2_test_eval.py` (데이터셋마다 1회, `done_<set>.json`), `siglip2/scripts/test_g6.sh` (데이터셋마다 공용 잠금). 사진·클래스 이름·프롬프트는 `rn50/datasets.py`, 전처리·텍스트 분류기는 dev와 같은 SigLIP2, 정수 커널(ORT_ENABLE_ALL), 배치 50, 4스레드
- 데이터: CIFAR-100 test 1만, ImageNetV2 matched-frequency 1만(HF `vaishaal/ImageNetV2`, sha256 `data_sha256.txt`), Pets 3,669 · Flowers 6,149 · Cars 8,041 · Food 25,250 · Caltech 6,084 (공식 test 전체)
- 요약: `siglip2/s2_test_summary.py` (커밋 `1845f58`) — 같은 사진끼리 짝지은 층화 부트스트랩(시드 0, 1000회, PRD 5절)
- 모든 행 dirty=false. 행의 commit(85cf04b·4eb3aed·29e5e75)은 실행 시점 HEAD이고, 그 사이 바뀐 것은 문서뿐이다 (평가 코드 `s2_test_eval.py`는 e6a8d2b 이후 변경 없음)

## 결과 (`test.jsonl`, `test_summary.json`)

| 데이터셋 | 장수 | FP32 top-1 | 최종 top-1 | Δ [95% 구간] | cos |
|---|---|---|---|---|---|
| Pets | 3,669 | 95.28 | 94.49 | −0.79 [−1.31, −0.27] | 0.968 |
| Flowers | 6,149 | 85.69 | 84.18 | −1.51 [−2.03, −1.02] | 0.959 |
| Cars | 8,041 | 93.28 | 92.21 | −1.07 [−1.46, −0.70] | 0.951 |
| Food | 25,250 | 92.76 | 91.60 | −1.16 [−1.36, −0.93] | 0.961 |
| Caltech | 6,084 | 84.70 | 85.08 | +0.38 [0.00, +0.71] | 0.942 |
| **고화질 5종 평균** | 49,193 | **90.34** | **89.51** | **−0.83 [−1.01, −0.66]** | 0.956 |
| CIFAR-100 (별도) | 10,000 | 76.92 | 72.21 | −4.71 [−5.39, −4.07] | 0.881 |
| ImageNetV2 (별도) | 10,000 | 71.28 | 69.33 | −1.95 | 0.964 |

- **dev와 비교**: 고화질 Δ dev −0.78 → test −0.83 (dev에서 고른 구성이 test에서도 같은 수준). 데이터셋별 순서도 비슷하다 (Flowers가 가장 크게 떨어짐).
- **Caltech FP32가 dev(95.0)보다 낮다(84.7)**: 공식 test에 background 등 어려운 클래스 사진 비중이 큰 것으로 보인다 (추정). RN50 트랙과 같은 사진·클래스 처리다.
- **ImageNetV2 동점 주의**: OpenCLIP의 ImageNet 클래스 이름에 같은 이름이 두 쌍 있다 ("missile" 657·744, "sunglasses" 836·837) → 텍스트 벡터가 같아 점수가 정확히 동점이 된다. 공통 `topk_metrics`(argsort, RN50과 같은 함수)는 71.28 / 69.33(Δ −1.95), 사진별 npz(argmax, 앞 번호 선택)는 71.33 / 69.36(Δ −1.97 [−2.44, −1.48])으로 0.05%p 다르다. 보고서는 RN50과 같은 함수 값을 쓴다. 해당 라벨 사진은 40장(0.4%)이다.
- Hailo 구성은 test로 평가하지 않았다 (D-040, 본선 실칩에서).
