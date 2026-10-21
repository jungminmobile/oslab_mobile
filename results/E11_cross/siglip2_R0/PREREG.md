# E11b 사전 등록 — SigLIP2 R0에 RN50 기법 적용 (결과 전 고정, 2026-10-21 15:00 KST)

- 정민 지시: "첫 번째 예선 후보(R0)에 S1~S3를 모두 적용하고, 메모리·속도·정확도 중 하나라도 좋아지면 알려줘."
- 기준 모델 R0 = SigLIP2 B/16 fc2L2, U8 활성값 + reduce_range, Percentile 99.999, highres 256장 시드 0, MatMul만 INT8(블록 5·9 fc2 FP32), hist-chunk 16 (D-030).
  - 노트북에서 같은 명령으로 만든 파일 `C:\s2art\onnx\siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx` sha256 `cb710af4…`를 기준으로 쓴다 (Release `5eee7839…`와 CPU 차이로 다름, E5a에서 dev −0.62%p).
- 코드: 준혁 SigLIP2 코드(worktree `oslab_s2`, 1cc17c4)를 고치지 않고 쓴다. 새 스크립트는 `oslab_mobile/e11b/`(브랜치 `jungmin/e11-cross`).
- 측정: 정민 노트북 258V, 충전 중, 전원 모드 균형, s2_venv(ORT 1.25.0), 정수 커널. 정확도 = dev 전체(CIFAR A + 고화질 5종, `s2_paired_ci.py`), R0 대비 짝 부트스트랩 1000회 층화. 속도 = `s2_bench_speed` 1·2·4T × 3세션. 메모리 = peak RSS.

## 실험
| # | 바꾸는 것 |
|---|---|
| S1 | 보정 데이터: cifar256, mix256, highres64, highres311(후보 풀 전체) |
| S2 | Bias Correction: 양자화된 MatMul 뒤 bias(Add 상수)에서 채널별 평균 쏠림을 앞 층부터 차례로 뺀다 (보정 256장 = 양자화 보정과 같은 사진). |
| S3 | 정수계획: Linear 단위별 INT8/FP32 선택을 4T 지연 예산 안에서 cos 합 최대로 (E10 방식) |

## 판정 (E11과 같음)
- 채택 후보(속도·크기): ① R0 대비 95% 구간이 0 포함 또는 > 0 ② 1·2·4T 모두 R0보다 느리지 않음 ③ 4T −5% 이상 또는 peak RSS −5% 이상 또는 파일 −10% 이상.
- 채택 후보(정확도): ① 하한 > 0 ② 4T 지연 +5% 이내 ③ 모든 스레드에서 FP32보다 빠름.
- 그 외 효과 없음, 상한 < 0이면 해로움. 여러 설정 중에서는 조건을 만족하는 것 중 정확도 점추정이 가장 높은 하나만 후보.
- 정민 요청에 따라 "메모리·속도·정확도 중 하나라도 R0보다 좋은 것"(점추정, 판정과 별개)은 따로 표시해 바로 알린다.

## 계획서와 다른 점 (결과 전)
1. 기준이 M2rot_ort가 아니라 R0다 (M2rot FP32 파일 없음, 정민 지시).
2. S1의 highres512 → highres311: 노트북 SigLIP2 보정 후보 풀이 311장뿐이다.
3. S2 관문: SigLIP2 그래프에는 Conv가 없어 정수 커널 확인(③ 정수)은 하지 않는다. zero-delta sha = R0, bias만 다름, fake 감소율 ≥ 90%만 본다. RN50 E5b에서 고친 두 가지(FP32 기준 텐서 이름, Add 출력에서 확인)를 처음부터 반영한다.
