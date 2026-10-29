# G6_m2rot_ort / profile — 최종 예선 모델(M2rot_ort)의 ORT 연산별 시간 (10/20)

목적: 예선 모델(CPU, ORT)에서 속도를 더 올릴 여지가 있는지 확인. 특히 양자화 변환 노드(QuantizeLinear·DequantizeLinear, Q/DQ)가 시간의 얼마를 쓰는지.
사전 기준(결과 전, 쭈리와 공유): Q/DQ 등 줄일 수 있는 부분이 **15% 이상**일 때만 예선 모델 변경(=test 재사용)을 검토한다.

실행 (이 PC i7-1165G7, 공용 잠금, 커밋 2ccb6fe, dirty 아님):
```bash
RN50_RESULTS=../results/siglip2/G6_m2rot_ort/profile "$SIGLIP2_PYTHON" s2_profile_ort.py --threads 1 4 \
  --models fp32=siglip2_image_fp32.onnx R0=siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx \
           M2rot=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
```
배치 1, pets_dev 워밍업 10장 + 30장, ORT_ENABLE_ALL. raw: `profile.jsonl`, 환경 `env.json`.

## 결과 (M2rot_ort, 커널 시간 비율)

| 연산 묶음 | 1스레드 | 4스레드 |
|---|---|---|
| 정수 행렬곱 (QLinearMatMul 79개) | 56.8% | 50.0% |
| **Q/DQ 변환 (Q 79, DQ 66)** | **6.2%** | **12.8%** |
| QLinearSoftmax (13) | 11.8% | 7.5% |
| BiasGelu (13) | 8.2% | 6.2% |
| Add·Mul·Sub (M2 마스크 연산 포함: Add 96, Mul 44, Sub 31) | 7.7% | 10.0% |
| LayerNorm | 4.2% | 3.5% |

## 결론

- Q/DQ는 4스레드에서 12.8%로, 스레드를 늘려도 잘 줄지 않는다 (1스레드 6.2%). 그러나 **전부 없애도 상한 약 13%**이고,
  LayerNorm·GELU·Softmax 앞뒤의 변환은 형식을 바꿔도 남으므로 실제로 줄일 수 있는 양은 그보다 작다.
- M2 마스크 연산(Mul·Sub와 늘어난 Add)은 4스레드에서 약 4~5%다.
- **사전 기준(15%)에 못 미친다 → 예선 모델(M2rot_ort)은 바꾸지 않는다.** test를 다시 쓸 이유가 없다.
- 주의: 프로파일링은 자체 오버헤드가 있고 모델당 30장 1회라, 모델 사이의 절대 지연 비교에는 쓰지 않는다
  (이번 실행에서 4스레드 R0 349 ms vs M2rot 232 ms로 나왔지만, 정식 측정(`../speed`)에서는 둘이 ±3% 안이었다). 같은 모델 안의 비율만 본다.
