# 제출 ONNX 개수 세기 (2026-10-24, 보고서 2장·5.2 숫자 출처)

`"$SIGLIP2_PYTHON" siglip2/scripts/onnx_counts.py <O-AB 7a01044c> <FP32 be19a8ad> <M2rot FP32 23236123(Release)>` → `onnx_counts.json`. 최적화 전 파일 기준(ORT 실행 때 융합한 뒤 개수와 다름).

| 항목 | 값 |
|---|---|
| O-AB 노드 | QuantizeLinear 105, DequantizeLinear 158, MatMul 79, SkipLayerNormalization 24, LayerNormalization 2, Softmax 13, Conv 1 |
| O-AB INT8 저장 값 | 92,102,400개 = 가중치 MatMul 53개 92,012,544개(92.01M) + 채널별 영점 89,856개 |
| 레지스터 캐시 상수 (바탕 M2rot FP32) | 31개 텐서(`kc_`·`vc_` 20, `uc_` 7, 마스크 4), **20,992개(약 0.02M)** |
| 회전으로 접힌 LayerNorm γ·β | 36,864개 (원래 94,552,320 − 36,864 + 20,992 = 바탕 94,536,448) |
- 보고서 v9의 "레지스터 캐시 상수 약 0.16M"은 틀렸다. 0.16M(164,233)은 O-AB 파일 전체 값 개수와 원래 FP32의 차이(눈금·영점 포함)였다 → 보고서 정정.
