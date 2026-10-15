# portable_eval.py 검증 — 이 PC (10/16 03:21~03:32)

`siglip2/scripts/portable_eval.py`(다른 기기 측정용, numpy + onnxruntime만)를 공유 묶음(`share/20261015_xplat`, GitHub pre-release `siglip2-xplat-20261015`)으로 이 PC에서 돌린 결과가 기존 스크리닝과 **정확히 같다**:

| 모델 | portable_eval (dev 6×200) hr Δtop-1 / cos | 기존 `s2_screen.py` (같은 200장) |
|---|---|---|
| fc2L2 S8 | +0.3 / 0.9557 | +0.3 / 0.9557 (`../../G3_cumulative/screen.jsonl`) |
| fc2L2 U8 + rr | +0.1 / 0.9545 | +0.1 / 0.9545 (`../../G3_cumulative/qkv_precision/screen.jsonl`) |
| FP32 | 0.0 / 1.0 | — |

→ 준용(ARM Orin)이 같은 묶음으로 잰 값을 이 표와 바로 비교할 수 있다. 실행: `siglip2/scripts/night_1016.sh` 3단계, 커밋 `run_info.json`.
