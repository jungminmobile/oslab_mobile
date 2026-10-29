# 같은 Release 묶음(dev 6종 × 200장)의 사진별 예측 — 세 CPU 비교 (2026-10-24)

- 입력: 이 PC i7-1165G7(AVX-512 VNNI, Windows) `../i7-1165G7/pred_*.npz` (10/23 portable_eval, 이번에 커밋) /
  준용 Ultra 9 285K(avx_vnni, WSL2)·Jetson AGX Orin(ARM dotprod) `origin/siglip2/b32-portable-junyong` (`3356672`) `out_junyong_285k/`·`out_junyong_agx/`
- 모두 Release `siglip2-b32-submit-20261023`의 같은 파일, ORT 1.25.0, 4스레드. 새로 실행한 것은 없다(파일 비교만).
- 명령: `"$SIGLIP2_PYTHON" siglip2/scripts/xplat_compare3.py` → `compare_3way.json`

| 모델 | i7 ↔ 285K (x86 두 대) | i7 ↔ AGX | 285K ↔ AGX |
|---|---|---|---|
| FP32 | 0장 다름, cos 차이 최대 2.4e-7 | 0장, 3.0e-7 | 0장, 2.4e-7 |
| **O-AB (제출)** | **1장 다름**(Pets), 맞음/틀림 바뀐 것 0장, cos 차이 최대 0.0048 | 25장, 맞음/틀림 17장, 0.026 | 26장, 17장, 0.026 |
| M2rot (출력 재반올림 있음) | **0장, cos 차이 0 (비트 동일)** | 47장, 26장, 0.042 | 47장, 26장, 0.042 |

- 독립 재계산: 285K ↔ AGX 값(26/17, 47/26, 최대 cos 차이)이 준용 `compare_junyong.json`과 같다.
- 해석: x86(VNNI) 두 대에서 M2rot은 비트 동일, O-AB는 사진 1장만 다르다(정답 여부는 같음). O-AB의 `MatMulIntegerToFloat`(정수 곱 뒤 float 출력)가
  명령어 집합에 따라 float 반올림이 조금 다른 것으로 보인다(확인 안 함). ARM은 두 INT8 모델 모두 약 2~4%의 사진이 다르지만 고화질 평균 Δ는 같다(준용 README 2절).
- 데이터셋별 다른 사진(O-AB, i7 ↔ AGX): CIFAR A 12, Pets 1, Flowers 1, Cars 6, Food 3, Caltech 2.
