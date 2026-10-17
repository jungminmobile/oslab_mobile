# Hailo D256L1 교차 확인 — 정민 노트북 vs 쭈리 PC (2026-10-17, D-011·D-027)

**결론: 통과 (같은 품질). 같은 입력·같은 DFC 5.4.0으로 정민 노트북에서 다시 만든 D256L1은 에뮬레이터 top-1이 같고(Pets dev 200장 92.0%), HEF 크기(99,831,808 B)와 컨텍스트 수(17)도 같다. 다만 비트 단위로 같지는 않다** — 사진별 cos 차이 평균 0.005(최대 0.041), 예측 일치 97%(200장 중 6장 다름), cos 평균 0.9756 vs 0.9752.

## 방법
- 묶음: GitHub pre-release `siglip2-hailo-crosscheck-20261017` (FP32 b1 ONNX, 보정 256장, `g4_D256L1_official_b9fc2.alls`, Pets dev 200장, 쭈리 PC 결과). 입력 sha256은 정민 PC에서도 모두 OK (`jungmin_raw/crosscheck_steps.txt`)
- 실행: `siglip2/hailo/crosscheck/run_crosscheck.sh` (COMPILE=1) — parse → optimize → 에뮬레이터(SDK_QUANTIZED) → compile → profiler
- 환경: 정민 노트북 Intel Core Ultra 7 258V (8코어), WSL2 Ubuntu 22.04, Python 3.10.12, DFC 5.4.0, GPU 없음 / 쭈리 PC i7-1165G7, WSL2 Ubuntu-22.04, DFC 5.4.0, GPU 없음
- raw: `jungmin_raw/` (작은 로그만. hailo_sdk.core.log 4.5 MB 등은 `D:\mokerton\artifacts\siglip2\share\from_jungmin_20261017\`)

## 결과

| 항목 | 정민 노트북 | 쭈리 PC | 같음? |
|---|---|---|---|
| optimize 알고리즘 순서 | Mixed Precision → LayerNorm Decomposition → Statistics → … → Bias Correction → Layer Noise Analysis | 같음 | O |
| 에뮬레이터 Pets 200장 top-1 | 92.0% | 92.0% | O |
| cos 평균 (FP32 대비) | 0.97561 | 0.97519 | 거의 (+0.0004) |
| 사진별 cos 차이 | 평균 0.0051, 최대 0.041 | — | X (비트 단위 아님) |
| 예측 일치율 | 97% (194/200) | — | X |
| HEF 크기 | 99,831,808 B | 99,831,808 B | O |
| 컨텍스트 (프로파일러) | 17 | 17 | O |
| HEF sha256 | `e0be125a…` | `b33a266f…` | X |
| 컴파일 시간 | 37분 | 1시간 25분 | (정민 노트북이 빠름) |

- HAR의 sha256은 parse 단계부터 다르다 (`dad5ece6…` vs `3d8e143e…`). HAR은 압축 파일이라 파일 시각 등 메타데이터가 달라 sha256 비교 자체가 의미가 없다.

## 왜 비트 단위로 같지 않나 (가설)
1. **CPU에 따른 부동소수 차이**: 보정 통계·바이어스 보정은 TensorFlow CPU 계산이다. CPU 명령어(쭈리 AVX-512 vs 정민 AVX-VNNI/AVX2)·코어 수에 따라 합산 순서가 달라 눈금(scale)이 아주 조금 달라질 수 있다. 8비트 반올림 경계 근처 값이 다른 쪽으로 떨어지면 출력이 달라진다.
2. **DFC 안의 무작위성**: 보정·바이어스 보정에서 표본 순서나 무작위 초기화를 쓴다면 같은 PC에서도 실행마다 다를 수 있다.
- 두 가설을 구분하려면 쭈리 PC에서 optimize를 한 번 더 돌려 출력이 같은지 보면 된다 (WSL 약 1시간, 미실행).

## 보고서에 쓸 점
- "같은 입력·같은 DFC 버전이면 다른 PC에서도 같은 품질(top-1·HEF 크기·컨텍스트)의 HEF가 나온다. 다만 양자화 최적화 결과가 비트 단위로 같지는 않아, 본선 실칩 수치는 실제로 올린 HEF 하나로 잰다."
- 본선에서는 실칩에 올린 HEF의 sha256을 기록하고, 그 파일로만 측정한다.

## 같은 PC 재실행 시도 (10/18 01:01) — 중단
- `scripts/tonight_1018.sh`: 01:01:26 시작(CPU 60초 5%), parse 성공(37초) 뒤 optimize 초반 01:02:05에 Windows 여유 메모리 0.63 GB.
- 메모리 감시(`scripts/mem_guard_wsl.py`)가 감지했지만 콘솔 인코딩 오류(cp949로 "—" 출력)로 죽어 WSL을 못 내렸다 → 01:02:30 수동으로 WSL 종료, 잠금 해제. 여유 메모리 1 GB 미만이 약 25초 이어졌다. 개인연구 세션 확인: 그 시각 두 폰(S24U·S23U) run이 막 시작됐지만 둘 다 정상 완료·QC 통과, adb 시간 초과 없음 — 손실 없음.
- 원인: 이 PC는 밤에도 WSL 없이 여유 메모리가 약 4~5 GB뿐이다(메모리 압축 2.5 GB 등). DFC optimize는 6~10 GB가 필요해 폰 측정과 함께 안전하게 돌릴 수 없다.
- 대책: 감시 스크립트 출력 UTF-8 고정·종료를 출력보다 먼저. 다음 시도는 시작 전 여유 메모리 ≥ 9 GB를 확인하고, 폰 측정이 없는 시간에만. 가설 1·2 판별은 결과에 영향이 없어 우선순위 낮음.
