# B/32 제출 모델 O-AB: ORT `enable_mem_pattern=False`(E12-1 C3)의 메모리·출력 (2026-10-24, 이 PC)

- 목적: 보고서 메모리 수치가 어느 세션 설정 값인지 정하고(준용 v9 검토 M7-1), B/32에서 C3 효과를 직접 잰다. E12-1(정민, 쭈리 재현)은 B/16 모델로만 쟀다.
- 코드: 브랜치 `exp/E12-1-jjury` `7228d87`(dirty 아님)의 `e12/bench_mem_session.py`·`e12/check_equal.py` 그대로. `SIGLIP2_ARTIFACTS=D:/mokerton/artifacts/siglip2_b32`(B/32 256 캐시), 공용 잠금 안.
- 명령: `bench_mem_session.py --models oab=<7a01044c> fp32=<be19a8ad> --combos C0 C3 --threads 4 --n 50 --repeats 3 --interval-ms 10`, `check_equal.py --model <7a01044c> --combos C0 C3 --n 200 --threads 4`
- 실행 스크립트(임시): scratchpad `mem_b32.sh` — 위 두 명령을 잠금으로 감싼 것뿐.

| 모델 | C0 (ORT 기본 세션) peak working set | C3 (`enable_mem_pattern=False`) | 변화 | 3회 모두 낮음 |
|---|---|---|---|---|
| **O-AB** | 214.5 MiB (213.3 / 214.5 / 214.7) | 206.0 MiB (205.9 / 206.0 / 207.7) | **−4.0%** | 예 |
| FP32 | 462.7 MiB | 460.5 MiB | −0.5% | 예 |

- 출력: 고화질 5종 × 200장에서 C3 = C0 **비트 동일**(max_abs_diff 0.0, 예측 100% 같음).
- 주의: 이 스크립트의 peak(추론 50장, 표본 스레드 10 ms)는 `cross_latency.py --mem`의 값(O-AB 193.3 MiB)과 측정 정의가 달라 절대값을 섞지 않는다. 보고서 메모리 −57.4%는 `cross_latency.py` 기본 세션 값이다.
- B/16에서는 C3가 −8.5~−9.3%였다(E12-1). B/32는 −4.0%로 더 작다.
- C3는 우리 실행 스크립트에 넣었을 때만 적용되고, 채점 측이 세션을 직접 만들면 적용되지 않는다 → 보고서에는 "선택 사항"으로만 적는다.
