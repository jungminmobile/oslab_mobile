# 2026-10-07 야간 실험 결과

계획·해석은 `PRD_2026-10-07.md` (6절 결과, 8절 결론).

- 측정 환경: 클라우드 리눅스, CPU 2스레드 (AVX512-VNNI 지원), Python 3.11, ONNX Runtime 1.25
- 평가 데이터: dev (train 분할, 데이터셋마다 약 1000장). **test 점수 아님**
- `exec` 필드: `int` = 정수 커널(ORT 기본 최적화), `fake` = fake-quant(Q/DQ를 소수로 계산)

| 파일 | 내용 |
|---|---|
| `*/baselines.jsonl` | FP32 기준값 |
| `*/evaluate.jsonl` | 모델 × 데이터셋 한 줄씩: top1, top5, cos_mean, MiB, exec |
| `*/speed.json` | 장당 ms (번갈아 3회 반복) |
| `*/night_progress.log` | 실행 로그 (시각·단계) |
| `rn50/tables/` | R1R2(단계별·누적), R3(stem+단계), R4(보정 출처), R5(stem 내부), R6·R6b(4비트), R7(속도), R8(fake-quant 재확인) |
| `rn50x4/tables/` | X1X2, X4, X6, X7, X8 (RN50x4 같은 실험) |

표 다시 만들기: `cd rn50 && python summarize.py --results ../archive/2026-10-07_night/results_rn50 --filter c4`
