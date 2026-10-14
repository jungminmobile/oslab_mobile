# G3 후보 지연 측정 (비공식, 이 PC i7-1165G7)

## 첫 측정 — 오염, 표에 쓰지 않음

| 파일 | 스레드 | 시각 | 상태 |
|---|---|---|---|
| `speed_20261014_115955.json` | 1 | 11:43~12:00 | 다른 세션의 pip 설치(11:45~12:10, 잠금 없이 실행)와 겹침 → 오염 |
| `speed_20261014_121212.json` | 2 | 12:00~12:12 | 같은 이유로 오염. 회차 사이 편차 큼 (FP32 259 / 229 / 224 ms) |
| `speed_20261014_122402.json` | 4 | 12:12~12:24 | 설치 뒤 Windows Defender 검사(CPU 약 20%)와 겹침, 회차 편차 약 25% → 다시 잼 |

- 실행: `scripts/speed_after_g4.sh` → `scripts/g3_cumulative.sh speed` (워밍업 20, 200장 × 3회 번갈아, 배치 1, PRD 5절).
- 그 전 두 번의 시작(10:50, 10:52)은 측정 전에 실패했다. 잠금 도구가 `bash`를 실행할 때 WSL bash가 잡혀 환경 변수가 전달되지 않았다 (`speed_after_g4.sh` 주석, 커밋 `349ab5b`로 수정). 측정값은 없다.

## 재측정

- 스레드 1·2: `scripts/speed_rerun_t12.sh`, 스레드 4: `scripts/speed_rerun_t4.sh`. 같은 모델·같은 조건이고 공용 잠금 안에서 잰다.
- 결과는 이 폴더의 새 `speed_*.json` 파일이다. 표는 결과가 나오면 `../README.md`와 `../../candidates/README.md`에 채운다.

## 중단 기록 (10/14 14:07)
- `runs.jsonl` 마지막 줄(14:07, 스레드 1)은 시작 직후 Windows Update(`wuaucltcore.exe` 약 5코어)와 Defender가 겹쳐 중단한 실행이다. 결과 json은 만들어지지 않았다.
- 다음 측정: `scripts/speed_when_idle.sh` (Windows Update가 조용해지면 잠금 → 잠금 안에서 전체 CPU 25% 미만 확인 후 측정).
