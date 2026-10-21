# E12-1 ORT 세션 메모리 설정 (런타임 최적화) — 실험계획서

- 작성: 2026-10-22, 정민 (계획: Claude 세션 A / 실행: 별도 채팅방)
- 시리즈: E12 (ONNX 런타임·그래프 최적화 6종) 중 1번
- 상태: **계획 확정 대기**. 결과 후 판정 규칙 변경은 정민 승인 필요
- 폴더: `results/E12-1_ORT세션메모리설정/` (계획서 = 이 파일, 결과 = `results/` 하위)

## 1. 목적
모델 파일(ONNX)을 바꾸지 않고, **모델을 실행하는 파이썬 코드의 ORT 세션 설정 3개**만으로 추론 peak RAM을 줄일 수 있는지 확인한다.
- 분류: **런타임(시스템) 최적화**. 모델 최적화(양자화)와 분리해 보고한다.
- 적용 범위: **예선(ORT CPU)만**. 본선(HailoRT)에는 적용되지 않는다.
- 설정은 ONNX 파일에 저장되지 않는다 → 채택 시 제출 실행 코드 기본값 + README 명시 필요.

## 2. 가설
| 스위치 | ORT 설정 | 기대 |
|---|---|---|
| S-P (prepack) | `so.add_session_config_entry("session.disable_prepacking", "1")` | 가중치 정리본 사본 제거 → 메모리 감소. 행렬곱 느려질 수 있음 |
| S-A (arena) | `so.enable_cpu_mem_arena = False` | 큰 창고 미리 잡기 제거 → 메모리 감소. 할당 비용으로 약간 느려질 수 있음 |
| S-M (mem pattern) | `so.enable_mem_pattern = False` | 배치표 미리 잡기 제거 → 소폭 감소. 입력 크기 고정(224)이라 속도 손해 작음 |

참고(준혁 285K, `G3RQ_m2rot/ort/model_stats`): M2rot_ort 로드 후 191.6 MiB → 추론 중 peak 232.4 MiB. 즉 **peak는 추론 구간**에서 나왔다(+40.8 MiB) → arena·mem pattern이 줄일 여지가 있는 구간.

## 3. 대상 모델
| 이름 | 파일 | sha256 | 위치 (정민 노트북) |
|---|---|---|---|
| FP32 | `siglip2_image_fp32.onnx` (354.5 MiB) | `333cad08e68dc85a5036f8f7396bc61d5548eb69ea365a0c5d83ff15c48840e0` (Release `siglip2-xplat-20261015`과 일치, 10/22 03:25 확인) | `C:\s2art\onnx\` |
| **M2rot_ort** (주 대상) | `siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx` (96,180,230 B) | `e84c3727f51e3b97cd59d9ab9604e01acc78f81493926a0696fbf96dc9c60f9e` (Release `siglip2-m2rot-ort-20261019`과 일치, 10/22 03:25 확인) | `C:\s2art\onnx\` (Release에서 받음, 체크섬 `C:\s2art\onnx\SHA256SUMS_m2rot_ort.txt`) |

- **관문 G0:** 실행 직전 두 파일의 sha256을 다시 계산해 위 값과 같은지 확인하고 `results/g0_sha.json`에 기록한다. 다르면 **실험 중단하고 정민에게 보고**한다 (R0로 대체하려면 정민 승인 필요).
- 기존 측정과 같은 모델 파일·같은 입력 전처리(`oslab_s2/siglip2/s2common.py`)를 쓴다.

## 4. 조합 (5개)
| ID | S-P prepack | S-A arena | S-M mem pattern | 의미 |
|---|---|---|---|---|
| C0 | 켬 | 켬 | 켬 | 기준 (ORT 기본값) |
| C1 | **끔** | 켬 | 켬 | prepack만 끔 |
| C2 | 켬 | **끔** | 켬 | arena만 끔 |
| C3 | 켬 | 켬 | **끔** | mem pattern만 끔 |
| C4 | **끔** | **끔** | **끔** | 전부 끔 |

고정 설정: `graph_optimization_level = ORT_ENABLE_ALL`, `intra_op_num_threads` = 측정 항목별 지정, `inter_op_num_threads = 1`, 환경변수 `OPENBLAS_NUM_THREADS=1`, `HF_HUB_OFFLINE=1`, `PYTHONIOENCODING=utf-8`.

## 5. 측정 방법

### 5-1. 관문 G1 — 출력 동일성 (정확도 대신)
- M2rot_ort, 4스레드, 고해상도 dev 200장(기존 E11b와 같은 이미지·전처리)으로 C0 대비 C1~C4 임베딩을 비교한다.
- 통과: **최대 절대 차이 = 0** (비트 동일). 0이 아니면 dev 5종 top-1 + paired bootstrap(1000회)으로 정확도를 재고 정민에게 보고한다.

### 5-2. 메모리
- (모델 × 조합)마다 **새 프로세스** 3회, 중앙값. 4스레드, batch 1, 50장. 워밍업 없음(peak는 첫 실행 포함).
| 지표 | 정의 |
|---|---|
| **M-peak (주 지표)** | 프로세스 전체 peak working set (`psutil.Process().memory_info().peak_wset`). 기존 `bench_mem.py`와 연속 |
| M-create | 세션 생성 직후 peak working set |
| M-infer | 추론 구간 최대 working set (별도 스레드, 10 ms 간격 샘플링) |
| M-private | 추론 구간 최대 private bytes (`memory_info().private`, 10 ms 샘플링) |
| base | ORT·numpy import + 입력 준비 후, 세션 생성 전 working set |

### 5-3. 속도
- 1·2·4스레드 × 세션 3회. 세션 사이 2분 휴식. 세션마다 조합 순서 회전 (S1: C0→C4, S2: C2→C1, S3: C4→C0 역순 등 — 실제 순서 기록).
- 조합마다 워밍업 20장 → 200장 측정, 장당 지연 중앙값.
- 판정은 **같은 세션 안 C0 대비 비율**의 3세션 중앙값 (노트북 drift 대응).

## 6. 판정 규칙 (사전 등록)
| 판정 | 조건 (모두 만족) |
|---|---|
| **채택** | ① G1 통과 ② M2rot M-peak가 C0 대비 **−5% 이상**, 3회 모두 같은 방향 ③ 4스레드 지연 C0 대비 **+5% 이내** ④ 1·2스레드 지연 **+10% 이내** |
| 부분 채택 | ①② 만족, ③ 또는 ④ 불만족 → "메모리 우선 모드" 옵션만 제공, 기본값 유지 |
| 효과 없음 | M-peak 감소 5% 미만 |

- 여러 조합이 채택이면 M-peak 최소 조합. 차이가 2% 이내면 더 빠른 조합.
- FP32는 판정에 쓰지 않고 공정 비교(모델 효과 vs 런타임 효과 분리)용으로만 쓴다.

## 7. 구현 (실행 채팅방이 작성)
| 파일 | 내용 |
|---|---|
| `e12/session_opts.py` | `make_session(path, combo, threads)` — C0~C4 → SessionOptions. 그 외 설정은 4절 고정값 |
| `e12/check_equal.py` | G1. 입력·전처리는 `oslab_s2/siglip2/s2common.py` 사용 |
| `e12/bench_mem_session.py` | 5-2. 자식 프로세스 + 샘플링 스레드 + psutil |
| `e12/bench_speed_session.py` | 5-3. `--session N`으로 1회분 실행 |
| `e12/run_e12_1.bat` | 단계별 실행용 (단계 사이 알림을 위해 한 번에 다 돌리지 않음) |

- 환경: 정민 노트북(Intel Core Ultra 7 258V), Windows, `s2_venv`(ORT 1.25). 실행 전 `results/env.json`에 OS·Python·ORT·numpy·psutil 버전, CPU, 전원 상태 기록.
- `rn50\.venv`는 수정·삭제 금지. 새 패키지가 필요하면 정민에게 먼저 묻는다.
- 브랜치 `jungmin/e12-runtime` (`jungmin/e11-cross`에서 분기). **로컬 커밋만**, push 금지.

## 8. 결과 파일 (`results/E12-1_ORT세션메모리설정/results/`)
`env.json`, `g0_sha.json`, `equal.json`, `memory.jsonl`, `speed.jsonl`, `run_log.txt`, `REPORT.md`(판정·원인 분석·결론)

## 9. 일정 (E11b S3 종료 후 시작)
| 단계 | 예상 |
|---|---|
| G0 파일·sha 확인 | 10분 |
| 스크립트 작성·점검 (2장 시험 실행) | 40분 |
| G1 동일성 (M2rot) | 약 2분 |
| 메모리 M2rot / FP32 | 약 3분 / 약 4분 |
| 속도 세션당 M2rot / FP32 | 약 7분 / 약 16분 (3세션 합 약 70분 + 휴식) |
| 판정·REPORT·커밋 | 30분 |
| **합계** | **약 3시간** (drift 시 1.3~1.5배) |

## 10. 위험과 대응
| 위험 | 대응 |
|---|---|
| 측정 흔들림 (R0 4T 52 → 81 ms 사례) | 같은 세션 C0 대비 비율, 3세션 중앙값. 전원 연결·절전 끄기·무거운 앱 종료 |
| peak가 세션 생성에 몰려 이득이 가려짐 | M-create / M-infer 분리 보고 |
| 심사위원이 ONNX만 가져다 실행 | 제출 실행 코드 기본값 + README에 설정 3개 명시 |
| 양자화 효과와 섞임 | FP32도 같은 조합으로 측정, 표에서 분리 |

## 11. 보고 표 형식
| 모델 | 설정 | M-peak (MiB) | Δ vs C0 | M-infer | M-private | 4T 지연 (ms) | Δ vs C0 | 1T / 2T Δ | 출력 동일 |
|---|---|---|---|---|---|---|---|---|---|
| FP32 | C0 / 선택 | | | | | | | | |
| M2rot_ort | C0 / 선택 | | | | | | | | |

## 12. 실행 지침 (실행 담당 채팅방용)

### 12-1. 지켜야 할 규칙
- 쉬운 한국어, 간결하게. 실행 전 계획·시간 보고, 약 25%마다 진행 보고.
- 실제로 실행한 것만 보고. 판정 규칙(6절) 변경은 정민 승인 후.
- `rn50\.venv` 수정·삭제 금지. 커밋은 로컬만, push는 목록 보여주고 승인 후.
- E11b S3(`C:\s2art\E11b\run_log.txt`에 `S3 ALL DONE`)가 끝나기 전에는 측정 시작 금지.

### 12-2. 알림 규칙 (정민이 무거운 작업을 끄기 위함)
아래 8개 단계 **시작 3분 전**에 알림(푸시 + 채팅 메시지)을 보내고, 3분 기다린 뒤 시작한다. 단계가 끝나면 "끝, 다음 단계 ○분 뒤"를 보낸다.
| # | 단계 | 예상 | PDF 읽기 |
|---|---|---|---|
| 1 | G1 동일성 (M2rot 5조합) | 약 2분 | 가능 |
| 2 | 메모리 M2rot (5조합 × 3회) | 약 3분 | 가능 (큰 작업만 X) |
| 3 | 메모리 FP32 (5조합 × 3회) | 약 4분 | 가능 (큰 작업만 X) |
| 4 | 속도 세션 1 (M2rot + FP32) | 약 23분 | **닫기** |
| 5 | 속도 세션 2 | 약 23분 | **닫기** |
| 6 | 속도 세션 3 | 약 23분 | **닫기** |
| 7 | (예비) G1 불통과 시 정확도 재측정 | 약 30분 | 가능 |
| 8 | (예비) 재측정이 필요할 때 | — | 단계에 따름 |

### 12-3. 명령어 (cmd)
```bat
:: 0) 공통
cd /d "C:\study\3\2\2026 모바일 학술대회\oslab_mobile"
git switch -c jungmin/e12-runtime jungmin/e11-cross
set "PY=C:\study\3\2\2026 모바일 학술대회\s2_venv\Scripts\python.exe"
set "OUT=results\E12-1_ORT세션메모리설정\results"
set "ONNX=C:\s2art\onnx"
set "M2=siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx"
set "FP=siglip2_image_fp32.onnx"
set OPENBLAS_NUM_THREADS=1
set HF_HUB_OFFLINE=1
set PYTHONIOENCODING=utf-8

:: G0) sha 확인 — M2rot e84c3727f51e3b97cd59d9ab9604e01acc78f81493926a0696fbf96dc9c60f9e, FP32 333cad08e68dc85a5036f8f7396bc61d5548eb69ea365a0c5d83ff15c48840e0
::     (M2rot 파일이 없어졌을 때만: gh release download siglip2-m2rot-ort-20261019 -p "%M2%" -D "%ONNX%")
certutil -hashfile "%ONNX%\%M2%" SHA256
certutil -hashfile "%ONNX%\%FP%" SHA256

:: 스크립트 점검 (2장 시험)
"%PY%" -u e12\check_equal.py --model "%ONNX%\%M2%" --combos C0 C4 --n 2 --threads 4 --out "%OUT%\smoke_equal.json"

:: 1) G1 동일성  [알림 #1]
"%PY%" -u e12\check_equal.py --model "%ONNX%\%M2%" --combos C0 C1 C2 C3 C4 --n 200 --threads 4 --out "%OUT%\equal.json"

:: 2)·3) 메모리  [알림 #2, #3]
"%PY%" -u e12\bench_mem_session.py --models "m2rot=%ONNX%\%M2%" --combos C0 C1 C2 C3 C4 --threads 4 --n 50 --repeats 3 --interval-ms 10 --out "%OUT%\memory.jsonl"
"%PY%" -u e12\bench_mem_session.py --models "fp32=%ONNX%\%FP%"  --combos C0 C1 C2 C3 C4 --threads 4 --n 50 --repeats 3 --interval-ms 10 --out "%OUT%\memory.jsonl"

:: 4)~6) 속도 세션 1·2·3  [알림 #4, #5, #6], 세션 사이 2분 휴식
"%PY%" -u e12\bench_speed_session.py --session 1 --models "m2rot=%ONNX%\%M2%" "fp32=%ONNX%\%FP%" --combos C0 C1 C2 C3 C4 --threads 1 2 4 --warmup 20 --n 200 --out "%OUT%\speed.jsonl"
"%PY%" -u e12\bench_speed_session.py --session 2 --models "m2rot=%ONNX%\%M2%" "fp32=%ONNX%\%FP%" --combos C2 C1 C4 C3 C0 --threads 1 2 4 --warmup 20 --n 200 --out "%OUT%\speed.jsonl"
"%PY%" -u e12\bench_speed_session.py --session 3 --models "m2rot=%ONNX%\%M2%" "fp32=%ONNX%\%FP%" --combos C4 C3 C2 C1 C0 --threads 1 2 4 --warmup 20 --n 200 --out "%OUT%\speed.jsonl"

:: 7) 판정·보고 후 로컬 커밋
git add e12 "results/E12-1_ORT세션메모리설정"
git commit -m "E12-1 ORT 세션 메모리 설정: 측정·판정"
```

### 12-4. 끝나면
1. `results/REPORT.md`에 11절 표 + 판정 + 원인 분석 + 결론 작성.
2. 메모리·속도·정확도 중 하나라도 C0보다 좋으면 정민에게 바로 알림.
3. 프로젝트 문서로 요약 저장 (`claude/24_E12-1_ORT세션메모리설정_결과_2026-10-2x.md`).

## 13. 결과
(실행 후 기록)
