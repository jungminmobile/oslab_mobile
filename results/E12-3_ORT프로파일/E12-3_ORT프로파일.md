# E12-3 ORT 프로파일 (제출 모델 B/32 O-AB) — 실험계획서

- 작성: 2026-10-23, 정민 (계획: Claude 세션 A / 실행: 별도 채팅방)
- 시리즈: E12 (ONNX 런타임·그래프 최적화) 중 3번째 실행 (원래 목록의 "보류 2")
- 상태: **실행 완료 (10/23 00:41)** — 확정본 (10/23 리뷰 1차 필수 수정 5개 + 이종 코어 확인, 2차 필수 수정 A·B + 권장 5개 반영). 결과 후 판정 규칙 변경은 정민 승인 필요
- 폴더: `results/E12-3_ORT프로파일/` (계획서 = 이 파일, 결과 = `results/` 하위)

## 1. 목적
최종 제출 구성(**B/32 O-AB + 세션 설정 C3**)의 예선(ORT CPU) 지연이 **어느 연산에서 나는지** 잰다. 원본(B/32 FP32)과 같은 방법으로 재서 최적화로 무엇이 줄었고 무엇이 남았는지 본다. 모델·설정은 바꾸지 않는다.
결과로 다음 실험을 고른다: **E12-4 스레드·spinning** / **E12-5 Add 등 정수화** / 둘 다 안 함.

## 2. 배경
- 준혁 B/16 M2rot_ort 프로파일 (`820890b`, i7-1165G7, 10/20): 4T 기준 정수 행렬곱 50.0%, **Q/DQ 12.8%**, Add·Mul·Sub 10.0%(M2 마스크 4~5%), Softmax 7.5%, GELU 6.2%, LayerNorm 3.5%. 결론 "Q/DQ를 다 없애도 13%, 기준 15% 미달 → 예선 모델 안 바꿈". (그 표의 분모는 커널 합이다. 이번 실험은 분모가 다르므로 경향만 비교한다.)
- 그 뒤 바뀐 것: 제출 모델이 **B/32**(토큰 49개, 입력 256)이고 **O-AB**(MatMul 출력 Q/DQ 제거 → `MatMulIntegerToFloat`, SkipLayerNorm 24개)다.
- 토큰이 4배 적어 행렬곱 비중이 줄고, 연산 개수에 비례하는 고정비(작은 원소별 연산, 스레드 동기화)의 비중이 커질 것으로 예상한다 (가설).
- 258V(Lunar Lake)는 **P코어 4개 + LP E코어 4개**인 이종 코어다. 스레드 효율이 낮게 나오면 동기화가 아니라 E코어 배정 때문일 수 있다.

## 3. 모델 (2개)
| ID | 구성 | 파일 | sha256 | 받는 곳 |
|---|---|---|---|---|
| **S (최종)** | B/32 O-AB (회전 + M2 + O-B + O-A INT8) + **C3** | `siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx` (102,763,369 B) | `7a01044c6ba32bf4f5a12aba82eb19e06753f13c8bad43b9f6fcfcd9f484dbf6` | Release `siglip2-oab-20261022` |
| **F (원본)** | B/32 FP32 + **C0 (ORT 기본)** | `siglip2_b32_image_fp32.onnx` (378,356,721 B) | **0단계에서 Release `SHA256SUMS.txt`의 값을 이 칸에 적어 고정** | Release `siglip2-b32-candidate-20261020` |

- 판단은 **S**로 한다. F는 "최적화로 무엇이 줄었는지"를 보여 주는 기준.
- **S−F 차이는 순수한 모델 최적화 효과가 아니다.** 세션 설정(S = C3, F = C0)이 달라 `enable_mem_pattern` 효과가 섞인다. 판정에는 쓰지 않고 표 각주로 적는다.
- 입력: 1×3×256×256 float32, 전처리 x/127.5 − 1. 프로파일은 시간만 보므로 **실제 사진 1장(pets dev)을 256으로 맞춰 반복 입력**한다 (불가하면 고정 시드 난수, 그 사실을 기록).

## 4. 측정 방법
| 항목 | 값 |
|---|---|
| 스레드 | 1, 2, 4 |
| 세션 설정 | S = C3 (`enable_mem_pattern=False`), F = C0 (기본) |
| 반복 | 세션 하나에서 **120장** 실행. 집계할 때 **앞 20장의 `model_run`과 그 시간 구간에 든 노드 이벤트, `session_initialization` 이벤트를 버린다** (ORT Python은 세션 생성 때만 프로파일을 켤 수 있음). 이것을 **3세션** (세션마다 모델·스레드 순서 바꿈, 사이 2분 휴식) |
| 그래프 최적화 | `ORT_ENABLE_ALL` (제출과 같음) |
| 프로파일 | `SessionOptions.enable_profiling = True`, `profile_file_prefix`로 파일 분리. **각 `sess.run`을 `time.perf_counter`로 감싸 벽시계 시간도 같이 기록** |
| 무프로파일 기준 | 같은 모델·스레드·설정으로 프로파일 없이 120장(앞 20장 버림), **`perf_counter`** p50. **오버헤드 보정에만** 쓴다 |
| 코어 고정 확인 (2b) | S에서 **P코어 고정 vs 고정 없음**. 고정은 두 겹으로 한다: ① **프로세스 전체를 P코어 4개로 제한**(`psutil.Process().cpu_affinity([P코어 0-based 번호])`, 메인 스레드 포함) ② ORT 작업 스레드 3개에 `intra_op_thread_affinities`(ORT 번호 체계). 4T 1회씩 + **P코어 고정 1T 1회**(고정 상태 스레드 효율 계산용) |
| P코어 번호 | 0단계에서 `EfficiencyClass`로 찾고 **0-based 번호와 ORT용 번호(1부터 세는 방식)**를 `env.json`에 둘 다 기록. **E코어로 일부러 고정해 확실히 느려지는지 1회 확인** (또는 작업 관리자로 확인) |

### 집계 정의
- 장당 `model_run` 시간 = 프로파일의 `model_run` 이벤트 `dur` (워밍업 제외 100장).
- 장당 커널 합 = 그 `model_run` 시간 구간 안의 노드 `kernel_time` 이벤트 `dur` 합.
- **커널 밖 = model_run − 커널 합** (같은 프로파일 파일 안에서 계산).
- **오버헤드 = (프로파일 켠 실행의 `perf_counter` p50) − (무프로파일 `perf_counter` p50).** 두 값 모두 `perf_counter` 기준.
- **커널 밖(보정) = 커널 밖 − 오버헤드.**
- **모든 % 값의 분모 = 장당 `model_run` 시간** (커널 밖 포함).

### 집계 지표
| 지표 | 정의 | 무엇을 알려 주나 |
|---|---|---|
| **연산 묶음별 비율·시간** | 묶음별 커널 시간 ÷ model_run (%)와 장당 ms. 묶음은 준혁 `s2_profile_ort.py`의 `group()`을 그대로 쓰고, `MatMulIntegerToFloat`·`SkipLayerNormalization`만 추가 분류 | 어디가 무거운가 |
| **S vs F 묶음별 ms 변화** | 같은 묶음의 장당 ms 차이 (세션 설정 차이 포함, 각주) | 최적화로 줄어든 곳 / 남은 곳 |
| **연산 묶음별 스레드 효율** | (1T 시간) ÷ (4T 시간 × 4). 고정 없음은 2단계 1T·4T, P코어 고정은 2b의 고정 1T·4T로 계산 | 4코어를 잘 쓰는가 |
| **커널 밖(보정)** | 위 정의 | 연산 사이 대기·스레드 깨우기 비용 → E12-4 근거 |
| 노드 호출 수 | 묶음별 장당 노드 수 | 작은 연산이 많은지 |
| 상위 20개 노드 | 장당 시간 순 | 특정 블록에 몰리는지, **형태 연산(Transpose·Reshape)** 비중 확인 |

## 5. 다음 실험 결정 규칙 (사전 등록)
S, 4T, 3세션 중앙값. 모든 %의 분모 = 장당 model_run 시간(워밍업 20장 제외).
오버헤드 = (프로파일 켠 실행 perf_counter p50) − (무프로파일 perf_counter p50).
커널 밖(보정) = (model_run − Σkernel) − 오버헤드.
"정수화 가능" = ORT 1.25 CPU EP에 정수 커널이 있는 연산만 (0단계에서 목록 확인·기록. 없는 연산은 표에만 적고 규칙 계산에서 뺀다).

| 우선 | 조건 | 결정 |
|---|---|---|
| 1 | 커널 밖(보정) ≥ 15% 또는 스레드 효율 < 0.5인 묶음이 ≥ 20%, **그리고 P코어 고정(2b)에서도 같은 조건(커널 밖(보정) ≥ 15% 또는 해당 묶음 ≥ 20%)이 성립** | **E12-4** |
| 2 | Q/DQ + 정수화 가능 원소별 연산 ≥ 20% | **E12-5 검토** (정확도 위험이 커서 계획서 따로) |
| 3 | 정수 행렬곱(`QLinearMatMul`·`MatMulIntegerToFloat`) ≥ 70% 이고 1·2 불성립 | **보류**, 예선 지연 확정 |

- 1과 다른 조건이 함께 성립하면 1을 먼저 한다.
- 2b는 1회 측정이다. 2b 값이 기준(15% / 20%)의 **±2%p 이내**면 2b를 1회 더 재서 두 값의 평균으로 판단한다.
- 조건 1이 **P코어 고정에서 사라지면**, 원인은 이 노트북의 코어 배정이다. 코어 번호는 PC마다 달라 제출 설정으로 옮길 수 없으므로 E12-4로 가지 않고 **"측정 방법 개선(측정 시 코어 배정 기록)"**으로 기록한다.
- E12-4를 진행하면 **258V 외 CPU 1대에서 교차 확인**한다 (예선 채점 PC 사양 미상, 본선 RPi5는 같은 종류 코어 A76 ×4).
- 15%·20%·70% 기준은 준혁 B/16 프로파일의 "15% 미만이면 모델 안 바꿈" 기준을 따랐다.
- 결과로 모델이나 제출 설정을 바꾸지 않는다. 이 실험은 **다음 실험 선택만** 한다.

## 6. 실험 순서
| # | 단계 | 예상 |
|---|---|---|
| 0 | 다른 측정이 도는지 확인 · 결과 폴더 만들기 · 두 모델 받기, sha 비교(다르면 멈추고 정민에게 보고) · F sha를 3절에 기록 · **ORT 1.25 CPU EP 정수 커널 목록**(`QLinearAdd`·`QLinearMul` 등) 확인 · **P코어 번호 확인(0-based / ORT용) + E코어 고정 시 느려지는지 확인** · `env.json`(OPENBLAS_NUM_THREADS 포함) | 20분 |
| 1 | 스크립트 작성 (`e12/profile_ort.py`: 준혁 `s2_profile_ort.py` 기반 + 입력 256, 모델별 C0/C3, 120장·앞 20장 제외, `perf_counter` 벽시계, 무프로파일 p50, 프로세스·ORT 이중 코어 고정 옵션 / `e12/profile_summary.py`: 5절 정의대로 집계), 2장 시험 | 25분 |
| 2 | 프로파일 3세션 (S·F × 1·2·4T + 무프로파일 p50) | 약 25분 |
| 2b | S: 고정 없음 4T, P코어 고정 4T, P코어 고정 1T 각 1회 (기준 ±2%p 이내면 1회 더) | 약 5분 |
| 3 | 집계·결정·REPORT·로컬 커밋 | 25분 |
| | **합계** | **약 1시간 40분** |

## 7. 위험과 대응
| 위험 | 대응 |
|---|---|
| 프로파일 오버헤드가 커널 밖에 쌓여 규칙 1이 쉽게 성립 | 보정값(5절)으로 판정 |
| 오버헤드를 서로 다른 기준(model_run dur vs 벽시계)으로 빼서 작게·음수로 나옴 | 두 p50 모두 `perf_counter` 기준 |
| 서로 다른 실행을 빼서 커널 밖이 음수·0 근처 | 커널 밖은 같은 프로파일 파일 안에서 계산 |
| 워밍업이 프로파일에 섞임 | 120장 중 앞 20장 이벤트 제외 |
| E코어 배정을 동기화 문제로 오해 | 2b 코어 고정 재측정, 규칙 1 단서 |
| 메인 스레드가 고정되지 않아 고정이 절반만 됨 | 프로세스 전체 P코어 제한 + ORT 작업 스레드 고정 |
| 코어 번호 체계 혼동으로 E코어에 고정 | 0-based / ORT용 번호 둘 다 기록, E코어 고정 시 느려지는지 확인 |
| 노트북 drift (R0 52 → 81 ms 사례) | 3세션 중앙값, 전원 연결·절전 끄기, 세션 사이 휴식 |
| `s2_profile_ort.py`가 224 입력·pets 캐시 전제 | 새 스크립트에서 입력 크기를 인자로 받음. 원본 코드는 수정하지 않음 |
| 기기 차이 (준혁 i7-1165G7 vs 정민 258V), 분모 차이 | B/16과는 "경향"만 비교. 수치 비교는 같은 PC의 S·F끼리만 |
| O-AB 이전 INT8을 빼서 "O-AB가 없앤 몫"은 안 보임 | 결정에는 필요 없음. 필요하면 Release `siglip2-oab-20261022` 기록 인용 |

## 8. 실행 지침 (실행 담당 채팅방용)
- 쉬운 한국어, 간결하게. 실행 전 계획·시간 보고, 약 25%마다 진행 보고. 실제로 실행한 것만 보고.
- `rn50\.venv`, `oslab_s2` 수정 금지. 커밋은 로컬만, push 금지.
- 노트북에서 다른 측정이 돌고 있으면 시작하지 않는다.
- 관문(sha 불일치 등)에 걸리면 명령 창을 닫지 말고 **멈춘 뒤 정민에게 보고**한다.
- 알림: 2단계(프로파일 3세션) 시작 3분 전 1회만 ("약 25분, 무거운 작업 닫아 주세요").
- 환경: 정민 노트북(Core Ultra 7 258V), Windows, `s2_venv`(Python 3.11.9, ORT 1.25.0). 브랜치 `jungmin/e12-runtime` 이어서.

```bat
cd /d "C:\study\3\2\2026 모바일 학술대회\oslab_mobile"
set "PY=C:\study\3\2\2026 모바일 학술대회\s2_venv\Scripts\python.exe"
set "OUT=C:\study\3\2\2026 모바일 학술대회\oslab_mobile\results\E12-3_ORT프로파일\results"
set "ONNX=C:\s2art\onnx"
set "S=siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx"
set "F=siglip2_b32_image_fp32.onnx"
set "S_SHA=7a01044c6ba32bf4f5a12aba82eb19e06753f13c8bad43b9f6fcfcd9f484dbf6"
set OPENBLAS_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
if not exist "%OUT%" mkdir "%OUT%"

:: 0) 모델 받기 (없을 때만) + sha 비교 (다르면 멈추고 보고)
if not exist "%ONNX%\%S%" gh release download siglip2-oab-20261022 -p "%S%" -D "%ONNX%"
if not exist "%ONNX%\%F%" gh release download siglip2-b32-candidate-20261020 -p "%F%" -D "%ONNX%"
gh release download siglip2-b32-candidate-20261020 -p SHA256SUMS.txt -O "%OUT%\b32_SHA256SUMS.txt" --clobber
certutil -hashfile "%ONNX%\%S%" SHA256 | findstr /i "%S_SHA%" || echo [중단 필요] S sha 불일치 - 다음 단계로 가지 말고 보고
certutil -hashfile "%ONNX%\%F%" SHA256
type "%OUT%\b32_SHA256SUMS.txt"
::    → F 값이 SHA256SUMS와 다르면 멈추고 보고. 같으면 3절 표에 해시 기록
:: 0-2) 정수 커널 목록·P코어 번호(0-based / ORT용)·E코어 고정 확인·환경 기록
"%PY%" -u e12\check_env.py --out "%OUT%\env.json"
::    (ORT CPU EP 등록 연산 중 QLinear*/Q* 목록, EfficiencyClass별 논리 프로세서 번호 0-based·ORT용,
::     E코어 고정 1회 vs P코어 고정 1회 짧은 지연 비교, OPENBLAS_NUM_THREADS 기록)

:: 1) 시험 (2장)
"%PY%" -u e12\profile_ort.py --models "S=%ONNX%\%S%:C3" --threads 4 --img 256 --n 4 --discard 2 --session 0 --out "%OUT%\smoke.jsonl"

:: 2) 프로파일 3세션 [알림 1회], 사이 2분
"%PY%" -u e12\profile_ort.py --models "S=%ONNX%\%S%:C3" "F=%ONNX%\%F%:C0" --threads 1 2 4 --img 256 --n 120 --discard 20 --noprof-baseline --session 1 --out "%OUT%\profile.jsonl"
timeout /t 120 /nobreak > nul
"%PY%" -u e12\profile_ort.py --models "F=%ONNX%\%F%:C0" "S=%ONNX%\%S%:C3" --threads 1 2 4 --img 256 --n 120 --discard 20 --noprof-baseline --session 2 --out "%OUT%\profile.jsonl"
timeout /t 120 /nobreak > nul
"%PY%" -u e12\profile_ort.py --models "S=%ONNX%\%S%:C3" "F=%ONNX%\%F%:C0" --threads 4 2 1 --img 256 --n 120 --discard 20 --noprof-baseline --session 3 --out "%OUT%\profile.jsonl"

:: 2b) 코어 고정 확인 (S): 고정 없음 4T / P코어 고정 4T / P코어 고정 1T
::     --affinity-pcores = 프로세스 전체 P코어 제한(psutil cpu_affinity, 0-based) + ORT 작업 스레드 3개 고정(ORT용 번호)
"%PY%" -u e12\profile_ort.py --models "S=%ONNX%\%S%:C3" --threads 4 --img 256 --n 120 --discard 20 --noprof-baseline --session affinity_none --out "%OUT%\affinity.jsonl"
"%PY%" -u e12\profile_ort.py --models "S=%ONNX%\%S%:C3" --threads 4 --img 256 --n 120 --discard 20 --noprof-baseline --affinity-pcores --env "%OUT%\env.json" --session affinity_p4 --out "%OUT%\affinity.jsonl"
"%PY%" -u e12\profile_ort.py --models "S=%ONNX%\%S%:C3" --threads 1 --img 256 --n 120 --discard 20 --noprof-baseline --affinity-pcores --env "%OUT%\env.json" --session affinity_p1 --out "%OUT%\affinity.jsonl"
::     기준(15% / 20%)의 ±2%p 이내면 affinity_p4를 1회 더 (--session affinity_p4b)

:: 3) 집계 후 로컬 커밋
"%PY%" -u e12\profile_summary.py --in "%OUT%\profile.jsonl" --affinity "%OUT%\affinity.jsonl" --env "%OUT%\env.json" --out "%OUT%\summary.json"
git add e12 "results/E12-3_ORT프로파일"
git commit -m "E12-3 ORT 프로파일 (B/32 O-AB): 연산별 시간·다음 실험 결정"
```
- 스크립트 이름과 인자 이름은 예시다. 실제로 쓴 명령을 `run_log.txt`에 기록한다.

## 9. 보고 형식
분모 = 장당 model_run 시간 (S·F 각자). ms = 장당 시간.

| 연산 묶음 | S 1T (ms) | S 4T (%) | S 4T (ms) | S 4T 스레드 효율 | F 4T (ms) | S − F (ms)¹ | 정수화 가능² | B/16 M2rot 4T (준혁, 참고³) |
|---|---|---|---|---|---|---|---|---|
| 행렬곱 (S 정수 / F float) | | | | | | | — | 50.0% |
| Q/DQ | | | | | — | | — | 12.8% |
| Add·Mul·Sub | | | | | | | | 10.0% |
| Softmax | | | | | | | | 7.5% |
| GELU | | | | | | | | 6.2% |
| LayerNorm / SkipLN | | | | | | | | 3.5% |
| 형태 연산 (Transpose·Reshape 등) | | | | | | | — | — |
| **커널 밖 (보정 전 / 보정 후)** | | | | — | | | — | — |
| **model_run 합계** | | 100% | | | | | | |
| 프로파일 p50 / 무프로파일 p50 (**둘 다 perf_counter 기준**) | | | | | | | | |

¹ S−F에는 세션 설정 차이(C3 vs C0)가 섞여 있다. 판정에 쓰지 않는다.
² 0단계 목록 기준. "없음"이면 규칙 2 계산에서 뺀다.
³ 준혁 표는 분모가 커널 합이라 직접 비교하지 않는다.

코어 고정 확인 (S): 고정 없음 4T / P코어 고정 4T의 model_run p50, 커널 밖(보정), 스레드 효율 < 0.5 묶음 비율(고정 효율은 고정 1T·4T로 계산). P코어 번호(0-based / ORT용)와 E코어 고정 확인 결과.

결정: (E12-4 / E12-5 / 보류 / 측정 방법 개선) + 근거 한 줄.

## 10. 결과
2026-10-23 00:24~00:41 실행. 상세: `REPORT.md`.

- **결정: E12-4 스레드·spinning 진행.** 규칙 1b(스레드 효율 < 0.5 묶음 = model_run의 43.0%, P코어 고정 45.8%)가 성립. 규칙 2(Q/DQ+Add+Mul 26.9%)도 성립하지만 우선순위대로 E12-4 먼저. 규칙 3(정수 행렬곱 42.5%) 불성립.
- 규칙 1a(커널 밖 보정)는 판단 불가: 프로파일 오버헤드(4T 6.6 ms)가 커널 안에 쌓여 보정값이 음수(−21.7%).
- 사후 민감도(판정에 안 씀): 노드당 오버헤드를 빼도 1b = 27.0%로 결론 같음. 규칙 2는 18.0%로 기준 아래 → E12-5 근거 약함.
- 무프로파일 1T→4T 효율: S 0.60, F 0.78. P코어 고정 효과 없음(16.43 → 16.60 ms).
