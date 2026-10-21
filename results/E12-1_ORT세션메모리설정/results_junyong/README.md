# E12-1 재현 — 용 (연구실 데스크탑, WSL, 속도만)

정민 E12-1(PR #10, 커밋 `53f108f`)을 모델은 새로 만들지 않고, `e12/` 코드도 고치지 않은 채 다른 PC에서 다시 쟀다.

## 환경
| 항목 | 값 |
|---|---|
| CPU | Intel Core Ultra 9 285K (24코어, AVX-VNNI 있음, AVX-512 없음) |
| OS | WSL2 Ubuntu 22.04.5 (커널 6.6.114.1-microsoft-standard-WSL2), 메모리 32 GB |
| Python / ORT / numpy | 3.11 / onnxruntime **1.25.0** / 2.4.4 (`~/venvs/siglip2`) |
| 전원 | 데스크탑(상시 전원). 측정 중 다른 무거운 프로세스(DFC, 다른 측정) 없음을 `ps`로 확인 |
| 코드 | `53f108f` (dirty 아님, `env.json`의 `git_oslab_mobile`) |

## 준비 (코드 수정 없이 맞춘 것)
- SigLIP2 코드: `session_opts.S2_SIGLIP`는 `<저장소 상위>/oslab_s2/siglip2`를 찾는다. 심볼릭 링크 `~/oslab_s2 → ~/oslab_m2rot_ort`로 맞췄다. 이 체크아웃은 `1cc17c4`를 포함하고, `s2common.py`·`scripts/env_info.py`가 `1cc17c4`와 같다.
- 모델: 로컬 M2rot_ort의 sha256 `e84c3727…0f9e`가 Release `siglip2-m2rot-ort-20261019`의 SHA256SUMS와 같다. 그래서 로컬 파일을 썼다. 4단계 B16 O-AB(O1)는 `db2c1fdd…`(O1 README의 값과 같음)를 썼다.
- 캐시: `SIGLIP2_ARTIFACTS=~/siglip2_work/artifacts`. `pets_dev_x.npy`는 (990, 3, 224, 224) float32, 값 범위 [−1, 1](SigLIP2 전처리)이다. dev 5종 캐시가 모두 있다.
- 실행: [`../tools_junyong/run_e12_wsl.sh`](../tools_junyong/run_e12_wsl.sh) — README의 Windows 명령을 bash로 옮겼다. G1을 돌린 뒤 속도 3세션을 잰다. 세션별 조합 순서와 세션 사이 2분 휴식은 README와 같다. 모델은 하나만 넣었다(fp32 제외).

## 빠진 것
- **메모리(M-peak): WSL이라 속도만 쟀다.** `bench_mem_session.py`는 Windows `peak_wset` 전용이다. 이 WSL은 Windows 실행 파일 호출(interop)이 꺼져 있어서 Windows 쪽 파이썬도 쓸 수 없었다.
- **`analyze_e12_1.py`는 `memory.jsonl`이 없으면 멈춘다**(16행 FileNotFoundError). 그래서 같은 공식(같은 세션 안에서 C0 대비 p50 비율의 3세션 중앙값)으로 속도만 계산하는 [`../tools_junyong/speed_only_table.py`](../tools_junyong/speed_only_table.py)를 따로 두었다. 출력은 `speed_only_summary.json`이다.

## 결과 1 — M2rot_ort (`results_junyong/`)
G1: **통과** (C1~C4 모두 C0과 비트 동일, dev 5종 × 200장)

| 설정 | 1T p50 (ms) | 1T Δ | 2T p50 (ms) | 2T Δ | 4T p50 (ms) | 4T Δ | 4T Δ 세션별 (%) |
|---|---|---|---|---|---|---|---|
| C0 기본 | 101.05 | — | 55.36 | — | 31.97 | — | |
| C1 prepack 끔 | 102.92 | +2.2% | 58.09 | +5.2% | 33.60 | +5.4% | +5.1 / +5.4 / +6.0 |
| C2 arena 끔 | 100.52 | −0.3% | 54.92 | +1.0% | 32.40 | +1.3% | +0.2 / +2.1 / +1.3 |
| **C3 mem pattern 끔** | 98.07 | **−0.3%** | 53.95 | **−0.9%** | 31.47 | **−0.8%** | −1.6 / −0.8 / −0.6 |
| C4 전부 끔 | 103.80 | +4.9% | 58.57 | +5.9% | 33.89 | +6.4% | +6.6 / +6.4 / +4.5 |

## 결과 2 — B16 O-AB (O1, `results_junyong_b16oab/`)
G1: **통과** (C1~C4 모두 C0과 비트 동일)

| 설정 | 1T p50 (ms) | 1T Δ | 2T p50 (ms) | 2T Δ | 4T p50 (ms) | 4T Δ | 4T Δ 세션별 (%) |
|---|---|---|---|---|---|---|---|
| C0 기본 | 90.39 | — | 48.90 | — | 27.36 | — | |
| C1 prepack 끔 | 93.61 | +4.0% | 51.23 | +4.8% | 29.18 | +6.7% | +7.7 / +6.7 / +6.3 |
| C2 arena 끔 | 89.68 | −1.1% | 48.09 | −1.8% | 27.70 | +1.5% | +2.2 / +1.3 / +1.5 |
| **C3 mem pattern 끔** | 90.86 | **+0.8%** | 47.82 | **−2.4%** | 27.43 | **+0.2%** | +3.1 / −1.1 / +0.2 |
| C4 전부 끔 | 94.21 | +4.2% | 50.92 | +4.1% | 29.75 | +9.1% | +9.1 / +11.2 / +6.9 |

## 정민 결과와 비교 (속도만)
| 설정 | 정민 4T Δ (258V, Windows) | 용 4T Δ M2rot (285K, WSL) | 용 4T Δ O-AB |
|---|---|---|---|
| C1 prepack 끔 | +8.5% | +5.4% | +6.7% |
| C2 arena 끔 | **+57.1%** | **+1.3%** | +1.5% |
| C3 mem pattern 끔 | −1.6% | −0.8% | +0.2% |
| C4 전부 끔 | +62.3% | +6.4% | +9.1% |

- **C3는 같은 경향이다.** 출력이 비트 동일하고, 1·2·4T 지연이 모두 ±2.5% 안이다(판정 조건 ③ +5% 이내, ④ +10% 이내 통과). 두 모델 모두 그렇다.
- **C2(arena 끔)는 다르다.** 정민 PC(Windows)는 4T +57%였는데, 이 PC(WSL·Linux)는 +1.3%로 거의 차이가 없다. 그래서 C4의 손해도 +62% → +6%로 대부분 사라지고, 남은 손해는 prepack 끔(C1) 몫이다. 원인 가설은 arena를 끄면 매 실행마다 OS 메모리 할당기를 직접 쓰게 되는데, Windows 힙이 Linux glibc malloc보다 훨씬 느리다는 것이다(검증 안 함). 예선 심사 환경이 Windows라면 정민 결과를 따르는 것이 맞다.
- 메모리를 재지 않았으므로 C3의 M-peak −8.8%는 이 PC에서 확인하지 못했다.
- 참고: 이 PC의 C0 4T는 M2rot 31.97 ms, O-AB 27.36 ms(−14.4%)다. 다른 세션에서 잰 값이지만 O1의 B16 O-AB 4T −13.2%와 비슷하다.
