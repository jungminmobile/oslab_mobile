# E14-4 재현 — 준용 데스크톱 (Core Ultra 9 285K)

정민 노트북 결과(`../results/`, 계획서 11절)를 다른 PC에서 `e14/README_E14.md` 절차대로 재현했다. e14 코드·정민 결과 폴더는 수정하지 않았다.

## 결론
- **정확도: 정민과 비트 단위로 같다.** 고화질 top-1 O 89.67 / F2 89.71 / P2 89.88 — 데이터셋별 값·CIFAR·CI·바뀐 장 수까지 동일.
- **속도: P2가 O 대비 4T −11.8%, 1T −11.9%, 2T −10.5%** (정민 −10.3 / −11.6 / −10.2). 방향·크기 재현, 4T는 이 PC에서 조금 더 좋음.
- **메모리: P2 peak −17.3%, 세션 생성 peak −19.8%** (정민 −16.6 / −19.0). 재현.
- summary 판정도 같다: **pick = P2** (P2·P2b 채택 후보). 단 P1은 정민 PC에선 G3에서 탈락했지만 이 PC에선 G3을 통과한 뒤 summary에서 기각(아래).

## 환경
| | 이 PC (준용) | 정민 노트북 |
|---|---|---|
| CPU | Intel Core Ultra 9 285K (24C/24T, Arrow Lake) | Intel Core Ultra 7 258V (8C/8T, Lunar Lake) |
| OS | Windows 11 Pro 10.0.26200 | Windows 10.0.26300 |
| Python / ORT / onnx / numpy | 3.11.9 / 1.25.0 / 1.23.1 / 2.4.4 | 같음 |
| 전원 | 데스크톱(AC), 전원 구성표 **균형 조정**, 전원 모드 오버레이 없음 → **최고 성능 아님** | 전원 연결 |
| 스크립트 커밋 | `e1f8428` (`exp/E14-junyong`, clean) | `eeabc2e` (dirty) |
| 경로 | `E14_ART=C:\e14data`, `E14_WORK=C:\e14data\e14`, `E14_OUT=C:\e14data\results_junyong` (D 드라이브 없음) | |

- 데이터 sha256: `e14/E14_DATA_SHA256.txt`의 20개 파일 **모두 일치**, 목록 밖 파일 없음.
- `make_models_e14.py`: O `7a01044c` OK, F2 `03f4de18` · P1 `450db00f` · P2 `e465d10d` · P2b `6df4ac09` **모두 원본과 같음**.
- 측정 중 다른 부하: 각 단계 전 WSL load1 = 0.00, Windows CPU 1~5% (`run_all_driver.log`).
- 다른 PC 대체 경로: `env_source: EnvLite`, 지연 입력 `input_src: dev256_fallback`(README_E14대로), step 0 ⑥(나)는 `oslab_submit` 없어 zoo_pre 사본 사용.
- `pip install -r requirements_e14.txt`가 한글 주석 때문에 cp949 디코드 오류(`UnicodeDecodeError: 'cp949' codec ...`) → `PYTHONUTF8=1`로 설치(파일 수정 없음).
  **다음 재현자 메모** (README_E14는 e14 코드라 고치지 않고 여기만 적음): 한국어 Windows에서는 설치 전에 UTF-8 모드를 켠다.
  ```bat
  set PYTHONUTF8=1
  e14venv\Scripts\pip install -r e14\requirements_e14.txt
  ```
  PowerShell이면 `$env:PYTHONUTF8='1'`.

## 정확도 (dev 전체, 고화질 = pets·flowers·cars·food·caltech 평균)
| 모델 | 고화질 top-1 | 정민 | 같은가 |
|---|---|---|---|
| O | 89.67 (89.6689) | 89.67 | 데이터셋별 값까지 동일 (E14-2 `acc.json`의 O와 비교) |
| F2 | 89.71 (89.7096) | 89.71 | 동일 |
| P1 | 89.61 (F2 대비 −0.10%p, CI [−0.39, +0.20]) | (G3 탈락으로 미측정) | — |
| P2 / P2b | 89.88 (89.8748, F2 대비 +0.17%p) | 89.88 | 동일 (`acc.json`에서 date·git·P1 빼면 전부 같음) |

O는 E14-4 단계에 정확도 측정이 없어 `tools/acc_O.py`(step_acc와 같은 `R1.session/embed/l2n` 경로)로 따로 재서 `acc_O.json`에 남겼다. CPU 세대가 달라도(Arrow Lake vs Lunar Lake) 같은 ORT 1.25.0 x86 커널이 같은 결과를 냈다.

## 속도 (p50, 3세션 중앙값; O 대비 %는 정민과 같은 방식 = 같은 세션 안 비율의 중앙값)
| 모델 | 1T | 2T | 4T |
|---|---|---|---|
| O | 31.22 ms | 19.14 ms | 13.06 ms |
| F2 | 29.01 ms | 17.80 ms | 12.02 ms |
| P1 | 28.93 ms | 17.92 ms | 11.95 ms |
| P2 | 27.52 ms | 17.14 ms | 11.51 ms |
| **P2 vs O (이 PC)** | **−11.9%** | **−10.5%** | **−11.8%** |
| P2 vs O (정민) | −11.6% | −10.2% | −10.3% |

절댓값은 이 PC가 빠르다(O 4T 13.06 vs 16.36 ms). 세션 간 편차 작음(P2 4T 11.50~11.56 ms).

## 메모리 (C3, 4T, 50회, 3회 중앙값)
| 모델 | peak | O 대비 | 세션 생성 peak | O 대비 |
|---|---|---|---|---|
| O | 192.4 MiB | — | 192.4 MiB | — |
| F2 | 191.6 | −0.4% | 191.6 | −0.4% |
| P1 | 167.6 | −12.9% | 165.3 | −14.1% |
| **P2** | **159.2** | **−17.3%** | **154.4** | **−19.8%** |
| P2b | 159.9 | −16.9% | 155.4 | −19.2% |
| 정민 P2 | 161.2 (O 193.2) | −16.6% | 156.4 | −19.0% |

## 정민과 다른 점: P1
- G3(빠른 4T 속도, F2 대비): 정민 +3.7% / +0.2% → 탈락, 이 PC −0.2%(+0.17%) / −2.8% → 통과. P1(FP32 패치 MatMul)은 F2와 속도 차이가 거의 없어 게이트 경계에 걸려 있고, CPU에 따라 갈린다.
- 그래서 acc·mem·speed에 P1이 더 들어갔고, summary에서 **기각**(4T 조건 F2 대비 −0.7%로 미달, 정확도 조건 미달). 최종 선택(P2)에는 영향 없음.

## 파일
`summary.json` 등은 `run_e14_4.py` 산출물 그대로 (`C:\e14data\results_junyong`에서 복사, 13 MB `g2_emb.npz` 제외).
`acc_O.json` — O 정확도 추가 측정. `run_all_driver.log` — 단계 순서·부하 확인·전원 구성표 기록.
`tools/` — `e14env.ps1`(환경변수), `run_all.ps1`(step 0 → summary, 세션 사이 120 s, 단계 전 유휴 확인), `acc_O.py`.
실행: 2026-10-27 15:42 ~ 15:59.
