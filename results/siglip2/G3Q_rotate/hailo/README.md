# G3Q-H: 회전 모델(rothad)의 Hailo HEF — 에뮬레이터·Hailo-10H 실칩에서 D256L1과 비교 (설계, 2026-10-18)

**1~4절(설계·기준)은 결과를 보기 전에 고정한다.** 결과는 5절 이후에 덧붙인다. **채택은 팀 결정이고, 여기서는 결과 표만 만든다.**

## 1. 질문
ORT에서 회전(rothad)은 cos를 올렸지만 top-1은 그대로였다 (상위 README 8절). Hailo DFC는 보정·눈금 방식이 달라서(LN 분해, 바이어스 보정, 16비트 층 혼합) 같은 회전 ONNX가 다른 효과를 낼 수 있다. 같은 `.alls`(D256L1)로 rothad HEF를 만들어 팀 구성과 비교한다.

## 2. 비교 구성 (같은 PC·같은 DFC로 만든 것끼리)
| 이름 | 입력 ONNX (b1) | `.alls` | 보정 | 
|---|---|---|---|
| `D256L1` (이 PC에서 다시 만든 것) | 원래 FP32 b1 (Release `siglip2-hailo-crosscheck-20261017`의 `siglip2_image_fp32_b1.onnx`) | `siglip2/hailo/g4_D256L1_official_b9fc2.alls` | `calib_highres256_s0_nhwc.npy` (`s2_dfc_calib.py --n 256`) |
| `rothad_D256L1` | `siglip2_image_fp32_rothad_b1.onnx` (sha256 `7a07bdee…`, G3a 파싱 통과) | 같은 파일 그대로 | 같은 파일 |

- **팀 HEF(쭈리 PC, sha256 `b33a266f…`)는 이 PC와 GitHub Release 어디에도 없다** (10/18 확인). 그래서 이 PC에서 같은 `.alls`·같은 보정으로 D256L1을 다시 만들어 비교 기준으로 쓴다 (ORT 비교에서 R0를 이 PC 파일로 한 것과 같은 원칙). HEF sha256이 `b33a266f…`와 다를 수 있다 (G4_crosscheck: 정민 노트북 HEF도 `e0be125a…`로 달랐다). 다르면 기록만 한다.
- 명령은 팀 기록(G4_dfc_screen 6-3절, G4_compile)과 같다: `hailo parser onnx <b1> --hw-arch hailo10h` → `hailo optimize <har> --hw-arch hailo10h --model-script g4_D256L1_official_b9fc2.alls --calib-set-path calib_highres256_s0_nhwc.npy` → `s2_dfc_screen.py --k 200 --bs 8` → `hailo compiler <opt.har> --hw-arch hailo10h` → `hailo profiler`.
- 층 이름 확인 (optimize 전): 두 HAR의 층 ↔ ONNX 노드 표(`s2_har_layers.py`)에서 `.alls`가 가리키는 층(`conv1`, `conv4`, `conv5`, `conv24`, `conv25`, `conv41`, `ew_add*`)이 원래 팀 표(`siglip2/hailo/layer_map_siglip2.csv`)와 같은 ONNX 노드를 가리켜야 한다. 다르면 **멈추고 보고**.
- 환경: 이 PC (WSL Ubuntu-22.04, Ultra 9 285K), DFC 5.4.0 `~/hailo10_venv`, `CUDA_VISIBLE_DEVICES=-1`, 저장소 밖 `~/siglip2_work/hailo_g3q/`에서 실행. optimize·에뮬레이터·compile은 `flock ~/siglip2_work/dfc.lock` 안에서.
- 실칩: AGX Orin + Hailo-10H (`pci/0005:01:00.0`), HailoRT 5.3.0 (드라이버 `hailo1x_pci` 5.3.0). DFC 5.4.0 HEF가 돈다 (6절 1단계).

## 3. 판정 기준 (결과 전에 고정)
| ID | 무엇 | 방법 | 기준 |
|---|---|---|---|
| **H1** DFC 정확도 (에뮬레이터, SDK_QUANTIZED) | rothad_D256L1 vs D256L1 | `s2_dfc_screen.py --k 200 --bs 8`, dev 6종 × 앞 200장, FP32 ONNX 임베딩 대비 cos | rothad_D256L1 고화질 cos > **같은 PC에서 잰 D256L1** 고화질 cos 이면 **"개선 후보"**. 아니면 "개선 없음" |
| H2 실칩 정확도 | 두 HEF, AGX Hailo-10H | 같은 dev 6×200 사진, 사진별 임베딩 저장 → 데스크탑에서 top-1(클래스 임베딩 = dev 캐시 `clf`)·FP32 대비 cos | 기록 (판정 없음). 짝 차이와 95% 구간(층화 짝 부트스트랩 1000회, 시드 0)을 낸다 |
| H3 실칩 지연 | 두 HEF | 배치 1, 워밍업 후 반복, 두 HEF 번갈아 3회. `hailortcli monitor` 끔, `OPENBLAS_NUM_THREADS=1` | 기록. 그래프 구조가 같으니 같을 것으로 예상하고, 차이가 3회 중앙값 기준 5% 넘으면 원인(컨텍스트 수, 16비트 층 배치 등)을 기록 |
| (기록) 에뮬레이터 ↔ 실칩 | 같은 HEF의 실칩 결과 vs 같은 HAR의 에뮬레이터 결과 | 사진별 예측 일치율, cos 차이 | 기록 |

- **H1 기준값에 대한 주의**: 요청에 적힌 0.952는 구성 D(보정 64장, 최적화 수준 0, G4_compile 1절)의 값이다. 같은 `.alls`(보정 256장, 수준 1)인 D256L1의 팀 기록은 **0.9665**다 (G4_dfc_screen 6-3절, 쭈리 PC). H1은 같은 PC끼리 비교하고, 이 PC의 D256L1 값이 0.9665와 다르면 그 차이도 기록한다.
- 크기·컨텍스트 수는 기록한다 (팀 D256L1: 99.8 MB, 17 컨텍스트).

## 4. 실칩 측정 방법 (3단계 전에 확정, 연구실 기존 코드 확인 후 6절에 적는다)
- 입력: AGX `~/ssd/junyong/siglip2_xplat/dev200/` (dev 캐시 앞 200장과 같은지 확인). HEF 입력 형식(NHWC, 정규화가 HEF 안인지 밖인지, 입력 양자화)은 `s2_dfc_screen.py`·`s2_dfc_calib.py`와 같게: 정규화된 float32 NHWC를 HailoRT가 양자화.
- AGX는 연구실 공용이다. Hailo를 쓰기 전에 `ps`로 다른 Hailo 프로세스가 없는지 확인한다. 쓰는 중이면 기다리고 용에게 알린다. nohup + 로그, `~/ssd/junyong/` 아래만, sudo·기기 설정 변경·HailoRT 업그레이드 금지.

## 6. 진행 기록
### 1단계 버전 호환 (10/18 09:0x) — 통과
- AGX: `hailortcli --version` = 5.3.0, 패키지 `hailort`·`hailort-pcie-driver` 5.3.0, `hailortcli scan` → `pci/0005:01:00.0` (Hailo-10H, `lspci` Device 45c4). 측정 시점 다른 Hailo 프로세스 없음.
- 팀 D256L1 HEF가 없어서, 이 PC에서 DFC 5.4.0으로 만든 다른 HEF(텍스트 블록 `siglip2_text_blocks_b1.hef`, sha256 `9849e124…`, 11 컨텍스트)로 확인: `hailortcli parse-hef` 정상(`HEF Compiler Version: 5.4.0`, HAILO15H·HAILO10H 호환), `hailortcli run`은 "이 장치에서 지원 안 함, run2를 쓰라"는 오류, **`hailortcli run2 -t 5 set-net <hef>` 정상 (81.7 FPS)**. → **DFC 5.4.0 HEF ↔ HailoRT 5.3.0 실행 가능.** 이미지 HEF로도 3단계에서 다시 확인한다.

### 2단계 parse 결과 (10/18 09:0x)
- 원래 b1(Release `ce2b66b9…`) → HAR 200층, 층 ↔ ONNX 노드 표가 팀 표 `layer_map_siglip2.csv`와 **200층 모두 같다**. rothad b1 → HAR **175층**.
- 차이는 `normalization` 층 25개뿐이다: trunk LN 25개(norm1 ×12, norm2 ×12, trunk.norm)의 γ·β 곱셈·덧셈 층. rothad는 γ=1·β=0이라 DFC가 없앴다. `.alls`가 가리키는 층(`conv1`, `conv4`, `conv5`, `conv24`, `conv25`, `conv41`, `ew_add` 26개)과 나머지 174층은 이름·ONNX 노드가 같다. 이름이 같은데 다른 노드를 가리키는 것은 `normalization1` 하나다 (D: 블록 0 norm1, rothad: 풀링 헤드 norm이 번호를 이어받음, `.alls`와 무관).
- → `.alls`를 그대로 쓴다. 단, rothad는 양자화되는 층이 25개 적다 (H3 지연에 영향 가능).
- 파일: `layers_siglip2_b16_224.csv`, `layers_siglip2_rothad_b16_224.csv`, `parse_*.log`, `parsed_har_sha256.txt`.

### 2단계 optimize·H1 (10/18, 커밋 `cb0e41f`·`bb9e935`, dirty 0, `run_info.txt`·`screen.jsonl`)
- optimize (최적화 수준 1, 보정 256장, 바이어스 보정 79블록): rothad 28분 27초, D256L1 28분 7초, 모두 rc=0. HAR sha256 rothad `a51af7af…`, D256L1 `fa32862d…` (`har_sha256.txt`).
- 같은 시간대에 G3R 세션의 optimize(`rcmaskbc`)가 잠금 없이 같이 돌았다 (메모리 여유 13 GB, 문제 없음). 스크리닝·compile은 잠금 + 여유 ≥ 15 GB 확인 후 시작.

| 구성 (에뮬레이터 SDK_QUANTIZED, dev 6×200) | CIFAR A | Pets | Flowers | Cars | Food | Caltech | 고화질 Δtop-1 | **고화질 cos** |
|---|---|---|---|---|---|---|---|---|
| FP32 | 74.5 | 93.5 | 85.0 | 93.5 | 87.5 | 95.0 | 기준 | 1 |
| D256L1 (이 PC) | 65.0 | 92.0 | 84.5 | 93.5 | 86.5 | 95.5 | −0.5 | 0.9666 |
| **rothad_D256L1** | **71.0** | 93.0 | 87.0 | 93.5 | 88.0 | 95.0 | +0.4 | **0.9775** |
| (참고) D256L1 팀 기록 (쭈리 PC, G4_dfc_screen 6-3절) | 66.0 | 92.0 | 85.0 | 93.0 | 86.5 | 96.0 | −0.4 | 0.9665 |

- **H1: rothad 0.9775 > D256L1 0.9666 → "개선 후보"** (3절 기준). 데이터셋별 cos는 6종 모두 rothad가 높다 (+0.008 ~ +0.037, CIFAR A가 가장 큼).
- 짝 비교 (`h1_paired.json`, 고화질 5종): top-1 차이 rothad − D256L1 = **+0.9 [−0.2, +2.1]%p** (구간에 0 포함, 200장), cos 차이 +0.0109, 사진의 89.3%에서 cos가 오름, 예측 일치 94.5%. CIFAR A top-1 +6.0%p.
- 이 PC의 D256L1(0.9666)은 팀 기록(0.9665)과 cos가 거의 같다 → 같은 `.alls`·보정이 다른 PC에서도 재현된다.
- ORT(상위 README 8절)에서는 회전 효과가 cos +0.014였는데, DFC에서는 +0.011이다. ORT와 달리 CIFAR A(저해상도)가 크게 좋아졌다 (ORT: −0.40%p, DFC: +6.0%p). 원인은 판단하지 않는다 (가설: DFC는 LN을 분해해 정수로 계산하고(`layer_norm_decomposition`) LN 출력의 채널 쏠림에 더 민감할 수 있다).

### 2단계 compile (10/18 10:20~11:11, 커밋 `bb9e935`)
| 구성 | compile | HEF 크기 | 컨텍스트 (profiler) | HEF sha256 | profiler: 가중치 / 층 / 16비트 SC |
|---|---|---|---|---|---|
| D256L1 (이 PC) | rc=0, 25분 5초 | **99,831,808 B (99.8 MB)** — 팀 HEF와 바이트 수 같음 | 17 | `2413dd5b…` (팀 `b33a266f…`와 다름) | 92.22 M / 343 / 475 |
| rothad_D256L1 | rc=0, 25분 19초 | **107,540,480 B (107.5 MB, +7.7 MB, +7.7%)** | 17 | `7243b76c…` | 92.22 M / 343 / 475 |

- 컴파일된 층·가중치 수·컨텍스트·16비트 SC 수가 같은데 HEF가 7.7 MB 크다. profiler 표에서는 L2 가중치 +1.1%, 컨텍스트 간 전송 +2%만 보인다. **원인은 확인하지 않았다.**
- 팀 기록의 compile 시간(1시간 25분, 쭈리 PC)보다 짧다 (25분, 285K).

### 3단계 실칩 (AGX Orin + Hailo-10H, HailoRT 5.3.0, 10/18 11:13~11:55)
- 실행 전 `ps`로 다른 Hailo·python 작업 없음을 확인했고, 사용자도 없었다 (`who` 비어 있음). HEF는 양쪽 sha256 확인 (`hef_sha256.txt`), `parse-hef` 정상 (입력 UINT8 NHWC 224×224×3, 출력 `ew_add26` UINT16 1×1×768).
- 코드 `siglip2/hailo/s2_hailo_chip.py` (eval: FLOAT32 입출력, 양자화는 HailoRT / bench: 입력 UINT8, 출력 UINT16, 배치 1). 분석 `s2_hailo_chip_eval.py` (dev 캐시 `clf`·`ref_fp32` 앞 200장, 에뮬레이터 스크리닝과 같은 기준). AGX dev200 묶음의 x·y는 dev 캐시 앞 200장과 같다 (`clf` 차이 ≤ 2e-7, FP32 예측 같음).
- 측정 도중 고친 버그 3개 (모두 커밋, 결과 전): bench의 출력 형식 UINT8 강제 → `configure` INVALID_ARGUMENT (`7dde9ce`). 입력 임시 버퍼가 비동기 실행 중 해제될 수 있음 (`6b8b5df`). 마지막 실행 직후 바로 VDevice를 해제하면 HAILO_TIMEOUT으로 멈춤 → 0.2초 대기 + `configured.shutdown()` (`da6831e`, `batch_bench.py`와 같은 순서). 실패한 bench 로그는 `chip/bench_fail/`에 있다. **고친 코드로 eval을 다시 돌려 임베딩이 처음 eval과 비트 단위로 같음을 확인했다** (`chip/eval2.log`).

#### H2 실칩 정확도 (`chip_eval.json`, dev 6×200)
| 구성 | CIFAR A | Pets | Flowers | Cars | Food | Caltech | 고화질 Δtop-1 | 고화질 cos |
|---|---|---|---|---|---|---|---|---|
| D256L1 실칩 | 66.5 | 91.5 | 84.5 | 92.5 | 86.5 | 96.0 | −0.7 | 0.9649 |
| **rothad_D256L1 실칩** | **69.5** | 91.5 | 87.5 | 93.5 | 88.5 | 95.5 | +0.4 | **0.9766** |

- rothad − D256L1 (실칩, 고화질 5종): top-1 **+1.1 [−0.1, +2.4]%p** (200장, 구간에 0 포함), cos **+0.0117**, 사진의 89.7%에서 cos가 오름, 예측 일치 94.2%, CIFAR A +3.0%p.

에뮬레이터 ↔ 실칩 (같은 HAR/HEF, 사진별):
| 구성 | 예측 일치 (고화질 / CIFAR A) | 사진별 cos 차이 평균 / 최대 | 실칩 − 에뮬레이터 cos | 실칩 − 에뮬레이터 top-1 [95%] |
|---|---|---|---|---|
| D256L1 | 95.6% / 78.5% | 0.0075 / 0.088 | −0.0017 | −0.2 [−1.2, +0.9] |
| rothad_D256L1 | 97.5% / 84.0% | 0.0052 / 0.063 | −0.0009 | 0.0 [−0.8, +0.9] |

- 실칩 정확도는 에뮬레이터와 구분되지 않는다. 회전 모델이 에뮬레이터와 더 잘 맞는다 (예측 일치 97.5 vs 95.6%).

#### H3 실칩 지연 (`chip/bench.json`, 배치 1, 워밍업 20 + 200회, 두 HEF 번갈아 3회, monitor 끔, `OPENBLAS_NUM_THREADS=1`)
| 회차 | D256L1 중앙값 (평균) ms | rothad 중앙값 (평균) ms |
|---|---|---|
| 1 | 45.32 (45.36) | 46.18 (46.21) |
| 2 | 45.35 (45.37) | 46.18 (46.22) |
| 3 | 45.45 (47.42, p90 51.4) | 52.35 (52.34) |
| **3회 중앙값** | **45.35** | **46.18 (+0.83 ms, +1.8%)** |

- `hailortcli run2 -t 10` 처리량 (FPS): 11:15 D256L1 22.78 → 19.98, rothad 21.18 → 19.68 / 11:52 D256L1 19.99·19.98, rothad 19.68·19.68.
- **H3: 차이 +1.8%로 5% 기준 안 → 원인 분석 대상 아님.** 1·2회차는 rothad가 매번 약 0.86 ms 느리다 (일관된 작은 차이, HEF가 7.7 MB 크고 컨텍스트 간 전송이 2% 많은 것과 같은 방향 — 추정).
- 3회차에 두 HEF 모두 약 52 ms로 늘었고 (D256L1은 회차 도중 전환), run2도 22.8 → 20.0 FPS로 내려갔다. 계속 부하를 받은 뒤 칩이 느린 상태로 바뀐 것으로 보인다 (온도·클럭 추정, 확인 안 함 — 기기 설정은 건드리지 않음). 같은 상태끼리 비교한 1·2회차와 run2(둘 다 낮은 상태) 모두 rothad가 1.5~1.9% 느리다.

## 7. 결과 요약 (채택은 팀 결정)
| 구성 | HEF 크기 / 컨텍스트 | 에뮬레이터 고화질 cos / Δtop-1 | 실칩 고화질 cos / Δtop-1 | 실칩 CIFAR A | 실칩 지연 배치 1 (3회 중앙값) | run2 FPS |
|---|---|---|---|---|---|---|
| D256L1 (이 PC 재빌드, `2413dd5b…`) | 99.8 MB / 17 | 0.9666 / −0.5 | 0.9649 / −0.7 | 66.5 | 45.35 ms | 20.0 ~ 22.8 |
| **rothad_D256L1** (`7243b76c…`) | **107.5 MB (+7.7%)** / 17 | **0.9775** / +0.4 | **0.9766** / +0.4 | **69.5** | 46.18 ms (+1.8%) | 19.7 ~ 21.2 |
| 차이 (rothad − D256L1) | +7.7 MB | +0.0109 / top-1 +0.9 [−0.2, +2.1] | +0.0117 / top-1 +1.1 [−0.1, +2.4] | +3.0 | +0.83 ms | −0.3 ~ −1.6 |

- **H1: 개선 후보** (에뮬레이터 cos 0.9775 > 0.9666). 실칩에서도 같은 크기로 재현된다 (cos +0.0117).
- top-1은 200장 × 5종 표본에서 두 경로 모두 구간이 0을 포함한다 (점추정 +0.9 / +1.1%p). 유의성을 보려면 dev 전체 비교가 필요하다. 에뮬레이터 dev 전체는 약 45분 × 9조각이고(G4 기록), 실칩은 1200장에 약 70초라 실칩 dev 전체가 훨씬 싸다 (제안, 하지 않음).
- 비용: HEF +7.7 MB, 배치 1 지연 +1.8%.
- 기준선 주의: 비교 대상 D256L1은 이 PC에서 다시 만든 HEF다. 팀 HEF(`b33a266f…`)와 크기는 같고 sha256은 다르다. 에뮬레이터 cos는 팀 기록과 같다 (0.9666 vs 0.9665).

## 8. dev 전체 실칩 짝 비교 (설계, 10/18 15:3x — 결과 전에 고정)
- **비교:** rothad_D256L1 HEF(`7243b76c…`) vs D256L1 HEF(`2413dd5b…`), 둘 다 이 PC에서 만든 것. AGX Hailo-10H 실칩, dev 6종 전체(CIFAR A 1000, Pets 990, Flowers 957, Cars 967, Food 985, Caltech 1024장), 두 HEF에 같은 사진.
- **입력:** 데스크탑 dev 캐시 `<set>_x.npy`를 그대로 AGX로 보낸다 (데이터셋별로 나눠 보내고 양쪽 sha256 확인, 합계 3.4 GB, AGX `~/ssd` 여유 746 GB). 실행은 지금까지와 같은 `s2_hailo_chip.py eval` (FLOAT32 입출력, NHWC).
- **계산(데스크탑):** FP32 기준 `ref_fp32`와 클래스 임베딩 `clf`는 데스크탑 dev 캐시 것 (`s2_hailo_chip_eval.py`).
- **지표:** 고화질 5종 Δtop-1 (rothad − D256L1) 짝 부트스트랩 95% 구간 (데이터셋별 층화, 시드 0, 1000회 — `s2_paired_ci.boot_hr`와 같은 함수), 고화질 cos, CIFAR A Δtop-1. FP32 대비 Δtop-1도 함께 적는다.
- **판정:** 구간 하한 > 0 → **"rothad가 유의하게 좋음"**. 구간이 0을 포함 → **"동률"**. (하한 < 0이고 상한 < 0 → "rothad가 유의하게 나쁨"으로 적는다.) 동률이면 지연 차이(+1.8%, 6절 H3)와 HEF 크기 차이(+7.7%)를 함께 적어 팀이 판단한다. 채택은 팀 결정.
- **기록용 1 — 칩 온도:** `hailo_platform` `Device.control.get_chip_temperature()` (읽기 전용, ts0·ts1 °C). eval 전후, 그리고 지연 3회를 다시 재며 회차마다 전후에 읽는다 (6절 3회차 지연 증가가 온도 때문인지 보기 위함). 측정 중에는 읽지 않는다. `hailortcli monitor`는 쓰지 않는다 (켜 두면 5초 사이 52 → 54 °C로 올랐다, 15:26 확인). 설정 변경 없음.
- **기록용 2 — HEF 크기 차이:** 컴파일 로그·profiler에 가중치 압축 항목이 있으면 두 HEF를 비교하고, 없으면 "미확인"으로 둔다 (`.alls`는 `compression_level=0`, profiler `compression_rate`는 두 HEF 모두 N/A).

## 9. dev 전체 실칩 결과 (10/18 15:28~15:41, AGX Hailo-10H, `chip_eval_full.json`, `chip_full/`)
- 실행 전 AGX 사용자·다른 Hailo 프로세스 없음 확인. dev 캐시 `x.npy` 6개 양쪽 sha256 OK (`chip_full/` 기록, AGX `~/ssd/junyong/hailo_g3q/devfull/`). 실행 `chip_full/devfull_run.sh` (nohup). 임베딩(각 약 17 MB)은 저장소에 넣지 않고 데스크탑 `~/siglip2_work/hailo_g3q/chip_full/`에 둔다 (sha256 `chip_full/emb_sha256.txt`).
- 분석 커밋 `7b964b9` (`s2_hailo_chip_eval.py --k 0`). FP32 기준·clf는 데스크탑 dev 캐시.

| 구성 (실칩, dev 전체) | CIFAR A (1000) | Pets (990) | Flowers (957) | Cars (967) | Food (985) | Caltech (1024) | 고화질 Δtop-1 (FP32 91.66%) | 고화질 cos |
|---|---|---|---|---|---|---|---|---|
| D256L1 | 71.10 | 91.82 | 84.01 | 92.24 | 86.19 | 95.12 | −1.79 | 0.9654 |
| **rothad_D256L1** | **72.90** | 92.22 | 85.79 | 92.76 | 87.92 | 94.92 | **−0.94** | **0.9766** |
| rothad − D256L1 | +1.80 | +0.40 | +1.78 | +0.52 | +1.73 | −0.20 | **+0.85 [+0.27, +1.45]** | +0.0112 |

- **판정 (8절 기준): 구간 하한 +0.27 > 0 → "rothad가 유의하게 좋음".** 사진의 90.4%에서 cos가 오르고, 예측 일치 94.0%. 고화질 5종 중 4종에서 좋아지고 Caltech만 −0.20.
- 독립 재계산 (다른 코드: 정규화 없이 argmax, 다른 시드 12345·5000회): 점추정 +0.85, 95% [+0.25, +1.46] — 같다.
- 참고 (같은 기준 FP32 91.66%): ORT 최종 R0 −0.62 [−1.09, −0.08], ORT rothad −0.61 (상위 README 8절). 실칩 D256L1(−1.79)은 ORT보다 1.2%p 낮고, 회전으로 그 차이의 약 70%가 줄었다 (−0.94).
- 비용 (변경 없음): HEF +7.7 MB (+7.7%), 배치 1 지연 +1.8~1.9% (아래). 채택은 팀 결정.

### 기록용 1: 칩 온도와 지연 (`chip_full/temps.jsonl`, `bench_r*.json`, 읽기 전용 `get_chip_temperature`, ts0 / ts1 °C)
| 시점 | 온도 | 지연 중앙값 D256L1 / rothad (ms) |
|---|---|---|
| 15:26 대기 (측정 전 확인) | 50.5 / 50.0 | — |
| 15:28 eval 전 | 60.4 / 59.9 | — |
| 15:33 D256L1 dev 전체 eval 뒤 (5분 30초 연속) | **92.3 / 91.6** | — |
| 15:39 rothad eval 뒤 | **95.9 / 95.3** | — |
| bench 1~3회 전후 (15:39~15:41) | 92.0 → 90.7 | 51.38 / 52.35, 51.37 / 52.34, 51.38 / 52.32 |
| (6절, 11:51 측정, 온도 미기록) 1·2회차 | — | 45.32 / 46.18, 45.35 / 46.18 |

- 칩이 약 91~92 °C일 때 두 HEF 모두 **약 13% 느리다** (D256L1 45.35 → 51.38, rothad 46.18 → 52.34). 6절 3회차의 52 ms와 같은 값이다. → **6절 3회차 지연 증가는 칩 온도 상승 때문일 가능성이 크다.** (이번에 식은 상태의 지연과 온도를 같은 실행에서 함께 재지 않았으므로 "온도 때문"을 직접 보이지는 못했다. 클럭 값은 읽지 않았다.)
- 두 HEF의 차이는 뜨거운 상태에서도 같다: +0.96 ms (+1.9%).
- 정확도는 온도와 무관해 보인다: 6절에서 같은 HEF로 eval을 두 번 돌린 임베딩이 비트 단위로 같았다. 이번 dev 전체 eval 동안 온도가 크게 올랐지만 (60 → 96 °C), 앞 200장의 예측을 6절 결과와 비교해 확인할 수 있다 (아래).
- `hailortcli monitor`를 켜 두면 5초 사이 52 → 54 °C로 오른 것을 봤다 (15:26). 측정 중에는 켜지 않았다.

### 기록용 2: HEF 크기 차이 (+7.7 MB)
- 컴파일 로그(`compile.log`)에 압축(compress) 관련 줄이 두 HEF 모두 **0개**다. profiler는 둘 다 `compression_level: 0`, `compression_rate: N/A`다 (`.alls` `compression_level=0`).
- → 가중치 압축 비교 항목이 없다. **"회전으로 가중치 분포가 퍼져 압축이 덜 됐다"는 가설은 미확인**으로 둔다. (6절: 가중치 수·층·컨텍스트·16비트 SC 수 같음, L2 가중치 +1.1%, 컨텍스트 간 전송 +2%.)
- 앞 200장 임베딩, dev 전체 실행(뜨거운 상태) vs 6절 실행 — D_D256L1: 비트 단위 같음 = True, 최대 차이 0
- 앞 200장 임베딩, dev 전체 실행(뜨거운 상태) vs 6절 실행 — rothad_D256L1: 비트 단위 같음 = True, 최대 차이 0
