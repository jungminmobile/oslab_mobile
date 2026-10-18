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
