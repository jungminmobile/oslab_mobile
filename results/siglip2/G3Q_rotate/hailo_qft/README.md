# G3Q-QFT: Hailo 미세조정(QFT)을 Hailo 최종(rothad_D256L1)에 (설계, 2026-10-18 23:3x)

> **상태 (10/18, 용 결정): GPU 경로는 접는다 (5절). 대신 CPU 미세조정 시험 1회(6절)로 시간만 잰다.**

**1~3절(설계·기준)은 결과 전에 고정한다. 판정용 실험이다. 채택은 팀 결정.** 지금은 0·1단계만 하고 2단계(본 실행)는 rothad_D256L1 vs M2rot_D256L1 실칩 비교 결과를 본 뒤 이긴 쪽에 적용한다 (용 지시).

## 1. 무엇을
- 입력 HAR: rothad 파싱 HAR `~/siglip2_work/hailo_g3q/siglip2_rothad_b16_224.har` (sha256 `303a8828…`, `../hailo/parsed_har_sha256.txt`). `.alls`는 `siglip2/hailo/g4_D256L1_official_b9fc2.alls`(수준 1)에 미세조정 한 줄만 더한다. 보정 사진은 지금 쓰는 `calib_highres256_s0_nhwc.npy` 256장 그대로 (D-035, 데이터 구성 변경 없음).
- 기준값 (같은 PC, CPU optimize, `../hailo/` 6절): rothad_D256L1 스크리닝 고화질 cos **0.9775**, 출력 SNR **16.63 dB** (`optimize_rothad_D256L1.log` 59~60행).

## 2. DFC 5.4.0 미세조정 옵션 (설치된 `hailo_model_optimization` 소스에서 확인, `mo_config_model.py` `FineTuneConfig`, `acceleras_definitions.py`)
| 옵션 | 기본값 | 이 실험 |
|---|---|---|
| `policy` | 수준 2부터 자동 켜짐 (수준 1에서는 꺼짐) | `enabled` (명시) |
| `dataset_size` | **1024** (`DEFAULT_DATASET_SIZE`) — 보정 데이터가 이보다 적으면 예외 | **256** (보정 256장 그대로, 데이터 늘리지 않음) |
| `epochs` | **4** (`DEFAULT_EPOCHS`) | 1단계 1, 2단계 4 (아래) |
| `batch_size` | 보정 batch 크기 | 1단계 2 (OOM이면 1) |
| `learning_rate` | `2e-4 / 8 × batch_size` | 기본 |
| `def_loss_type` / `loss_layer_names` | `l2rel` / 출력 층 | 기본 |
| `val_images` | 4096 (보정 데이터에서 `take`, 부족해도 예외 없음) | 기본 |
| `lr_schedule_type`, `warmup_epochs`, `decay_epochs` | cosine restarts, 1, 1 | 기본 |
- 순서 (`optimization_flow.post_quantization_optimization`): GPTQ → train_encoding → **bias_correction** → adaround → block_round → **finetune**. 수준 1의 바이어스 보정 뒤에 미세조정이 돈다.
- 문법: `post_quantization_optimization(finetune, policy=enabled, dataset_size=256, epochs=E, batch_size=B)`.

## 3. 단계와 기준 (결과 전에 고정)
| 단계 | 내용 | 통과 기준 / 판정 |
|---|---|---|
| **0** GPU 경로 확인 | rothad_D256L1과 같은 `.alls`(미세조정 없음)를 GPU를 켜고 optimize → 출력 SNR, 스크리닝(dev 6×200, `s2_dfc_screen.py`, CPU 에뮬레이터) | 스크리닝 고화질 cos가 0.9775와 **0.002 이내**. 다르면 멈추고 보고 |
| **1** 짧은 시험 | 미세조정 켬, epochs 1, batch 2 (OOM이면 1) | 에러 없이 끝남. GPU 메모리 최대·시간 기록 (판정 아님). OOM이 batch 1에서도 나면 멈추고 보고 |
| 2 본 실행 (보류) | epochs 4 (DFC 기본; 1단계 시간으로 추정해 3시간이 넘으면 그 이하로 줄이고, 실행 전에 이 표에 적는다) → 스크리닝 | 고화질 cos ≥ 0.9775 + **0.003** → "개선 후보" → compile → AGX 실칩 dev 전체 짝 비교 (rothad_D256L1 대비, 층화 짝 부트스트랩 시드 0·1000회). 하한 > 0이면 팀에 제안. HEF 크기·컨텍스트·실칩 지연도 기록 |
- 공통: DFC는 `flock ~/siglip2_work/dfc.lock` 안에서, 시작 전 `free -g` 여유 ≥ 15 GB와 `nvidia-smi` GPU 메모리 여유 확인. GPU를 쓰는 optimize만 `CUDA_VISIBLE_DEVICES=0` (`gpu_env.sh`), 스크리닝(에뮬레이터)·compile은 `-1`.

## 4. 진행 기록
### 0단계 시도 전 GPU 확인 (10/18 23:2x) — **막힘, 멈춤**
- `~/hailo10_venv`(TF 2.19.1)에서 GPU는 보인다: `list_physical_devices('GPU')` → GPU:0, `get_device_details` → RTX 5060, compute capability (12, 0).
- 그러나 **첫 연산에서 실패한다**: `tf.random.normal` 뒤 elementwise `Mul`이 `CUDA_ERROR_INVALID_HANDLE` (`cuLaunchKernel` 실패). 비대화형 셸에 `~/.bashrc` 118~121행의 CUDA 변수(`gpu_env.sh`: PATH·LD_LIBRARY_PATH `/usr/local/cuda-12.5/lib64:/usr/lib/wsl/lib`·CUDA_CACHE_MAXSIZE 4 GB·CUDA_CACHE_PATH)를 넣어도, `bash -i`(대화형, .bashrc 전체)로 돌려도 같다.
- 참고 정보 (읽기만 함): TF 빌드 `cuda_version 12.5.1`, `cuda_compute_capabilities ['sm_60','sm_70','sm_80','sm_89','compute_90']` → sm_120용 바이너리가 없어 compute_90 PTX를 드라이버가 즉석 변환해야 한다. 드라이버 610.57.01 (UMD CUDA 13.3). venv 안에 pip CUDA 라이브러리(`nvidia_cuda_runtime_cu12` 12.8.90, `nvidia_cublas_cu12` 12.8.4.1, `nvidia_cudnn_cu12` 9.10.2.21 등)도 있고 시스템 CUDA 12.5와 함께 있다. `~/.nv/ComputeCache`는 1.3 MB(변환 결과가 거의 쌓이지 않음).
- 원인 가설 (확인 안 함): (가) compute_90 PTX → sm_120 즉석 변환이 TF 2.19 커널에서 실패 (Blackwell을 TF 2.19가 공식 지원하지 않음) (나) venv의 pip CUDA 12.8 라이브러리와 시스템 CUDA 12.5가 섞여 로드됨.
- **0단계를 시작하지 않았다** (optimize·잠금 사용 없음). 패키지 설치·venv 변경은 금지라 더 진행하지 않는다.

## 5. GPU 경로 불가: 원인 가설과 확인한 사실 (10/18, 용 결정으로 접음)
확인한 사실:
- TF 2.19.1(`~/hailo10_venv`, DFC 5.4.0이 버전을 고정)은 RTX 5060을 장치로는 본다 (GPU:0, compute capability 12.0).
- 첫 GPU 연산에서 `CUDA_ERROR_INVALID_HANDLE` (`cuLaunchKernel` 실패, elementwise `Mul`). CUDA 변수 넣은 비대화형 셸, `bash -i` 대화형 셸 모두 같다.
- **CUDA 12.5 설치 전에도 같은 에러가 났다**: 10/18 15:3x, `CUDA_VISIBLE_DEVICES=-1`을 빠뜨린 채 `hailo_model_optimization`을 import했을 때 그 패키지의 첫 TF 연산(`tf.constant(0.0) + 1.0`, `AddV2`)이 같은 `CUDA_ERROR_INVALID_HANDLE`로 실패했다 (G3Q-H 9절 작업 기록). 즉 시스템 CUDA 12.5 설치 여부와 관계없이 같은 지점에서 실패한다.
- TF 빌드 정보: `cuda_compute_capabilities ['sm_60','sm_70','sm_80','sm_89','compute_90']` — sm_120용 바이너리가 없다. 드라이버 610.57.01 (UMD CUDA 13.3). venv 안에 pip CUDA 12.8 라이브러리, 시스템에 CUDA 12.5. 변환 캐시 `~/.nv/ComputeCache` 1.3 MB.
원인 가설 (확인 안 함):
- (가, 유력) TF 2.19의 자체 GPU 커널(compute_90 PTX)이 sm_120에서 실행되지 않는다. 장치 인식·라이브러리 로드는 되지만 커널 실행 단계에서 핸들이 무효다. CUDA 12.5 설치 전후 에러가 같다는 것이 이 가설과 맞는다.
- (나) venv의 pip CUDA 12.8 라이브러리와 시스템 CUDA가 섞여 로드된다 — 설치 전에도 같은 에러였으므로 시스템 CUDA 12.5가 원인일 가능성은 낮다.
- DFC가 TF 버전을 고정하고 venv 변경이 금지라 TF를 바꿔 확인할 수 없다 → **GPU QFT는 하지 않는다.**

## 6. CPU 미세조정 시험 1회 (10/18 23:18~23:45, `CUDA_VISIBLE_DEVICES=-1`, 잠금 안, 커밋 `65c789c` dirty 0)
- `.alls`: `siglip2/hailo/g3q_qft_cpu_trial.alls` = D256L1 `.alls` + `post_quantization_optimization(finetune, policy=enabled, dataset_size=32, epochs=1, batch_size=2)`. 입력 rothad 파싱 HAR, 보정 256장 그대로. 스크립트 `siglip2/hailo/g3q_qft_cpu_trial.sh`. 시작 시 메모리 여유 29 GB, 부하 0.04.
- **결과: 에러 없이 끝남 (rc=0).** 전체 26분 52초, 최대 RSS 25.0 GB (`/usr/bin/time -v`). 로그 `optimize_qft_cpu_trial.log`.

| 단계 | 시간 |
|---|---|
| LN 분해 · 통계 수집 · 기타 | 약 2분 |
| 바이어스 보정 (256장) | 15분 27초 |
| **미세조정 (32장, 1 epoch, batch 2 = 16 스텝)** | **3분 31초** = 첫 스텝 185초 (그래프 준비) + 나머지 15스텝 약 1.4초/스텝 |
| 층 노이즈 분석 | 2분 55초 |

- 손실(`l2rel`, 출력 `ew_add26`): 0.152 → 0.156. 16스텝은 전부 warmup 구간(`warmup_epochs=1`)이라 의미 없음. 출력 SNR 17.34 dB (미세조정 없음 16.63 dB) — **판정·스크리닝은 하지 않았다** (기록만).
- **256장 × 4 epoch (DFC 기본 epochs) 환산:** 256 × 4 ÷ batch 2 = 512 스텝 × 1.4초 ≈ 12분 + 첫 스텝 3분 → **미세조정 약 15분**, 나머지 optimize 약 23분 → **optimize 1회 약 40분** (여유를 잡아 1시간 이내). 3시간 기준보다 짧다. batch를 보정 기본값(더 큰 값)으로 하면 스텝 수는 줄고 스텝당 시간·메모리는 늘어난다 (재지 않음).
- 2단계(본 실행)는 용 지시대로 하지 않았다 (rothad_D256L1 vs M2rot_D256L1 실칩 비교 뒤 결정).
