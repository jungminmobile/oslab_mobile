# 저장소 지도 (repo_map) — SigLIP2 트랙 관점

- 상태: **초안** (2026-10-13, G0 첫 세션). 기준 커밋 `f969f83` (main).
- 목적: SigLIP2 트랙이 RN50 트랙과 같은 조건으로 실험하려면 무엇을 그대로 가져다 쓰고, 무엇을 바꿔야 하는지 정리한다.
- 이 문서는 RN50 쪽 파일을 **읽기만** 하고 쓴 것이다. RN50 쪽 코드는 수정하지 않았다.
- "확인 필요"가 붙은 항목은 코드를 실행해 보지 않고 읽어서만 판단한 것이다.

---

## 1. 저장소 구조 (main 기준, 추적 파일 56개)

```
oslab_mobile/
├── PRD.md               ★ 현재 실험 계획 (RN50 트랙 기준, 정민 작성 10/8). 실험 ID E0~E9
├── README.md            구조·핵심 규칙·협업 요약
├── CLAUDE.md            저장소 전체 AI 작업 규칙 (협업 규칙: 브랜치·옵션·best 파일·smoke test)
├── AGENTS.md            다른 AI 도구용 요약 → CLAUDE.md
├── .gitignore           data/, cache/, onnx/, weights/, *.onnx, *.hef, *.har, *.npy, rn50/results/ 등 제외
├── rn50/                RN50 트랙 코드 (정민) — 공통 규칙도 여기 있음
│   ├── common.py        경로·전처리·데이터 구간·ONNX 세션·임베딩·분류기·지표
│   ├── datasets.py      dev 캐시·보정 후보 생성 / test 전체 평가
│   ├── evaluate.py      모델 × dev 데이터셋 정확도·cos
│   ├── bench_speed.py   지연 반복 측정 (번갈아 측정)
│   ├── quantize_onnx.py ORT 정적 PTQ (QDQ), 노드 선택, 보정 방법·출처
│   ├── export_onnx.py   CLIP-RN50 → ONNX (+AttnPool q1)
│   ├── baseline_rn50.py PyTorch FP32 베이스라인 (정확도·파라미터·GMACs·지연·peak RSS)
│   ├── summarize.py     evaluate.jsonl → 마크다운 표
│   ├── prompts/cifar100_templates.json   CIFAR-100 18개 템플릿 (CLIP 논문)
│   ├── requirements.txt
│   └── README.md        설치·데이터 받기·실행 순서
├── hailo/               DFC 작업 공간 (README, DFC_SETUP.md). 스크립트는 아직 없음
├── results/             실험 결과 (실험 ID별 폴더). 현재 E0_measure/2026-10-12_repro_check 만 있음
├── docs/CODE_REVIEW.md  10/6·10/7 코드 검토 (보정 16장 버그 등)
├── archive/2026-10-07_night/   이전 실험. 참고만, 수치 사용 금지
└── siglip2/             이 트랙 (README.md만 추적됨, 나머지 문서는 이 PR에서 추가)
```

아직 없는 것 (PRD E0에서 만들 예정): `configs/rn50_best.json`, `rn50/smoke_test.py`, `.github/`(Actions 없음).

## 2. RN50 트랙 현황 (git·문서 기준)

| 항목 | 상태 |
|---|---|
| 최근 RN50 코드 변경 | `1ad6318` (PRD 검토 반영), `37a6816`, `06cc292`. **E0 코드 수정은 아직 main에 없음** |
| E0 출발점 | 10/12 재현 확인 완료: 정리된 코드가 10/7 결과를 그대로 재현 (`results/E0_measure/2026-10-12_repro_check/README.md`) |
| DFC | 정민 노트북 WSL `Ubuntu-22.04`, `~/hailo10_venv`, **DFC 5.4.0** 설치 완료 (`hailo/DFC_SETUP.md`) |
| E1 (DFC 관문) | 아직 결과 없음 |
| 측정 PC | 클라우드(Xeon, AVX512-VNNI, 2코어) + **정민 노트북(Core Ultra 7 258V, AVX-VNNI)**. 주 측정 조건 = 노트북 스레드 4 (PRD 4-3) |

→ E0에서 공통 코드(`evaluate.py`, `bench_speed.py`, `quantize_onnx.py`, `common.py`, `datasets.py`)의 기본값과 기록 형식이 바뀔 예정이다. SigLIP2는 **E0 PR이 main에 들어온 뒤의 코드**를 기준으로 맞추는 것이 안전하다.

## 3. PRD.md 핵심 (SigLIP2에 영향 있는 것만)

| 항목 | PRD 내용 | 절 |
|---|---|---|
| 목표 | 8비트 중심 양자화로 정확도·속도·용량·메모리 **파레토 곡선**, 최종 지점은 팀 상의 | 1 |
| 배점 | 예선: 보고서·설계 35, 기법 25, 성능 개선 25, 정확도 15 | 1 |
| 제출 | 예선은 채점자가 로컬 PC에서 **ONNX** 직접 실행 (베이스라인 + 최적화 모델) | 2 R1 |
| 이식성 | 같은 구성이 DFC `--hw-arch hailo10h`로 `.hef` 컴파일 성공해야 함 | 2 R2 |
| 속도 하한 | 측정한 **모든 스레드 수**에서 자기 FP32보다 느리면 제외 | 1 |
| 정수 커널 | 보고 수치는 ORT 기본 설정(`ORT_ENABLE_ALL`). **fake-quant는 쓰지 않는다(디버깅만)** | 2 R5 |
| 범위 | 이미지 인코더만 양자화, 텍스트는 FP32로 1회 계산 | 2 R6 |
| 스택 | Python 3.11, torch 2.14.0, onnx 1.23.1, **opset 17**, onnxruntime 1.25.0, QDQ, 가중치 S8 대칭 채널별, 활성값 8비트 텐서별 (기본 S8S8로 바꿀 예정) | 3 |
| S8S8 vs U8S8 | x86에서는 S8S8도 U8S8로 바뀌어 계산이 같다. 비-VNNI 포화 대책은 `reduce_range`·U8U8·`session.x64quantprecision=1` | 3 주석 |
| ORT ↔ DFC | DFC는 float ONNX에서 다시 양자화. INT8 층 → `a8_w8`, FP32 층 → `a16_w16` | 4-2 |
| 데이터 구간 | 보정: CIFAR train[0:45000] 또는 고화질 5종 train(dev 제외 해시 구간), **256장·시드 0** / dev **6종**: CIFAR train[45000:46000](A) + 고화질 5종 SHA-1 해시 약 1000장씩. CIFAR B([46000:47000])는 최종 후보 확인에만 (코드의 `dev all`은 B도 만든다) / test: 7종 전체, **E7에서 1회**. ImageNetV2는 test 전용 | 5 |
| 지표 | top-1/5, Δtop-1(짝 부트스트랩 1000회 95% CI), cos, 용량(MiB), 지연(워밍업 20, 200장×3회 번갈아), 처리량(배치 8), peak RSS(모델마다 새 프로세스) | 5 |
| 주 정확도 지표 | **고화질 5종 평균**. CIFAR-100·ImageNetV2는 별도 표기 (CIFAR는 Hailo 공식 수치 비교용) | 1, 5 |
| Caltech | BACKGROUND_Google·Faces·Faces_easy 포함, 이름은 background / off-center face / centered face | 5 |
| 기준 모델 | 자기 FP32 원본 export | 5 |
| 일정 | E0·E1 10/8~11 → E2·E3·E5a 10/12~15 → E4·E5b 10/16~20 → E8a·E6 10/21~23 → E7·E8b 10/24~26 → 보고서 10/27~11/1 | 9 |
| 주의 | PRD 3절 59행은 아직 "DFC는 미설치"라고 적혀 있다. `hailo/DFC_SETUP.md`(10/12)에는 5.4.0 설치 완료로 되어 있어 PRD 쪽이 오래된 문구 | 3 |
| 준혁 관련 | 비-VNNI PC 확보를 위해 "준혁 PC의 CPU 플래그 확인" 요청 (E0 ①) → **이 PC는 AVX512-VNNI 있음** (아래 6절) | 4-3 |

## 4. 공통 코드 분석 — SigLIP2에서 그대로 쓸 수 있는가

판정: ✅ 그대로 import / ⚠️ 환경변수·입력 이름만 맞추면 사용 / ❌ RN50 전용이라 수정 또는 별도 구현 필요

### 4-1. `rn50/common.py`

| 요소 | 판정 | 이유 |
|---|---|---|
| 경로 (`DATA_ROOT`, `EXTRA_ROOT`, `ONNX_DIR`, `CACHE_DIR`, `RESULTS_DIR`) | ⚠️ | 환경변수 `RN50_DATA/RN50_EXTRA/RN50_ONNX/RN50_CACHE/RN50_RESULTS`를 **import 시점에** 읽는다. SigLIP2 경로로 바꾸려면 import 전에 설정해야 한다 |
| `MODEL`, `IMG_SIZE`, `PREFIX`, `FP32_IMAGE`, `FP32_TEXT` | ❌ | `IMG_SIZE = {"RN50":224, ...}[MODEL]` — `CLIP_MODEL`에 SigLIP2 이름을 넣으면 **KeyError**. 기본값(RN50)으로 두면 224라 크기는 맞지만 파일 이름 앞부분이 `rn50_`이 된다 |
| `MEAN`, `STD`, `preprocess()`, `crop_u8()`, `to_input()` | ❌ | OpenAI CLIP 평균·표준편차, bicubic 짧은 변 리사이즈 + 가운데 자르기로 고정. SigLIP 계열은 평균·표준편차 0.5, 자르기 없는 리사이즈를 쓰는 것으로 알려져 있다 (**확인 필요**: open_clip `ViT-B-16-SigLIP2`의 `preprocess_cfg`로 확인). `crop_u8`은 `transforms[0]`, `[1]`이 Resize·CenterCrop이라고 가정한다 |
| `tokenizer()` | ❌ | `open_clip.get_tokenizer("RN50")` — CLIP BPE, 길이 77. SigLIP2는 다른 토크나이저 (**확인 필요**: open_clip 설정의 토크나이저와 길이) |
| `l2n()` | ✅ | 모델 무관 |
| `CALIB_RANGE`, `VAL_SPLITS`, `VAL_SIZE` | ✅ | CIFAR 구간 규칙. 그대로 써야 RN50과 같은 사진이 된다 |
| `TEMPLATES` (CIFAR 18개) | ✅ | 단, 프롬프트 템플릿은 SigLIP 권장 템플릿과 dev 비교 예정 (ROADMAP G2) |
| `cifar100()` | ❌ | 내부에서 `preprocess()`(CLIP 전처리)를 붙인다 |
| `fixed_sample()` | ✅ | |
| `session()` | ✅ | 스레드·그래프 최적화 설정. 모델 무관 |
| `onnx_path()` | ⚠️ | `ONNX_DIR` 기준 |
| `embed()` | ⚠️ | 입력 이름 `"image"` 고정, uint8이면 CLIP `to_input`으로 정규화, **기본 `fake_quant=True`**. SigLIP2 ONNX의 입력 이름을 `image`로 내보내고, float32로 미리 정규화한 입력을 주면 쓸 수 있다 |
| `zeroshot_classifier()` | ❌ | `FP32_TEXT` ONNX + CLIP 토크나이저 + 입력 이름 `tokens`. 프롬프트 앙상블 방식(정규화 → 평균 → 정규화)은 같게 따라가야 한다 |
| `topk_metrics()` | ✅ | top-1/5, cos_mean, cos_p1. SigLIP2도 top-1 순위 = 코사인 순위라 그대로 맞다 |
| `append_jsonl()` | ✅ | `RESULTS_DIR` 기준 |

### 4-2. `rn50/datasets.py`

| 요소 | 판정 | 이유 |
|---|---|---|
| `HIGHRES` (데이터셋 소스·분할·템플릿·n_pool), `CARS_T`, `CALTECH_T`, `CALTECH_RENAME`, `DEV_N`, `CAL_N` | ✅ | 같은 소스·같은 템플릿을 쓰려면 반드시 import. `CAL_N`은 E0에서 64 → 256으로 바뀔 예정 |
| `u_of()` (SHA-1 해시 → dev/보정 선택) | ✅ | **사진 선택 규칙의 핵심**. 이것을 쓰면 RN50과 같은 사진이 뽑힌다 |
| `ensure()`, `parquet_rows()`, `classnames()`, `imagenetv2_rows()`, `imagenet_meta()` | ⚠️ | 모델 무관. 단 `EXTRA_ROOT`(= `RN50_EXTRA`) 아래 상대 경로(`pets/train-0.parquet`, `pets/test-0.parquet` 등)로 내려받는다. 새로 받은 파일 삭제 여부는 호출 쪽이 정한다: `dev` 명령은 `--keep-downloads`가 없으면 지우고, `eval_test`는 항상 남긴다 |
| `decode()` | ❌ | `crop_u8`(CLIP 전처리) 사용 |
| `build_dev()` | ❌ | 사진 선택 + CLIP 전처리 + 분류기 + FP32 임베딩 + 저장이 한 함수에 묶여 있다. 선택된 사진의 **인덱스나 해시를 따로 저장하지 않으므로**, SigLIP2는 `u_of`·`HIGHRES`를 써서 같은 선택을 다시 하고 전처리만 바꿔야 한다. **고화질 보정 후보 풀(`calib_<이름>_u8.npy`)도 여기서 CLIP 자르기로 만들어진다** → SigLIP2는 자체 캐시에 다시 만들어야 한다 |
| `eval_test()` | ❌ | CLIP 전처리, `FP32_IMAGE`, **`fake_quant=True` 고정**, 기록에도 `"exec": "fake_quant"`를 고정으로 쓴다. E0에서 정수 커널로 바뀔 예정 |
| 캐시·결과 파일 이름 | ⚠️ **주의** | `cifar_A_x.npy`, `pets_dev_clf.npy`, `calib_pets_u8.npy`, `evaluate.jsonl`, `baselines.jsonl`처럼 **모델 이름이 들어가지 않는다.** 기본 경로가 `rn50/` 아래라서, SigLIP2가 `RN50_CACHE`·`RN50_RESULTS`를 설정하지 않으면 **RN50의 캐시·결과를 조용히 덮어쓴다** |

`rn50/README.md`의 데이터 받기 명령은 Pets를 `extra_data/pets_test.parquet`에 저장하지만, 코드는 `pets/test-0.parquet`을 찾는다 (README와 코드 불일치. 코드가 다시 내려받게 됨).

### 4-3. 나머지 스크립트

| 파일 | 판정 | 이유 |
|---|---|---|
| `evaluate.py` | ⚠️ | 캐시(`<이름>_x/_y/_clf/_ref_fp32.npy`)만 같은 형식이면 모델과 무관하다. 단 캐시 `x`가 uint8이면 CLIP 정규화가 적용되므로 SigLIP2 캐시는 float32로 저장하거나 정규화 방식을 바꿔야 한다. **기본 `--exec fake`** (E0에서 `int`로 바뀔 예정) |
| `bench_speed.py` | ⚠️ | 입력이 CIFAR A 앞 200장(CLIP 전처리)이지만 속도는 입력 내용과 무관하고 크기도 224라 그대로 쓸 수 있다. 입력 이름 `image` 필요. CIFAR 원본(`RN50_DATA`)이 있어야 실행된다. 현재 기본은 워밍업 10, 반복 3 (PRD는 워밍업 20). CPU 기록이 부정확 (E0에서 수정 예정) |
| `quantize_onnx.py` | ⚠️/❌ | `ImageCalib`(입력 이름 `image` 고정), `select_excluded()`(argparse 형태 객체에 `exclude/keep/only/only_exact` 필요), `METHODS`는 재사용 가능. 그러나 원본 파일 이름이 `{PREFIX}_{enc}_fp32{variant}.onnx`로 고정, 보정 사진이 CLIP 전처리(`cifar100()`, `calib_*_u8.npy` + `to_input`), 활성값 `QUInt8` 고정. 그대로 쓰면 **SigLIP2 보정 사진이 잘못 정규화된다** |
| `summarize.py` | ⚠️ | 형식은 맞지만 `--exec` 기본이 `int`라 `evaluate.py` 기본(`fake`) 결과를 걸러 낸다. `--results` 기본이 `rn50/results`이고 `RN50_RESULTS`를 읽지 않는다. `--strip` 기본값이 RN50 이름 규칙이고, 표에 `cifar_B`가 없다 → 인자를 명시해서 쓰면 사용 가능 |
| `export_onnx.py` | ❌ | CLIP-RN50 전용 (AttnPool q1). SigLIP2 전용 export를 새로 작성 |
| `baseline_rn50.py` | ❌ (참고) | RN50 전용. peak RSS(`mem_probe` 서브프로세스, 1ms 샘플링)는 **RN50 PyTorch 모델**을 재는 것이고, ONNX 모델용 RSS 측정은 아직 없다 (PRD E0 ③에서 추가 예정). fvcore GMACs 계산 방식은 **같은 방식으로 따라가야** 비교가 된다 |
| `prompts/cifar100_templates.json` | ✅ | |

### 4-4. import 방법과 주의점

- 공통 코드는 패키지가 아니다. 각 스크립트가 `import common as C`로 같은 폴더 모듈을 부른다.
  SigLIP2에서 쓰려면 `sys.path`에 `rn50/`을 넣고 `import common`, `import datasets` 해야 한다.
- 모듈 이름이 `common`, `datasets`로 흔해서 **이름 충돌 위험**이 있다 (예: Hugging Face `datasets` 패키지를 설치하면 `import datasets`가 그쪽을 가리킬 수 있다. sys.path 순서에 따라 다름 — **확인 필요**).
- 환경변수는 import 시점에 읽히므로 SigLIP2 스크립트는 `RN50_*` 환경변수를 import **전에** 설정해야 한다.
  특히 `RN50_CACHE`·`RN50_RESULTS`를 빼먹으면 RN50의 파일을 덮어쓴다 (4-2절). SigLIP2 쪽 진입 코드에서 설정 여부를 assert하는 것을 제안.
- 임베딩 차원 1024는 docstring에만 적혀 있고 코드는 차원과 무관하다. SigLIP2 ViT-B/16의 차원은 다르다 (768로 알려져 있음, **확인 필요**). 캐시·분류기 파일을 섞지 않도록 주의.

### 4-5. SigLIP2를 붙이기 위해 필요한 공통 코드 변경 (제안만, 직접 고치지 않음)

아래는 **공통 코드 PR(팀원 리뷰 필요)** 후보다. 모두 "기본 동작은 그대로"를 지키는 방식으로 제안한다.

1. `common.py`: 모델별 전처리 설정(평균·표준편차·리사이즈 방식·입력 크기)을 바꿀 수 있게 한다. 예: `CLIP_MODEL`이 표에 없으면 KeyError 대신 별도 설정을 받는다.
2. `datasets.py`: 사진 **선택**(해시·인덱스 목록)과 **전처리·저장**을 분리하거나, 선택된 사진의 해시 목록을 파일로 저장한다. 그러면 두 트랙이 "같은 사진"을 쓴다는 것을 파일로 증명할 수 있다.
3. `quantize_onnx.py`: 보정 사진 생성 함수를 모델별 전처리에 맞게 바꿀 수 있게 한다 (또는 SigLIP2는 `ImageCalib`·`select_excluded`만 import하고 보정 사진은 자체 생성).
4. 공통 코드를 패키지로 만들거나 모듈 이름을 바꿔 이름 충돌을 막는다 (선택).

→ 결정(D-009): 지금은 SigLIP2 쪽 어댑터로 시작하고, 공통 코드 PR은 정민의 E0 병합 뒤 연다.

## 5. requirements

| 패키지 | `rn50/requirements.txt` | SigLIP2 필요 조건 (ROADMAP G0) | 판단 |
|---|---|---|---|
| Python | 3.11 | 3.11 (이 PC venv: **3.11.9**) | 같음 |
| torch / torchvision | 2.14.0 / 0.29.0 (CPU 휠) | — | 같은 버전 사용 제안 |
| open_clip_torch | 3.3.0 | 2.31 이상 | 3.3.0으로 맞추면 조건 만족 |
| timm | (없음) | 1.0.15 이상 | **추가 필요**. SigLIP2 전용 의존성 → `siglip2/requirements.txt`에 따로 두는 것을 제안 |
| onnx / onnxruntime | 1.23.1 / 1.25.0 | RN50과 같은 버전 | 같음 |
| numpy, pillow, pyarrow, psutil, fvcore | 2.4.4, 12.2.0, 25.0.1, 7.2.2, 0.1.5.post20221221 | — | 같은 버전 |
| (DFC) | WSL Ubuntu 22.04, Python 3.10, DFC 5.4.0 | — | 예선 venv와 분리 |

이 PC의 SigLIP2 venv(`D:\mokerton\venvs\siglip2`)는 **아직 비어 있다** (pip 24.0, setuptools 65.5.0만 있음).

## 6. 이 PC 환경 (2026-10-13 확인)

| 항목 | 값 | 확인 방법 |
|---|---|---|
| CPU | 11th Gen Intel Core i7-1165G7 @ 2.80GHz, 4코어 8스레드 | `Win32_Processor`, WSL `/proc/cpuinfo` (일치) |
| 명령어 | avx2 ✅, avx512f ✅, **avx512_vnni ✅**, avx_vnni ❌ (avx512bw·vl ✅) | WSL `/proc/cpuinfo` 플래그. avx2·avx512f는 Windows `IsProcessorFeaturePresent`로도 교차 확인 |
| RAM | 15.7 GB | `Win32_ComputerSystem` |
| GPU | Intel Iris Xe (NVIDIA 없음 → CUDA 불가) | `Win32_VideoController` |
| OS | Windows 11 Home 10.0.26300 | |
| WSL | Ubuntu-24.04 (기본), Ubuntu-20.04. **Hailo venv 없음** (`~/hailo*` 없음) | `wsl -l -v`, `ls ~` |
| D 드라이브 | 466 GB 중 176 GB 남음 | `df -h /d` |

- 이 PC도 VNNI가 있으므로 **U8S8 포화(비-VNNI 문제)를 재현할 수 없다.** PRD E0 ①의 "준혁 PC 확인" 답이다.
