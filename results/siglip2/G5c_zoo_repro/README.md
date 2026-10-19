# G5c: Zoo 공개 .alls로 공개 HEF가 재현되는가 — SigLIP 1세대 B/16 (설계 2026-10-19)

> 상태: **완료 (10/19 22:48).** 1~4절은 결과 전에 고정했다 (커밋 3df20d9). 5~6절은 결과·해석이다. 기록용 측정이다. 채택 판정 없음.
> 브랜치 `siglip2/g5c-zoo-repro` (`siglip2/g5b-vendor-recipe` f826496에서 만듦), worktree `~/oslab_g5c`. 담당: 용 (용 요청). dev만 쓴다 (test 안 씀).

## 1. 목적과 질문
- 목적: G5b("Zoo v5.0.0 .alls를 우리가 컴파일한 SigLIP2 HEF")를 **벤더 수준**으로 봐도 되는지 판단한다.
- 방법: Zoo가 HEF를 공개한 SigLIP 1세대 B/16에서, 공개 ONNX와 공개 .alls로 우리가 직접 컴파일한 HEF가 공개 HEF와 같은 구조·속도·출력을 내는지 본다.

## 2. 재료 (커밋하지 않음, 경로·sha256만)
| 이름 | 위치 | 크기 B | sha256 |
|---|---|---|---|
| 공개 HEF (Zoo v5.4.0 hailo10h) | AGX `~/ssd/junyong/hailo_g5/siglip_b_16_image_encoder.hef` (G5 1절) | 99,061,760 | `e2198ed5975da551d9a807c3c22d694d7d3f1e9034b9fe3bb3562601f8f40650` |
| 공개 ONNX zip | S3 `…/ZeroShotClassification/siglip/siglip_base_patch16_224/image_encoder/pretrained/2025-03-16/siglip_base_patch16_224_vision_encoder.zip` → `~/siglip2_work/g5c/dl/` | 342,683,402 | `5342aaa6e491ee44281532188b2f83be8a44837414e2d42db2a76aae80ac6207` |
| 공개 ONNX | zip 안 `siglip_base_patch16_224_vision_encoder.onnx` → `~/siglip2_work/g5c/onnx/` | 369,335,785 | `d95f3e5f0240d6776ba6338b4cb2f6bbc856287c570255d94c0f195a600000a0` |
| 공개 .alls (v5.4.0 태그) | 이 폴더 `zoo_v540_siglip_b_16_image_encoder.alls` (G5 사본과 같음) | 1,145 | `8e31c5fb23e747de71c6cf007ac8b3432e566e7e5d3ae37a543d95c0cbc45a73` |
| yaml (v5.4.0) | `cfg/networks/siglip_b_16_image_encoder.yaml` → `~/siglip2_work/g5c/zoo/` | | `81e889c2…` (base `clip.yaml` `4029daa6…`) |
- 공개 ONNX: opset 14, 노드 642개, initializer 202개. 입력 `input.1` float32 [1,3,224,224] (정규화 후 값). 출력 `key` [1,196,768], `1725` [1,768] (`/head/Gather`). yaml대로 `input.1` → `'1725'`만 parse한다. LayerNorm은 ONNX에서 풀려 있다 (ReduceMean 52, Pow·Sqrt 26).
- yaml: base `clip.yaml`의 `normalization_params`(CLIP 평균·표준편차)는 .alls 첫 줄 `norm1 = normalization([127.5]*3, [127.5]*3)`이 대신한다. 우리도 .alls를 그대로 쓴다.

## 3. 구성 (결과 전 고정)
| 이름 | ONNX | .alls | 보정 64장 |
|---|---|---|---|
| **R-our** | 공개 | 공개 (그대로) | 우리 보정 풀(train highres, 시드 0) 앞 64장을 원본 픽셀로 = G5b와 같은 파일 `calib_pix64_s0_nhwc.npy` (sha256 `cf73051f…`) |
| **R-coco** | 공개 | 공개 (그대로) | Zoo와 같은 COCO val2017 근사 64장 (아래) |
| 기준 | — | — | 공개 HEF |
- **R-coco 보정:** Zoo의 `coco_clip_calib2017.tfrecord`는 S3에서 403이다 (`…/coco/2025-07-02/…`, `…/models_files/coco/2025-07-02/…` 둘 다). Zoo v5.4.0 `datasets/create_clip_vision_tfrecord.py calib`를 보면, 이 파일은 COCO val2017/*.jpg 전부를 **glob 순서로**(섞지 않음) 원본 크기 uint8로 담는다. 보정 때 `clip` 전처리(`classification_preprocessing.clip`)가 PIL bicubic으로 224×224에 바로 맞추고(crop 없음, 0~255), DFC는 앞 64장을 쓴다. Hailo 쪽 glob 순서는 알 수 없어서 **파일 이름 정렬 순서의 앞 64장**으로 근사한다 (`g5c_coco_calib.py`). 사진 64장이 Zoo와 다를 수 있다는 한계를 결과에 같이 적는다.
- 컴파일: `g5c_dfc.sh` (G5b `g5b_dfc.sh`와 같은 흐름) — DFC 5.4.0, `--hw-arch hailo10h`, `flock ~/siglip2_work/dfc.lock`, `CUDA_VISIBLE_DEVICES=-1`, 보정 64장 로그 assert, 단계마다 시간·최대 RSS.

## 4. 잴 것과 판정 (결과 전 고정)
### 4-1. 측정
1. 구조: `hailortcli parse-hef` (AGX) — 컨텍스트 수, HEF 크기, 입력·출력 형식·quant_info.
2. 속도: AGX에서 공개·R-our·R-coco HEF를 한 실행에서 번갈아, 배치 1·8 각 3세션 (`G5b_vendor_recipe/run.sh` 방식, `g5_hailo_bench.py`, 워밍업 20·200회, 64.6 °C 미만 시작, 다른 Hailo 프로세스 0개). 보고 값 = 세션별 mean_ms의 3세션 중앙값. 보조 run2 -t 10 FPS.
3. 출력 일치: dev 스크리닝 6×200 (고화질 5종 + CIFAR A, 각 앞 200장)을 각 HEF로 실칩 실행 (`G5b_vendor_recipe/g5b_hailo_chip.py eval --k 200`, 세 HEF 모두 `--pixel-models` = 원본 픽셀 입력).
   - 우리 HEF vs 공개 HEF 사진별 임베딩 cos (평균·최소)
   - 각 HEF vs SigLIP 1세대 FP32 (공개 ONNX를 ORT FP32 `ORT_ENABLE_ALL`로, 같은 입력) 임베딩 cos — 고화질 5종 평균, CIFAR A 따로
   - (가능하면) top-1: SigLIP 1세대 텍스트 인코더(google/siglip-base-patch16-224, FP32)로 클래스 임베딩을 한 번 만들어 계산. 30분 안에 안 되면 cos만 쓴다.
- 입력: 세 HEF·FP32 ONNX 모두 같은 dev 캐시 사진(SigLIP2용 전처리 후 정규화 값, `x·127.5+127.5`로 원본 픽셀)을 쓴다. SigLIP 1세대도 정규화 평균·표준편차가 0.5라 값의 범위는 맞다. resize 방식은 HF SigLIP 1세대 공식 전처리와 조금 다를 수 있지만, 비교하는 쪽 모두에 같은 입력이 들어가므로 HEF끼리·HEF↔FP32 비교는 공정하다.

### 4-2. 판정 (기록용, R-our vs 공개 HEF)
- **"재현됨"** = 컨텍스트 수 같음 AND 크기 차이 ±2% 이내 AND 배치 1 지연 차이 ±3% 이내 AND FP32 대비 고화질 cos 차이(우리 − 공개) ±0.005 이내.
- **"구조·속도만 재현"** = 앞의 셋은 만족하고 cos 조건만 실패 → 보정 데이터 차이로 해석하고, R-coco 결과로 확인한다.
- **"재현 안 됨"** = 구조나 속도 조건 실패 → .alls 밖에 다른 설정이 있을 수 있다. 보고서에서 G5b는 "벤더 .alls를 우리가 적용한 HEF"로만 표기한다.
- 크기·지연 차이는 공개 HEF 대비 비율 `(우리 − 공개) / 공개`다. R-coco도 같은 4개 조건을 표에 같이 적는다 (판정 문구는 R-our 기준).
- 모든 결과 행에 커밋 해시·dirty·DFC/HailoRT 버전·온도를 남긴다. 같은 오류가 2번 나면 멈추고 보고한다.

## 5. 결과 (10/19 20:04~22:48, 모두 실행한 값)

### 5-1. 컴파일 (DFC 5.4.0, 데스크탑 WSL, `logs/R-*/run_info.txt`)
| 구성 | 커밋 · dirty | 보정 sha256 | parse | optimize (RSS) | compile (RSS) | HEF B | sha256 |
|---|---|---|---|---|---|---|---|
| R-our | 3df20d9 · 0 | `cf73051f…` | 16초 | 4분 02초 (10.0 GB) | 53분 46초 (8.2 GB) | 99,168,256 | `e21fbb12fc8bdb209d021ea364a79ce4d4614f011c53e4df6711cdc024a86c4a` |
| R-coco | 1de7bf9 · 0 | `f3be15bf…` | 12초 | 3분 49초 (9.9 GB) | 53분 27초 (7.8 GB) | 98,922,496 | `f16b9355bce9f3ed19e46dd68fe1ffc9ada613c03c128d14658d8112875cccd3` |
- 공개 .alls를 **한 줄도 고치지 않고** 두 구성 모두 통과했다. 보정 64장 로그 assert도 통과했다. 경고는 "optimization_level=0 권장 안 함" 1개뿐이다.
- 두 구성 모두 bucket 3개, 컨텍스트 6 + 5 + 6 = 17.
- R-coco 보정: COCO val2017 zip (815,585,330 B, sha256 `4f7e2ccb…`)을 파일 이름 순서로 놓고 앞 64장을 썼다 (`000000000139.jpg` ~ `000000006723.jpg`, 목록 `logs/coco_calib64_nhwc.files.json`).
- HEF 위치 (커밋 안 함): 데스크탑 `~/siglip2_work/g5c/R-{our,coco}/comp/siglip_b_16_image_encoder.hef`, AGX `~/ssd/junyong/hailo_g5c/R_{our,coco}.hef`.

### 5-2. 구조 (`hailortcli parse-hef`, HailoRT 5.3.0, `chip/parsehef_*.txt`)
- 세 HEF 모두 같다: 컴파일러 5.4.0, **컨텍스트 17**, 입력 UINT8 NHWC 224×224×3 (quant_in 1·0 = 원본 픽셀, 정규화 내장), 출력 `ew_add26` UINT16 1×1×768.
- 크기: 공개 99,061,760 B, R-our +0.11%, R-coco −0.14%.

### 5-3. 실칩 (AGX Orin + Hailo-10H, HailoRT 5.3.0, run.sh 커밋 98c6ffa·dirty false, 다른 Hailo 프로세스 0개, `chip/run.log`)
| HEF | 크기 MB | 컨텍스트 | b1 ms | b8 ms (장/초) | run2 b1 / b8 FPS | 고화질 cos (FP32) | CIFAR A cos (FP32) | 고화질 Δtop-1 %p | CIFAR A Δ %p | vs 공개 cos: 고화질 평균 (최소) |
|---|---|---|---|---|---|---|---|---|---|---|
| **공개 HEF** | 99.1 | 17 | **31.52** | 137.67 (58.1) | 32.87 / 77.63 | **0.9738** | 0.9310 | −1.20 | −1.50 | — |
| **R-our** | 99.2 | 17 | **31.50** | 147.05 (54.4) | 32.87 / 76.82 | **0.9760** | 0.9211 | −0.90 | −3.50 | 0.9734 (0.4680) |
| R-coco | 98.9 | 17 | 32.26 | 180.71 (44.3) | 32.77 / 76.02 | 0.9738 | 0.9267 | −0.30 | −7.50 | 0.9702 (0.4863) |
- FP32 기준: 공개 ONNX를 ORT FP32로 돌린 임베딩, 같은 입력 (`refs_info.json`, 커밋 1de7bf9). 
- top-1: SigLIP 1세대 텍스트 인코더(google/siglip-base-patch16-224, FP32, HF transformers 5.19.0)로 만든 클래스 임베딩을 썼다. FP32 top-1은 CIFAR A 72.5 (Zoo 표 FP32 72.28과 가까움), pets 95.0, flowers 87.0, cars 92.5, food 83.5, caltech 94.5. 이 모두 200장 기준이라 top-1은 ±3%p 정도 흔들린다.
  - 토크나이저(SiglipTokenizer)에는 sentencepiece가 필요했다. 공용 가상환경은 건드리지 않고 scratch 폴더에만 설치해 PYTHONPATH로 썼다 (sentencepiece 0.2.2). 첫 시도는 이것이 없어서 실패했다 (`refs` 폴더의 refs_fail1.log).
- 데이터셋별 FP32 대비 cos 평균 (공개 / R-our / R-coco): cifar_A 0.9310 / 0.9211 / 0.9267, pets 0.9832 / 0.9823 / 0.9824, flowers 0.9796 / 0.9780 / 0.9781, cars 0.9627 / 0.9707 / 0.9614, food 0.9728 / 0.9770 / 0.9756, caltech 0.9705 / 0.9722 / 0.9713.
- 사진별 cos 최소 (고화질): 공개 HEF는 cars 0.47·food 0.50·caltech 0.56으로 크게 어긋난 사진이 있다. R-coco도 비슷하고 (0.50·0.66·0.63), R-our은 이런 사진이 없다 (0.91·0.94·0.89).
- 지연: 세션별 mean_ms의 3세션 중앙값. 세션별 값 — 공개 b1 31.518 / 31.518 / 31.515, R-our 31.468 / 31.504 / 31.510, R-coco 32.033 / 32.262 / 32.564. 세션 시작 온도 64.2~64.5 °C, eval 시작 62.1~64.5 °C (`chip/temps.jsonl`).
- 결과 행: `results.jsonl` (커밋·dirty·HEF sha256·quant_in·HailoRT·온도 포함). 실칩 임베딩 (커밋 안 함): 데스크탑 `~/siglip2_work/g5c/chip/eval_*/emb_*.npz` (공개 `eadadb8a…`, R-our `860c7931…`, R-coco `911a8287…`).

### 5-4. 판정 (4-2, 기록용)
| 구성 | 컨텍스트 같음 | 크기 차이 (±2%) | b1 지연 차이 (±3%) | 고화질 cos 차이 (±0.005) | 판정 |
|---|---|---|---|---|---|
| **R-our** | 예 (17 = 17) | +0.11% ✓ | −0.04% ✓ | +0.0022 ✓ | **재현됨** |
| R-coco (참고) | 예 | −0.14% ✓ | +2.36% ✓ | 0.0000 ✓ | 재현됨 |

### 5-5. 주의할 점 (숨기지 않음)
- **세션 안 측정 순서에 따라 지연이 달라 보인다.** 세션마다 공개 → R-our → R-coco 순서로 쟀다. 세 번째였던 R-coco가 배치 1·8 모두 가장 느렸다 (b1 세션마다 32.03 → 32.26 → 32.56, b8 180.7 ms). G5에서는 공개 HEF가 세 번째였는데 b1 33.34 ms·b8 179.2 ms였고, 이번에 첫 번째로 재자 31.52 ms·137.7 ms였다. 그래서 R-coco의 +2.36%와 배치 8 차이는 HEF 차이보다 순서 효과일 가능성이 크다 (원인 확인 안 함). 판정은 R-our(두 번째) 기준이고 기준 안이다.
  - 배치를 계속 흘려 넣는 run2 FPS는 세 HEF가 거의 같다 (b1 32.77~32.87, b8 76.0~77.6). 그래서 G5 보고서의 "배치 8 = 8장씩 넣고 기다리기" 값은 순서에 민감하다는 점을 같이 적어야 한다.
- **우리 HEF와 공개 HEF는 비트 단위로 같지 않다.** 사진별 cos는 R-our vs 공개 고화질 평균 0.9734이고, 이는 각 HEF와 FP32 사이 cos(약 0.974)와 비슷한 크기다. 즉 같은 .alls에서 나온 서로 다른 양자화이고, FP32와의 거리가 비슷하다.
- R-coco 보정은 Zoo 원본 tfrecord가 아니라 근사다 (사진 순서). 그래도 공개 HEF의 "크게 어긋난 사진" 패턴을 R-coco가 따라가고 R-our는 따라가지 않는다. 그래서 Zoo도 COCO 계열 보정을 썼다는 데 맞는 관찰이다 (가설).
- 입력은 SigLIP2용으로 만든 dev 캐시 사진이다 (4-1). SigLIP 1세대 공식 전처리로 만든 사진이 아니므로 top-1 절대값은 Zoo 표와 직접 비교하지 않는다.

## 6. 해석 (판정 없음)
- 공개 .alls + 공개 ONNX + DFC 5.4.0으로 우리가 컴파일한 HEF는 공개 HEF와 **구조(컨텍스트 17, 같은 입출력)·크기(±0.15%)·배치 1 속도(−0.04%)·FP32 대비 출력 품질(고화질 cos ±0.003)이 같은 수준**이다. Zoo 레시피가 .alls 밖의 숨은 설정 없이 재현된다고 본다.
- **G5b에의 적용:** 같은 방법(Zoo .alls 그대로 + DFC 5.4.0 + 우리 보정 64장)으로 만든 G5b SigLIP2 HEF는 "벤더 레시피를 벤더 수준으로 재현한 HEF"로 볼 근거가 생겼다. 단, 두 가지 단서를 함께 적는다.
  1. 보정 데이터가 Zoo(COCO)와 다르다. 이번 SigLIP 1세대에서는 FP32 대비 고화질 cos 차이가 ±0.005 안이었지만, 사진별로 크게 어긋나는 패턴은 보정 데이터에 따라 달랐다.
  2. SigLIP2 v5.0.0 레시피는 Zoo가 공개 HEF 없이 v5.1.0에서 뺀 것이다. 이번 확인은 "Zoo .alls를 우리가 적용하면 Zoo HEF와 같은 것이 나온다"까지이고, "그 SigLIP2 레시피가 Hailo가 최종으로 쓸 만하다고 본 것"이라는 근거는 아니다.
- 보고서 표기 제안: G5b = "Hailo Model Zoo v5.0.0 공개 레시피(.alls)를 그대로 적용해 컴파일한 HEF (같은 방식으로 SigLIP 1세대 공개 HEF가 재현됨, G5c)".

## 7. 실행 명령 (재현)
```bash
# 데스크탑 WSL (worktree ~/oslab_g5c)
TAG=R-our  CALIB=~/siglip2_work/g5b/calib_pix64_s0_nhwc.npy      nohup bash g5c_dfc.sh > ~/siglip2_work/g5c/R-our/nohup.log 2>&1 &
"$SIGLIP2_PYTHON" -I g5c_coco_calib.py --dir ~/siglip2_work/g5c/coco/val2017 --n 64 --out ~/siglip2_work/g5c/coco/coco_calib64_nhwc.npy
TAG=R-coco CALIB=~/siglip2_work/g5c/coco/coco_calib64_nhwc.npy nohup bash g5c_dfc.sh > ~/siglip2_work/g5c/R-coco/nohup.log 2>&1 &
PYTHONPATH=<sentencepiece 설치 폴더> "$SIGLIP2_PYTHON" g5c_siglip1_refs.py --onnx ~/siglip2_work/g5c/onnx/siglip_base_patch16_224_vision_encoder.onnx --s2dir ~/oslab_m2rot/siglip2 --out ~/siglip2_work/g5c/refs
# AGX ~/ssd/junyong/hailo_g5c (zoo_public.hef → ../hailo_g5/siglip_b_16_image_encoder.hef, R_our.hef, R_coco.hef, run.sh, g5b_hailo_chip.py, g5_hailo_bench.py, s2_hailo_temp.py)
nohup bash run.sh > run.out 2>&1 < /dev/null &
# 데스크탑: out/ → ~/siglip2_work/g5c/chip/
"$SIGLIP2_PYTHON" -I g5c_summarize.py --chip ~/siglip2_work/g5c/chip --refs ~/siglip2_work/g5c/refs --cache $SIGLIP2_ARTIFACTS/cache --sizes ~/siglip2_work/g5c/sizes.json --out results.jsonl
```
