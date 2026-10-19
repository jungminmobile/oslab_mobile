# G5c: Zoo 공개 .alls로 공개 HEF가 재현되는가 — SigLIP 1세대 B/16 (설계 2026-10-19)

> 상태: **설계 고정 (1~4절, 컴파일·실칩 결과 전).** 기록용 측정이다. 채택 판정 없음.
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

## 5. 결과
(아직 없음)
