# G5b: Zoo v5.0.0 SigLIP2-B/16 "벤더 레시피"를 직접 컴파일해 우리 Hailo 최종(M2rot2)과 비교 (설계 2026-10-19)

> 상태: **설계 고정 (1~4절, 컴파일·실칩 결과 전).** 기준선 측정이다. **채택 판정 없음**, 기존 모델·best 설정은 바꾸지 않는다.
> 브랜치 `siglip2/g5b-vendor-recipe` (`siglip2/g5-zoo-compare` 9c1ff74에서 만듦 — G5 측정 스크립트를 재사용하려고). worktree `~/oslab_g5b`. 담당: 용 (용 요청).
> 이 실험은 dev 스크리닝만 쓴다. test는 쓰지 않는다.

## 1. 질문과 대상
- 질문: Hailo가 Zoo v5.0.0에 넣었다가 v5.1.0에서 뺀 SigLIP2-B/16 이미지 인코더 레시피를 그대로 DFC 5.4.0으로 컴파일하면, 우리 M2rot2와 비교해 정확도·크기·속도가 어떤가?
- 같은 원본 모델(google/siglip2-base-patch16-224)이라 정확도도 직접 비교할 수 있다 (G5의 SigLIP 1세대와 다른 점).

| 이름 | 무엇 | 파일 (커밋하지 않음) | 크기 B | sha256 |
|---|---|---|---|---|
| **M2rot2_D256L1** | 우리 Hailo 최종 (D-043, 2차 빌드) | AGX `~/ssd/junyong/hailo_repro/M2rot2_D256L1.hef` | 110,436,352 | `040a3ae1ebd81d9979ec893b5a65cf1265370196e8cfd516350e016191752c6f` |
| 벤더 ONNX zip | Hailo export, AGX `~/siglip2-base-patch16-224_vision_encoder.zip` → 데스크탑 `~/siglip2_work/g5b/dl/` | | 342,829,181 | `9fdd14ab03f70b900f74d91f1b6a6ab7d76ca0c44e40bd92ac363181be8765b4` |
| 벤더 ONNX | zip 안 `siglip2-base-patch16-224_vision_encoder.sim.onnx` → `~/siglip2_work/g5b/onnx/` | | 369,302,595 | `a23f2a7d54bd962d9f8765e91df6fcc5376d6e33492934086e67d02591468f33` |
| 우리 FP32 ONNX | `$SIGLIP2_ARTIFACTS/onnx/siglip2_image_fp32.onnx` (Release siglip2_image_fp32.onnx) | | 371,672,881 | `333cad08e68dc85a5036f8f7396bc61d5548eb69ea365a0c5d83ff15c48840e0` |
| 벤더 레시피 .alls | Zoo v5.0.0 `siglip2_b_16_image_encoder.alls` (요청에 적힌 7줄 그대로) | 이 폴더 `siglip2_b_16_image_encoder_v500.alls` | | |

## 2. 사전 확인 (1단계, 이미 실행함 — 설계 전에 끝난 유일한 결과)
- 벤더 ONNX: opset 17, 노드 434개, initializer 200개. 입력 `onnx::Cast_0` float32 [1,3,224,224] (배치 1 고정, 정규화 후 값 — 첫 노드가 입력에 바로 patch Conv). 출력 2개: `key` [1,196,768], `1393` [1,768] (`/model/head/Gather`, MAP 헤드 뒤 pooled 출력). yaml의 끝 노드 `'1393'`만 쓴다.
- 우리 ONNX: opset 17, 노드 791개 (Constant 248개 등 shape 계산 노드가 많음, simplify 전), 입력 `image` [batch,3,224,224], 출력 `embedding` [batch,768].
- 같은 모델 확인 (`g5b_onnx_check.py`, ORT FP32 `ORT_ENABLE_ALL`): dev 캐시 6종 × 앞 11장 = 66장, 같은 정규화 입력. **cos 최소 0.9999998, 평균 1.0000000, 원래 출력 최대 절대 차이 9.5e-6** → 통과 (기준 0.999). 결과 `onnx_check.jsonl` (데스크탑 `~/siglip2_work/g5b/`).
  - 요청은 "64장"이었고 6종에 고르게 나누려고 66장을 썼다.

## 3. 방법 (결과 전 고정)
### 3-1. 컴파일 (`g5b_dfc.sh`)
- DFC 5.4.0, `--hw-arch hailo10h`, WSL Ubuntu-22.04, `CUDA_VISIBLE_DEVICES=-1`, 모든 단계 `flock ~/siglip2_work/dfc.lock` 안.
- parse: `--start-node-names onnx::Cast_0 --end-node-names 1393 -y`.
- optimize: v5.0.0 .alls 그대로. 보정 = 우리 보정 풀(`calib_highres256_s0_nhwc.npy`, train highres 시드 0, 256장)의 **앞 64장**을 원본 픽셀로 되돌린 것: `round(x·127.5 + 127.5)`, 0~255 (`g5b_calib.py`). .alls 첫 줄이 정규화를 HEF 안에 넣기 때문이다. 로그의 "Using dataset with N entries for calibration"이 모두 64인지 assert한다.
- compile: `hailo compiler ... --hw-arch hailo10h` (G3RQ와 같은 명령).
- 단계마다 `/usr/bin/time -v`로 시간·최대 RSS 기록.
- **5.4 문법 오류 규칙:** 에러가 나면 바꾼 줄과 이유를 5절에 적고 최소한으로만 고친다. 같은 오류가 2번 나면 멈추고 보고한다. 컴파일이 실패하면 그것도 결과로 적는다.

### 3-2. 구조 (3단계)
- 두 HEF에 `hailo parse-hef`: 크기, 컨텍스트 수, 입력 형식·quant_info (정규화 내장 여부).

### 3-3. 정확도 (4단계, 실칩, 간단 버전)
- dev 스크리닝 6×200: cifar_A, pets/flowers/cars/food/caltech_dev 각 앞 200장 (G3RQ M2rot 스크리닝·`s2_hailo_chip_eval.py --k 200`과 같은 사진·같은 기준).
- 실칩 임베딩: `g5b_hailo_chip.py eval` = `s2_hailo_chip.py eval`에 (가) `--k`(앞 k장만), (나) `--pixel-models`(원본 픽셀 입력 모델)만 더한 것. 두 옵션 모두 기본 꺼짐.
  - M2rot2: 지금과 같다 — 정규화된 float32를 HailoRT FLOAT32 입력으로 넣고 HailoRT가 quant_info(1/127, 128)로 uint8로 바꾼다.
  - 벤더 HEF: 같은 사진의 정규화 값을 `x·127.5 + 127.5`로 되돌려 float32로 넣는다 (HailoRT가 quant_info대로 uint8로 바꿈; scale 1·zp 0이면 원본 픽셀). 실제 quant_info는 결과에 기록한다.
  - 두 HEF를 **같은 run.sh 실행**에서 잰다.
- 지표 (`s2_hailo_chip_eval.py --k 200`, FP32 기준 = 캐시 `<set>_ref_fp32.npy`): 고화질 5종 평균 Δtop-1 (FP32 대비, %p), 고화질 cos (FP32 대비 평균), CIFAR A Δtop-1. 고화질 Δtop-1에는 짝 부트스트랩 95% CI(같은 스크립트)를 같이 적는다. 벤더 − M2rot2 차이도 같은 방식으로 적는다 (참고, 판정 없음).

### 3-4. 속도 (5단계, G3RQ repro `chip/run.sh`·G5 `run.sh` 방식)
- AGX Orin + Hailo-10H, HailoRT 5.3.0. **G5 세션이 끝나고 다른 Hailo 프로세스가 0개일 때만 시작한다** (run.sh가 확인하고 기다림, 최대 3시간).
- 온도: 세션 시작 전 max(ts0, ts1) < **64.6 °C** (쉬는 온도 62.6 + 2, G3RQ·G5와 같음). 30초마다, 최대 90분.
- `g5_hailo_bench.py` (G5와 같은 파일, 워밍업 20, 200회), **배치 1 세션 3번, 배치 8 세션 3번**, 세션 안에서 M2rot2 → 벤더 순서로 한 번씩 (번갈아). 호스트 전처리는 타이머 밖.
- 보고 값 = 세션별 mean_ms의 3세션 중앙값 (median_ms의 중앙값도 같이). 배치 8 처리량 = 8000 / mean_ms.
- 보조: `hailortcli run2 -t 10 set-net <hef> --batch-size {1,8}` FPS.

## 4. 결과 표 형식 (결과 전 고정)
| HEF | 크기 MB | 컨텍스트 | b1 ms | b1 FPS | b8 ms | b8 FPS | 고화질 Δtop-1 %p [95% CI] | 고화질 cos | CIFAR A Δ %p |
- 모든 행에 커밋 해시·dirty·DFC/HailoRT 버전·세션 시작 온도를 남긴다 (`results.jsonl`). AGX에는 저장소가 없으므로 스크립트 sha256을 run.log에 남긴다.
- 해석은 기술만 한다 (어느 쪽이 더 빠른지·정확한지, 차이가 컨텍스트 수·16비트 층 비율로 설명되는지에 대한 가설). **채택 판정은 하지 않는다.**

## 5. 결과
(아직 없음)
