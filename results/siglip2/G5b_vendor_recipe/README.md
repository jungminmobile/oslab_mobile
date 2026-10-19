# G5b: Zoo v5.0.0 SigLIP2-B/16 "벤더 레시피"를 직접 컴파일해 우리 Hailo 최종(M2rot2)과 비교 (설계 2026-10-19)

> 상태: **완료 (10/19 12:51).** 1~4절은 결과 전에 고정했다 (커밋 9cafd8c). 5~6절은 결과·해석이다. 기준선 측정이다. **채택 판정 없음**, 기존 모델·best 설정은 바꾸지 않는다.
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

## 5. 결과 (10/19, 모두 실행한 값)

### 5-1. 컴파일 (DFC 5.4.0, 데스크탑 Intel Ultra 9 285K, 30 GB, 커밋 9cafd8c, dirty 0, `logs/run_info.txt`)
| 단계 | rc | 시간 | 최대 RSS |
|---|---|---|---|
| parse | 0 | 12초 | 4.5 GB |
| optimize | 0 | 3분 38초 | 9.2 GB |
| compile | 0 | 44분 27초 | 9.8 GB |
- **v5.0.0 .alls는 DFC 5.4.0에서 한 줄도 고치지 않고 통과했다** (경고는 "optimization_level=0은 권장하지 않음" 1개).
- 보정: 로그 "Using dataset with 64 entries for calibration" 3번, 모두 64 → assert 통과. 보정 파일 sha256 `cf73051f…`, 원본 풀 `91f551e3…`. 풀 값이 원래 uint8 격자라서 픽셀로 되돌릴 때 반올림 오차는 0이었다.
- 층 정밀도 (optimized HAR): 305층. `conv_feature*` 규칙에 맞은 층은 38개 (`conv_feature_splitter*` = QKV를 나누는 conv) → a8_w8. 나머지 conv 40개는 a16_w16, LayerNorm 분해 층(reduce_mean·ew_sub·feature_multiplier·reduce_sum·ew_mult 각 26개)도 모두 16비트, attention matmul·softmax는 8비트. `ew_add*` 규칙에 맞는 층은 0개였다 (덧셈이 다른 층에 합쳐짐).
- compile은 bucket 3개로 나눴다: 컨텍스트 9 + 10 + 9 = **28**.
- HEF: 데스크탑 `~/siglip2_work/g5b/comp/siglip2_b_16_image_encoder.hef` = AGX `~/ssd/junyong/hailo_g5b/vendor_v500.hef`, 164,384,768 B, sha256 `afa356b5bc4a904cad7fbe0b23a1d75325de19180fcac09360cf6a22a5e6a000`.

### 5-2. 구조 (`hailortcli parse-hef`, HailoRT 5.3.0, `logs/out_parsehef_*.txt`)
| HEF | 크기 | 컨텍스트 | 입력 | quant_in (scale, zp) | 정규화 내장 | 출력 |
|---|---|---|---|---|---|---|
| M2rot2_D256L1 | 110.4 MB | 21 | UINT8 NHWC 224×224×3 | (1/127, 128) | 아니오 (호스트에서 정규화) | ew_add57 UINT16 1×1×768 |
| 벤더 v5.0.0 레시피 | 164.4 MB (+48.9%) | 28 | UINT8 NHWC 224×224×3 | (1, 0) | **예** (원본 픽셀 입력) | conv53 UINT16 1×1×768 |
- FP32 ONNX(371.7 MB) 대비 크기: M2rot2 −70.3%, 벤더 −55.8%.

### 5-3. 실칩 (AGX Orin + Hailo-10H, HailoRT 5.3.0, run.sh 커밋 154369e, dirty false, 다른 Hailo 프로세스 0개, `chip/run.log`)
| HEF | 크기 MB | 컨텍스트 | b1 ms | b1 FPS | b8 ms | b8 FPS | 고화질 Δtop-1 %p [95% CI] | 고화질 cos | CIFAR A Δ %p |
|---|---|---|---|---|---|---|---|---|---|
| **M2rot2_D256L1** | 110.4 | 21 | 47.55 | 21.03 | 242.83 | 32.94 | −0.30 [−1.10, +0.50] | 0.9864 | −0.50 |
| 벤더 v5.0.0 레시피 | 164.4 | 28 | 99.05 | 10.10 | 625.30 | 12.79 | −1.40 [−2.80, −0.10] | 0.9342 | −18.00 |
| 벤더 − M2rot2 | +48.9% | +7 | +108.3% | −52.0% | +157.5% | −61.2% | −1.10 [−2.50, +0.30] | −0.0522 | −17.50 |
- 정확도: dev 스크리닝 6×200 (각 앞 200장), Δ는 FP32 대비 (같은 사진의 FP32 ONNX 기준 임베딩). 고화질 = pets·flowers·cars·food·caltech 평균. 두 HEF의 고화질 예측 일치율 93.3%.
  - 데이터셋별 top-1 (M2rot2 / 벤더 / FP32): cifar_A 74.0 / 56.5 / 74.5, pets 91.5 / 93.0 / 93.5, flowers 86.5 / 86.5 / 85.0, cars 93.5 / 91.0 / 93.5, food 86.5 / 82.5 / 87.5, caltech 95.0 / 94.5 / 95.0.
  - 데이터셋별 cos (M2rot2 / 벤더): cifar_A 0.9615 / 0.8475, pets 0.9899 / 0.9636, flowers 0.9881 / 0.9557, cars 0.9864 / 0.9416, food 0.9830 / 0.8977, caltech 0.9844 / 0.9122.
  - 확인: M2rot2 고화질 cos 0.9864는 G3RQ 에뮬레이터 스크리닝 값 0.9866과 거의 같다. 벤더 HEF는 quant_in (1, 0)이 기록돼 원본 픽셀 입력이 맞게 들어갔다.
- 지연: 세션별 mean_ms의 3세션 중앙값. 세션별 mean_ms — M2rot2 b1 47.57 / 47.55 / 47.53, b8 242.83 / 242.36 / 243.94; 벤더 b1 99.27 / 99.05 / 98.87, b8 625.34 / 625.30 / 624.97. median_ms의 3세션 중앙값: M2rot2 b1 47.56 · b8 250.01, 벤더 b1 95.99 · b8 625.02.
  - 벤더 b1은 분포가 두 덩어리다 (median 96.0, p90 109.0, 최대 109.4). M2rot2 b1은 47.4~48.0 사이로 좁다.
  - M2rot2 b1 47.55 ms는 G3RQ repro(47.95 ms)와 0.8% 차이다.
- 보조 `hailortcli run2 -t 10` FPS: M2rot2 b1 21.58 · b8 41.62, 벤더 b1 10.49 · b8 15.30.
- 세션 시작 온도 max(ts0, ts1): 모든 세션 64.23~64.53 °C (기준 64.6 °C 미만), `chip/temps.jsonl`.
- 결과 행 (커밋·dirty·HEF sha256·quant_in·HailoRT·온도 포함): `results.jsonl`. 실칩 임베딩(커밋 안 함): 데스크탑 `~/siglip2_work/g5b/chip/eval_*/emb_*.npz` (M2rot2 `e970f12c…`, 벤더 `8233a5d5…`).

### 5-4. 진행 중 생긴 일 (숨기지 않음)
- 첫 실칩 실행(11:48)은 run.sh가 M2rot2 eval에도 `--pixel-models vendor_v500`을 넘겨 assert에서 멈췄다 (`g5b_hailo_chip.py`가 단일 모델 호출에서 이름이 없다고 거부). run.sh를 고치고(커밋 154369e) 11:51에 처음부터 다시 돌렸다. 첫 실행 출력은 AGX `hailo_g5b/aborted_1148/`에 두었고 결과에 쓰지 않았다. 그때 끝까지 돈 벤더 eval 임베딩은 재실행 결과와 같았다.
- `hailo parse-hef`는 DFC CLI에 없어서 AGX의 `hailortcli parse-hef`를 썼다. AGX의 sha256sum에는 `-L` 옵션이 없어 run.sh에서 뺐다 (커밋 77239dc).
- 사전 확인 ONNX 비교는 요청의 64장 대신 66장(6종 × 11)을 썼다.

## 6. 해석 (판정 없음, 기술만)
- **이번 조건에서는 M2rot2가 세 축 모두 벤더 레시피보다 낫다**: 크기 −33%(벤더 대비), 배치 1 지연 절반 이하(47.6 vs 99.1 ms), 배치 8 처리량 2.6배, 고화질 cos +0.052. 고화질 Δtop-1 차이(−1.10%p)는 CI가 0을 포함한다 (200장 × 5종이라 작은 차이는 가려낼 수 없음). CIFAR A는 차이가 크다 (−17.5%p).
- 벤더 레시피가 느린 이유 가설 (확인 안 함):
  1. **16비트 층이 많다 (주 가설).** QKV(8비트)를 뺀 모든 conv(o-proj·fc1·fc2·헤드)와 LayerNorm 분해 층 전부가 a16_w16이다. 16비트 연산은 자원을 더 쓰고, 컴파일 로그에서도 단일 컨텍스트 시도 때 16비트 SC가 181/45로 가장 크게 넘쳤다.
  2. 컨텍스트 수 차이(+7)로 설명되는 몫은 작다. G5에서 잰 값(M2rot2 − rothad2 = +1.63 ms, 컨텍스트 +4 → **컨텍스트당 약 0.41 ms**, `siglip2/g5-zoo-compare` 18702f0 `G5_zoo_compare/README.md` 108줄)으로 7개 ≈ **+2.9 ms**인데, 실측 차이는 +51.5 ms다. **그래서 차이의 대부분(약 48 ms)은 16비트 층이 많은 데서 나온다고 본다 (가설, 층별 시간은 재지 않음).**
     - (10/19 수정) 처음에는 "컨텍스트당 약 2 ms → +14 ms"로 적었다. 2 ms는 G5 설계 때의 옛 어림값이고, G5 실측(0.41 ms)이 나온 뒤라 실측으로 고쳤다. 결론 방향(컨텍스트 수만으로는 설명 안 됨)은 같고, 16비트 층 쪽 비중이 더 커졌다.
  3. bucket 3개로 나눈 구조와 bucket 사이 경계 비용. 벤더 b1 지연이 두 덩어리인 것도 이와 관련 있을 수 있다.
- CIFAR A가 크게 떨어진 이유 가설 (확인 안 함):
  1. 보정은 고화질 사진 64장, optimization_level=0(편향 보정·equalization 없음)이다. 32→224로 키운 저화질 CIFAR는 활성값 분포가 보정 사진과 달라서 8비트 QKV·attention에서 오차가 커질 수 있다.
  2. Zoo v5.0.0 yaml의 FP32 CIFAR-100 76.11은 적혀 있지만 양자화 후 수치는 공개되지 않았고, v5.1.0에서 레시피가 빠졌다. Hailo 쪽에서도 이 레시피를 완성본으로 보지 않았을 수 있다.
- 쓸모: 보고서에서 "벤더가 낸 공식 레시피(같은 모델·같은 DFC·같은 칩) 대비 우리 구성"이라는 기준선으로 쓸 수 있다. 채택 판정과 best 변경은 하지 않는다.

## 7. 실행 명령 (재현)
```bash
# 데스크탑 WSL (worktree ~/oslab_g5b)
source ~/siglip2_env.sh
"$SIGLIP2_PYTHON" -I g5b_onnx_check.py --vendor ~/siglip2_work/g5b/onnx/siglip2-base-patch16-224_vision_encoder.sim.onnx \
    --ours $SIGLIP2_ARTIFACTS/onnx/siglip2_image_fp32.onnx --cache $SIGLIP2_ARTIFACTS/cache --out ~/siglip2_work/g5b/onnx_check.jsonl
nohup bash g5b_dfc.sh > ~/siglip2_work/g5b/dfc_nohup.log 2>&1 &          # parse → optimize → compile (잠금 안)
# AGX (~/ssd/junyong/hailo_g5b: run.sh, g5b_hailo_chip.py, g5_hailo_bench.py, s2_hailo_temp.py, 두 HEF)
nohup bash run.sh > run.out 2>&1 < /dev/null &
# 데스크탑: AGX out/ → ~/siglip2_work/g5b/chip/ 복사 후
"$SIGLIP2_PYTHON" -I g5b_summarize.py --chip ~/siglip2_work/g5b/chip --cache $SIGLIP2_ARTIFACTS/cache --hefinfo logs/hefinfo.json --out results.jsonl
```

## 8. 후속 작업 B·C (설계 2026-10-19, 결과 전 고정)
> 같은 브랜치. 작업 A(6절 해석 수정)는 커밋 1aebd05. dev만 쓴다 (test 안 씀). 채택 판정 없음 — 아래 판정은 기록용이다.

### 8-1. 작업 B: dev 전체 실칩 평가 (벤더 vs M2rot2)
- 대상: `vendor_v500.hef`(afa356b5…), `M2rot2_D256L1.hef`(040a3ae1…). dev 전체 = 고화질 5종 4,923장(pets 990, flowers 957, cars 967, food 985, caltech 1,024) + CIFAR A 1,000장. AGX `hailo_repro/devfull/*_x.npy` (G3RQ repro와 같은 파일).
- 실행: `run_devfull.sh` — 다른 Hailo 프로세스 0개를 기다린 뒤, 같은 실행에서 M2rot2 → (냉각) → 벤더 순서로 `g5b_hailo_chip.py eval --x-npy`(--k 없음 = 전부). 입력은 7절까지와 같다 (M2rot2 정규화 값, 벤더 `--pixel-models` 원본 픽셀). 시작 온도 < 64.6 °C, nohup + 로그.
- 예전 M2rot2 dev 전체 실칩 임베딩(AGX `hailo_repro/out/eval_M2rot2_D256L1/`, 같은 HEF·같은 사진·같은 eval 코드)은 **새 실행과 출력이 같은지 확인하는 데만** 쓴다.
- 지표 (`g5b_devfull_summarize.py`, 정확도 계산은 s2_hailo_chip_eval.py `--k 0`과 같음): 고화질 Δtop-1 (FP32 대비)과 95% 구간, 고화질 cos, 데이터셋별 top-1, CIFAR A Δ. 구간과 짝 비교는 **기존 `s2_paired_ci.boot_hr`를 그대로 import**한다 (층화 짝 부트스트랩, 시드 0, 1000회).
- **판정 (기록용, 채택 아님):** M2rot2 − 벤더 고화질 Δtop-1의 95% 구간에서 하한 > 0이면 "M2rot2가 유의하게 높다", 0을 포함하면 "동률", 상한 < 0이면 "벤더가 높다".

### 8-2. 작업 C: 벤더 CIFAR A −18.0%p의 원인 (데스크탑 DFC 에뮬레이터)
- 질문: 손실이 양자화 때문인가, 입력 경로(원본 픽셀 입력·정규화 내장) 때문인가?
- 실행: `g5b_emu.py` — optimized HAR를 SDK_NATIVE(float)와 SDK_QUANTIZED로 돌린다 (s2_dfc_screen.py와 같은 `runner.infer`, NHWC, batch 8). DFC 5.4.0, `flock ~/siglip2_work/dfc.lock`, `CUDA_VISIBLE_DEVICES=-1`.
  - 벤더: `~/siglip2_work/g5b/siglip2_b_16_image_encoder_optimized.har`. 입력은 실칩 eval과 같은 변환 `x·127.5+127.5`를 반올림한 값 (칩에서는 HailoRT가 quant_in (1, 0)으로 반올림). 반올림 전후 최대 차이를 기록한다.
  - 기준: M2rot2 `~/siglip2_work/hailo_repro/opt2_M2rot.har` (HEF 040a3ae1을 만든 2차 빌드). 정규화 값 입력.
  - 사진: CIFAR A 앞 200장 + Pets 앞 200장 (대조군). 7절 실칩 6×200과 같은 사진이다.
  - 지표: 모드마다 top-1, FP32 ONNX 임베딩 대비 cos (평균·최소).
- **해석 규칙 (결과 전 고정, 벤더 CIFAR A 기준):**
  - **"float은 FP32와 같다"** = float 에뮬레이터 vs FP32 ONNX 임베딩 cos 평균 ≥ 0.999 **그리고** CIFAR Δtop-1 > −5%p. 두 조건을 모두 만족할 때만 이렇게 쓴다 (200장 top-1은 표본 잡음이 ±3%p 정도라 top-1만으로는 경계에서 흔들린다).
  - **입력 경로 문제를 의심** = float cos 평균 < 0.999 (top-1이 괜찮아도) **또는** float CIFAR Δtop-1 ≤ −5%p. 이 경우 우리 eval의 벤더 입력 생성 코드를 점검하고, 고친 뒤 실칩 CIFAR를 다시 잰다.
  - **진짜 양자화 손실** = "float은 FP32와 같다"이면서 SDK_QUANTIZED에서만 떨어지는 경우. 기록만 한다.
  - M2rot2의 두 모드 결과는 같은 사진에서의 비교 기준으로 함께 적는다.
- **양자화 에뮬레이터 vs 실칩 (추가 확인):** CIFAR A 200장(Pets 200장도 같이 기록)에서 SDK_QUANTIZED와 실칩(7절 실행, 같은 HEF·같은 사진)을 비교한다. 지표는 예측 일치율과 사진별 임베딩 cos(평균·최소).
  - "거의 같다" = 일치율 ≥ 90% **그리고** 임베딩 cos 평균 ≥ 0.99. 근거: 예전 G3Q/G3R 기록에서 같은 HEF의 에뮬레이터↔실칩 고화질 일치율이 0.94~0.975였다.
  - 벤더가 "거의 같다"를 못 넘고 M2rot2는 넘으면, 차이의 원인은 양자화가 아니라 **실칩 입력 처리(uint8 변환, 입력 순서)**에 있다고 기록하고, 입력 코드를 점검한다.

### 8-3. 기록
- 결과 행에 커밋·dirty·DFC/HailoRT 버전·온도·HEF/HAR sha256을 남긴다: B는 `results_devfull.jsonl`, C는 `logs/emu.jsonl`.
- 같은 오류가 2번 나면 3번째 시도 전에 멈추고 보고한다.

## 9. 후속 결과
(아직 없음)
