# G6b: B32 M HEF가 Zoo보다 느린 원인 나누기 + 지연을 Zoo 수준으로 + B16 vs B32 ORT 비교 (설계 2026-10-20)

> 상태: **완료 (10/20).** 1~5절은 결과 전에 고정했다 (커밋 e7cd97c). 결과는 6절부터다.
> **한 줄 결론:**
> - **Hailo는 속도 목표(Z 대비 +5%)를 맞춘 구성이 없어 "실패"**다. 가장 빠른 M_both는 +11.35%(16.5 → 18.4 ms)이고, 정확도는 Z보다 +3.17%p로 유지된다.
> - 16비트를 Zoo 집합으로 줄여도 M과 동률이다 → "회전 + sink로 16비트 축소 가능". resources_param은 출력을 바꾸지 않았고(비트 일치에서 추정), 지연은 M에서 출발하면 −12.3%, M_z16 위에서는 −4.3%만 줄였다 (두 효과가 많이 겹친다).
> - ORT에서 B32 INT8은 B16 M2rot_ort보다 0.46배 지연(스레드 4)이고, 절대 top-1은 −2.34%p [−3.00, −1.68]다.
> 브랜치 `siglip2/g6b-speed` (`siglip2/g6-zoo-transfer` 7b3b934에서 만듦), worktree `~/oslab_g6b`. 담당: 용 (용 요청). **test는 쓰지 않는다.**
> 배경: `results/siglip2/G6/README.md`. B32 배치 1 지연은 Z 16.49 ms, V 16.55, D 20.48, R 20.28, M 21.16. 늘어난 양의 대부분이 D − V 묶음에서 생긴다.

## 1. 질문

1. B32 M의 지연 증가 중 어느 만큼이 (a) 추가 16비트 층, (b) Zoo의 resources_param 제거 때문인가?
2. 회전 + sink로 거대 활성값을 없앴으니, 16비트를 Zoo 집합으로 줄여도 정확도가 유지되는가?
3. 지연이 Zoo 수준이면서 정확도가 Zoo보다 높은 구성이 있는가?
4. (판단 자료) B16 유지 vs B32 전환: Hailo와 ORT CPU에서 정확도·지연·크기 비교

## 2. Hailo 구성 (B32, 모두 G6 M과 같은 ONNX `a0eb502b…`·보정 256장(Zoo 전처리, 시드 0)·optimization_level 1·LN 분해 등 같은 줄)

| 이름 | 16비트 conv | resources_param | 컴파일 |
|---|---|---|---|
| M (G6, 다시 컴파일 안 함) | `conv4, conv5, conv24, conv25, conv41` (블록 0 fc1·fc2, 블록 5 fc1·fc2, 블록 9 fc2) | 없음 | — |
| **M_rb** | M과 같음 | Zoo B32 줄 `resources_param(max_compute_utilization=1.0, max_control_utilization=0.95, max_memory_utilization=0.75)` | 1 |
| **M_z16** | Zoo B32 집합을 옮긴 것: `conv3`(블록 0 attention 출력 투영), `conv4, conv5`(블록 0 fc1·fc2), `conv25`(블록 5 fc2) | 없음 | 2 |
| **M_both** | M_z16과 같음 | Zoo B32 줄 | 3 |
| Z (G6) | Zoo 공개 HEF `c1612329…` | | — |
| **B16 M2rot2** (참고, 다시 컴파일 안 함) | 우리 B16 Hailo 최종 (D-043, G5 `040a3ae1…`, 110.4 MB·21 ctx) | | — |

- 세 구성 모두 `norm1`·`conv1`·`format_conversion1`과 잔차 덧셈 26개는 16비트다 (G6 6-2 대응표와 같은 층).
- **Zoo `{ew_add*}`의 이식 (용 확인):** Zoo B32 그래프에서는 ew_add 26개가 모두 잔차 덧셈이다. 그래서 M 그래프에서는 마스크 연산 덧셈 27개를 뺀 **잔차 덧셈 26개**로 옮긴다 (M과 같은 목록).
- **Zoo 집합 대응 (G6 대응표 기준):** Zoo `conv3` = `/model/encoder/layers.0/self_attn/out_proj`, M 그래프 `conv3` = `/trunk/blocks.0/attn/proj`. `conv4·5·25`도 같은 블록·같은 층이다.
- **context switch(bucket):** Zoo B32 v5.4.0 .alls에는 bucket 줄이 없다 (S1B16 .alls에만 있음). 그래서 M_rb에 옮길 bucket 줄은 없고, 층 이름 대응도 필요 없다.
- .alls 3개는 결과 전에 커밋하고, dev를 본 뒤 고치지 않는다. 컴파일 실패 때만 최소한으로 고치고 기록한다.

## 3. 측정

### 3-1. Hailo (AGX Orin + Hailo-10H, G6 `g6_hailo_chip.py`·`run.sh` 최종판)

- **M_rb 비트 확인:** 양자화 설정이 M과 같으므로, M_rb와 M의 실칩 출력을 dev 앞 200장(6종 × 200)으로 비교한다. **비트 단위로 같으면** M_rb의 dev 전체 eval은 생략하고 M 값을 쓴다. 다르면 dev 전체를 잰다.
- **정확도:** M_z16·M_both를 dev 전체(고화질 4,923장 + CIFAR A 1,000장, G6와 같은 Zoo 전처리 캐시·HF FP32 기준·분류기)로 실칩 eval한다.
  - 쌍 비교 95% 구간 (`s2_paired_ci.boot_hr`): M_z16−M, M_both−M, M_both−Z, M_z16−Z, (M_rb가 M과 다르면) M_rb−M, M_rb−Z
- **B16 M2rot2 정확도 (용 지시):** G5b dev 전체 결과는 사진은 같지만 eval 코드(`g5b_hailo_chip.py`, 정규화 입력)와 분류기·FP32 기준 파일(timm)이 달라 **그대로 쓰지 않는다.** 대신 다음으로 같은 eval 코드(`g6_hailo_chip.py`)로 다시 잰다.
  - 데이터: G6 Zoo 전처리 224 캐시. 224에서는 timm 전처리와 Zoo 전처리가 같은 픽셀임을 기존 timm 캐시(`x·127.5+127.5`)와 비교해 확인하고 기록한다.
  - 이 HEF는 정규화가 HEF 밖이라, `g6_hailo_chip.py`가 quant_info를 보고 `x/127.5 − 1`을 넣는다.
  - FP32 기준: HF `google/siglip2-base-patch16-224` (= 우리 timm 모델, G6_pre 4-1 cos 0.99999)
  - G5b 값은 참고로만 나란히 적는다.
- **속도:** Z, M, M_rb, M_z16, M_both, B16 M2rot2를 같은 세션에서 번갈아 잰다. 배치 1·8, 3세션.
  - 세션과 HEF마다 칩 < 64.6 °C에서 시작한다. 95 °C를 넘으면 즉시 멈추고 보고한다.
  - 보조로 `run2 -t 10`과 parse-hef(컨텍스트·크기)도 낸다.
- **판정은 1번 속도 측정값으로 한다:** 배치 1 지연, 세 세션 평균.

### 3-2. ORT CPU (데스크탑, 비공식 — 공식은 정민 노트북, D-010)

- **B32 INT8 (B16 M2rot_ort 과정을 그대로 옮김, 옮긴 뒤 고치지 않음)**
  - 입력: B32 M FP32 ONNX(회전 + M2 통일 규칙 15@6:5, 7@10:9)를 **배치 가변**으로 export (G6 `s2_g6.py export --cfg m2rot`과 같은 모델, `--dynamic`만)
  - `s2_quantize.py --src <그 ONNX> --calib-source highres --calib-n 256 --seed 0 --method percentile --percentile 99.999 --act-type u8 --reduce-range --op-types MatMul --hist-chunk 16` (MatMul 전부 INT8, `--exclude` 없음). B16 M2rot_ort와 같은 옵션이다.
  - 보정 사진: 256 크기라 기존 224 timm 캐시를 쓸 수 없다. 같은 사진(highres 256, 시드 0, 같은 순서)의 **Zoo 전처리 256 캐시**(G6_pre `calib256_s0`)를 새 옵션 `--calib-npy`(기본 꺼짐)로 넣는다. 실제 보정 장수 256은 assert한다.
- **B16 M2rot_ort:** 배포 파일 `siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx`(sha256 `e84c3727…`)를 그대로 쓴다.
- **B16·B32 FP32:** B16 `siglip2_image_fp32.onnx`(Release `333cad08…`), B32 timm export(배치 가변, 회전 없음)
- **정확도:** 네 모델을 같은 eval 코드(ORT `ORT_ENABLE_ALL`)·dev 전체(Zoo 전처리, 크기는 모델에 맞게)·HF 분류기로 잰다.
  - 각 모델의 FP32 대비 Δtop-1(쌍 비교 95% 구간, 기준은 같은 모델의 ORT FP32)
  - 절대 top-1(고화질 5종 평균), CIFAR A는 따로
  - 정확도는 정수 커널(ORT 기본 설정) 값이다. CPU·VNNI·ORT 버전을 기록한다.
- **지연·메모리:** G3RQ-ORT·G8과 같은 규칙이다.
  - `--warmup 20 --n 200 --repeats 3`, 배치 1, 스레드 1·2·4 × 3세션, 네 모델을 같은 세션에서 번갈아 잰다. `OPENBLAS_NUM_THREADS=1`
  - 게이트: 측정 직전 WSL 안 다른 python·hailo 0개, 1분 평균 부하 < 2. Windows 쪽 부하는 확인하지 못한다.
  - peak 메모리는 `s2_model_stats.py`로 잰다.
  - 256 입력 때문에 측정은 `s2_g6b_bench.py`로 한다 (5절). **DFC 컴파일이나 다른 세션의 측정이 도는 동안에는 재지 않는다.**
- **크기:** ONNX 파일 바이트

## 4. 판정 (결과 전 고정)

| 항목 | 규칙 |
|---|---|
| Hailo 속도 목표 | 배치 1 지연(3-1 속도 측정, 3세션 평균)이 **Z 대비 +5% 이내** |
| **Hailo 성공** | 속도 목표를 만족하는 구성 중 하나 이상에서 (구성 − Z)가 정확도 **"우위"**(고화질 Δtop-1 쌍 비교 95% 구간 하한 > 0). 여러 개면 정확도가 가장 높은 것을 후보로 적는다 |
| 16비트 축소 영향 | M_z16 − M이 "동률"이면 **"회전 + sink로 16비트 축소 가능"**으로 기록한다. "열위"면 "축소하면 손실", "우위"면 그대로 기록한다 |
| 쌍 비교 | 하한 > 0 → 우위, 상한 < 0 → 열위, 그 밖 → 동률 (G6와 같음) |
| ORT | 판정하지 않는다. B16 대비 B32의 지연 비율과 절대 top-1 차이를 표로 낸다 |
| B16 유지 vs B32 전환 | 판정하지 않는다. 마지막 절에 비교표만 낸다 (용 지시) |

- 이 기준은 결과를 본 뒤 바꾸지 않는다. 바꿔야 하면 이유를 적고 용의 승인을 받는다.

## 5. 구현 (결과 전 커밋, 모두 기본값이 꺼진 옵션)

- `s2_quantize.py --calib-npy <N×3×H×W float32>`: 이 파일을 보정 사진으로 쓴다 (calib_images 대신, 장수 assert)
- `s2_model_stats.py --sizes 이름=크기`: 메모리 측정 입력 크기 (기본 없음 = 224). GMACs·파라미터는 기존대로 B16 모델로만 계산한다 (B32는 기록 안 함).
- **`s2_g6b_bench.py` (새 파일):** `rn50/bench_speed.py`의 `main()`은 모든 모델에 같은 224 입력 하나를 넣는다. 그래서 256 모델(B32)을 같은 세션에 넣을 수 없다.
  공통 코드는 고치지 않고(D-009), 측정 부분(세션 `rn50/common.session`, 예열, 번갈아 측정 루프, 타이머, 결과 필드)을 그대로 옮기고 **입력만 모델별 크기**(같은 CIFAR dev A 앞 200장의 Zoo 전처리 캐시)로 바꿨다. PRD 5절 조건 assert와 `runs.jsonl`·`env.json` 기록은 `s2_bench_speed.py`와 같다.
- `g6b_ort_speed.sh`: G8 `g8_speed.sh`와 같은 게이트·순서 (세션 1~3 × 스레드 1·2·4, 측정 직전 다른 python·hailo·DFC 0개, 부하 < 2, 5분마다 재확인) + peak 메모리
- `s2_g6.py export --dynamic`: 배치 가변 ONNX (기본은 b1)
- 회귀 확인: 옵션을 모두 끄면 기존 결과와 같아야 한다.
  - quantize: B16 M2rot_ort를 같은 명령(옵션 끔)으로 다시 만들어 배포 파일 `e84c3727…`과 sha256이 같은지
  - bench: `s2_g6b_bench.py`의 측정 루프가 `rn50/bench_speed.main`과 같은지 (diff로 확인해 기록, 측정값은 비교하지 않음)
  - export: b1의 sha256이 G6와 같은지

## 6. 결과

### 6-0. 판정 (4절 규칙 그대로)

| 항목 | 결과 | 판정 |
|---|---|---|
| Hailo 속도 목표 (배치 1, Z 대비 +5% 이내) | 가장 빠른 B32 구성 M_both 18.38 ms = Z 16.51 ms 대비 **+11.35%**. M_rb +12.4%, M_z16 +16.4%, M +28.2% | **모든 구성 미달** |
| **Hailo 성공** | 속도 목표를 만족하는 구성이 없다 | **실패** (후보 없음) |
| 16비트 축소 영향 | M_z16 − M = −0.21 [−0.70, +0.30] | **동률 → "회전 + sink로 16비트 축소 가능"** |
| (참고) 정확도 | M_z16 − Z = M_both − Z = +3.17 [+2.41, +3.95] 우위. M − Z = +3.38 [+2.62, +4.17] 우위 | 정확도 이득은 유지 |

- 정리: Zoo 수준 지연(+5%)에는 이르지 못했다. 지연 증가는 M +28.2% → M_both +11.35%로 줄었다. 정확도는 Zoo보다 +3.17%p 높은 채로 남았다.
- 지연 증가 분해 (배치 1, M에서 출발): resources_param 추가로 −2.61 ms (M → M_rb), 16비트 축소로 −1.95 ms (M → M_z16), 둘 다 −2.78 ms (M → M_both, 두 효과가 겹친다). 남은 +1.87 ms(+11.35%)는 이번 구성으로 설명되지 않는다. 남은 차이로는 회전·sink 그래프 자체(마스크 연산 27개, 컨텍스트 13 vs 12), 보정 256장·level 1, ONNX 출처가 있다. 이번 설계에서는 나누지 않았다.

### 6-1. 구성표 (`build.jsonl`, `chip/parsehef_*.txt`)

| HEF | .alls (sha256 앞 12) | HEF 크기 B | 컨텍스트 | HEF sha256 (앞 16) | 보정 로그 | optimize / compile |
|---|---|---|---|---|---|---|
| B32 Z (G6) | Zoo 공개 | 102,645,760 | 12 | `c16123297d83a36c` | — | — |
| B32 M (G6) | `g6_b32_M.alls` | 112,975,872 | 17 | `311655b72aad4dac` | 256 ×4 | (G6) |
| **B32 M_rb** | `0bbc8fc39cfa` | 115,990,528 | **12** | `a2ddb2bbc4dfb5d7` | 256 ×4 | 24분 56초 / 30분 15초 |
| **B32 M_z16** | `b890b644fe8e` | 108,273,664 | 17 | `e21b99303a022c88` | 256 ×4 | 22분 41초 / 21분 12초 |
| **B32 M_both** | `7eff7a7b90b1` | 111,955,968 | **13** | `2a381857c1e767fa` | 256 ×4 | 22분 34초 / 23분 36초 |
| B16 M2rot2 (G5) | `g3r_M2rot_D256L1equiv.alls` | 110,436,352 | 21 | `040a3ae1ebd81d99` | (G3RQ) | — |

- 세 빌드 모두 ONNX `a0eb502b…`(G6 M과 같음)이고, 커밋 `e7cd97c`에서 dirty 0, DFC 5.4.0으로 빌드했다. 입력은 모두 UINT8 NHWC이고, 정규화가 HEF 안에 있다 (B16 M2rot2만 정규화가 밖).
- resources_param을 넣으면 HEF가 커졌다 (M → M_rb +3.0 MB, M_z16 → M_both +3.7 MB). 대신 컨텍스트가 줄었다 (17 → 12·13).

### 6-2. Hailo 정확도 (`results_chip.jsonl`, 실칩 dev 전체 5,923장, G6와 같은 캐시·분류기·eval 코드)

| HEF | 고화질 top-1 | FP32 대비 Δ [95% 구간] | cos 평균 / 최소 | CIFAR A (Δ) | 시작 온도 °C |
|---|---|---|---|---|---|
| B32 Z (G6) | 85.74 | −4.22 [−4.95, −3.45] | 0.9366 / −0.205 | 70.8 (−4.4) | 62.3 |
| B32 M (G6) | 89.12 | −0.84 [−1.31, −0.37] | 0.9840 / 0.381 | 74.7 (−0.5) | 62.0 |
| B32 M_rb | = M (dev 200장 × 6종 비트 일치, 3-1 규칙대로 전체 eval 생략) | | | | |
| **B32 M_z16** | 88.91 | −1.05 [−1.57, −0.52] | 0.9835 / 0.242 | 74.4 (−0.8) | 49.7 |
| **B32 M_both** | 88.91 (M_z16과 dev 전체 비트 일치) | −1.05 [−1.57, −0.52] | 0.9835 / 0.242 | 74.4 (−0.8) | 49.6 |
| B16 M2rot2 (참고) | 91.26 (FP32 91.66) | −0.41 [−0.84, +0.05] | 0.9862 / 0.262 | 75.2 (−2.0) | 49.6 |

| 쌍 | 차이 (고화질 %p) | 95% 구간 | 판정 | CIFAR A |
|---|---|---|---|---|
| M_z16 − M | −0.21 | [−0.70, +0.30] | 동률 | −0.3 |
| M_both − M | −0.21 | [−0.70, +0.30] | 동률 | −0.3 |
| M_both − Z | +3.17 | [+2.41, +3.95] | 우위 | +3.6 |
| M_z16 − Z | +3.17 | [+2.41, +3.95] | 우위 | +3.6 |
| (G6) M − Z | +3.38 | [+2.62, +4.17] | 우위 | +3.9 |

- **B16 M2rot2 재측정 확인:** 같은 eval 코드로 잰 값은 G5b의 dev 전체 값과 고화질 Δ·구간·cos가 같다 (−0.41 [−0.84, +0.05], cos 0.9862). CIFAR Δ만 G5b −1.7 대 이번 −2.0이다. 칩 top-1은 75.2로 같고, FP32 기준이 timm ORT 76.9 대 HF 77.2로 다르기 때문이다.
- 224 픽셀 확인: 기존 timm 캐시(`x·127.5+127.5`)와 G6 Zoo 224 캐시의 최대 차이는 dev 전체 6종 모두 **0**이다.
- M_rb·M_both가 각각 M(dev 200장 × 6종)·M_z16(dev 전체)과 비트 단위로 같다. 그래서 resources_param은 양자화(가중치·눈금)를 바꾸지 않고 배치만 바꾸는 것으로 **추정**한다 (HAR을 직접 비교하지는 않음).

### 6-3. Hailo 속도 (판정은 배치 1, `results_speed.jsonl`, `chip/bench/`)

AGX Orin + Hailo-10H, HailoRT 5.3.0, G6 최종 `run.sh`(스크립트 sha256 `f6439972…`·`70a5de57…`, G6와 같음).
- 세션 3번 × HEF 6개를 같은 순서로 쟀다. HEF마다 칩 < 64.6 °C에서 시작했다 (최고 시작 64.58 °C).
- 측정 중 최고 온도는 86.9 °C였고, 95 °C를 넘지 않았다. 세 세션 사이 차이(최대−최소)는 배치 1에서 모든 HEF가 평균의 0.4% 이내다. 배치 8은 M_both만 3.0%이고, 나머지는 0.8% 이내다.

| HEF | 배치 1 ms | **Z 대비** | 배치 1 p90 | run2 FPS b1 | 배치 8 ms/묶음 (Z 대비) | 배치 8 장/초 | run2 FPS b8 |
|---|---|---|---|---|---|---|---|
| B32 Z (Zoo 공개) | 16.51 | **+0.0%** | 16.68 | 64.4 | 53.66 (+0.0%) | 149.1 | 225.5 |
| B32 M (G6) | 21.16 | **+28.2%** | 21.30 | 50.9 | 87.39 (+62.8%) | 91.5 | 116.9 |
| B32 M_rb | 18.55 | **+12.4%** | 18.66 | 58.3 | 65.66 (+22.4%) | 121.8 | 175.9 |
| B32 M_z16 | 19.21 | **+16.4%** | 19.34 | 56.5 | 75.23 (+40.2%) | 106.3 | 139.9 |
| B32 M_both | 18.38 | **+11.35%** | 18.55 | 59.1 | 62.29 (+16.1%) | 128.5 | 186.3 |
| B16 M2rot2 (참고) | 47.52 | **+187.9%** | 47.61 | 21.6 | 243.23 (+353.3%) | 32.9 | 41.7 |

- G6 측정(Z 16.49, M 21.16 ms)과 이번 측정(Z 16.51, M 21.16 ms)은 0.2% 안에서 같다.

### 6-4. ORT CPU (비공식 — 이 PC: Intel Core Ultra 9 285K, AVX-VNNI, AVX-512 없음, ORT 1.25.0, WSL2; 공식은 정민 노트북)

정확도는 `ort_accuracy.jsonl` (ORT_ENABLE_ALL 정수 커널, dev 전체, HF 분류기)이다. 지연은 `results_ort_speed.jsonl`·`speed/`이고, 배치 1, 세 세션의 ms/장 중앙값이다. 게이트 대기는 3번이었고, 측정 전후 다른 python·hailo는 0개였다. 메모리는 `model_stats/`이고 추론 중 peak RSS다.

| 모델 | 고화질 top-1 | 자기 FP32 대비 Δ [95% 구간] | CIFAR A (Δ) | t1 ms | t2 ms | t4 ms | 크기 MiB | peak 메모리 MiB |
|---|---|---|---|---|---|---|---|---|
| B16 FP32 | 91.66 | — | 76.9 | 237.65 | 123.65 | 67.14 | 354.5 | 566 |
| **B16 M2rot_ort** | 91.44 | -0.23 [-0.64, +0.18] | 72.6 (-4.3) | 89.50 | 49.51 | 29.71 | 91.7 | 246 |
| B32 FP32 | 89.96 | — | 75.2 | 83.41 | 45.19 | 26.66 | 360.8 | 610 |
| **B32 INT8 (M2rot 이식)** | 89.09 | -0.87 [-1.33, -0.38] | 75.4 (+0.2) | 30.83 | 19.08 | 13.61 | 98.1 | 232 |

- B16 값은 G3RQ-ORT 기록과 같다 (FP32 91.66, M2rot_ort −0.23 [−0.64, +0.18]). eval 경로가 맞다는 확인이 된다.
- B32 INT8의 FP32 대비 손실(−0.87)은 B16(−0.23)보다 크고 구간이 0을 포함하지 않는다. CIFAR는 반대로 B32가 손실이 없다 (+0.2 vs −4.3).
- B32 INT8은 B16 M2rot_ort보다 t1 0.34배, t2 0.39배, t4 0.46배 시간이 걸린다. 토큰 64개 대 196개 때문으로 보인다 (t1 비 0.34 ≈ 64/196, 검증 안 함). 크기는 +7%, peak 메모리는 −6%다.

## 7. B16 유지 vs B32 전환 비교 (판정하지 않음, 용 지시)

FP32 기준: B16 91.66%(CIFAR 77.2, Hailo 쪽 HF FP32) / 76.9(ORT 쪽 ORT FP32). B32 89.96%(CIFAR 75.2). 고화질 5종 dev 전체다. Hailo 쪽 Δ는 HF FP32 대비, ORT 쪽 Δ는 같은 모델의 ORT FP32 대비다.

| 경로 | 모델 | 고화질 top-1 | FP32 대비 Δ [95% 구간] | CIFAR A (Δ) | 지연 | 크기 |
|---|---|---|---|---|---|---|
| Hailo | **B16 M2rot2** (지금 Hailo 최종 후보, D-043) | 91.26 | -0.41 [-0.84, +0.05] | 75.2 (-2.0) | 배치 1 47.52 ms · 배치 8 33 장/초 | 110.4 MB |
| Hailo | B32 M (정확도 최고) | 89.12 | -0.84 [-1.31, -0.37] | 74.7 (-0.5) | 배치 1 21.16 ms · 배치 8 92 장/초 | 113.0 MB |
| Hailo | B32 M_both (지연 최소) | 88.91 | -1.05 [-1.57, -0.52] | 74.4 (-0.8) | 배치 1 18.38 ms · 배치 8 128 장/초 | 112.0 MB |
| ORT CPU (비공식) | **B16 M2rot_ort** (지금 예선 후보) | 91.44 | -0.23 [-0.64, +0.18] | 72.6 (-4.3) | t1 89.5 · t4 29.7 ms | 91.7 MiB |
| ORT CPU (비공식) | B32 INT8 (M2rot 이식) | 89.09 | -0.87 [-1.33, -0.38] | 75.4 (+0.2) | t1 30.8 · t4 13.6 ms | 98.1 MiB |

- 절대 top-1은 두 경로 모두 B16이 높다: Hailo는 B16 M2rot2 − B32 M = **+2.14%p**, ORT는 B16 − B32 = **+2.34%p** (반올림 전 값의 차). B32 FP32 자체가 1.70%p 낮다.
- 지연
  - Hailo 배치 1: B32 M_both 18.38 ms, B32 M 21.16 ms, B16 47.52 ms. B32가 0.39~0.45배다.
  - ORT 스레드 4: B32 13.6 ms 대 B16 29.7 ms로 0.46배다 (스레드 1은 0.34배).
- CIFAR A는 Hailo에서 B16이 조금 높고 (75.2 vs B32 M 74.7), ORT에서는 B32가 높다 (75.4 vs 72.6). B16 ORT는 CIFAR 손실이 크다 (−4.3).
- 같은 사진이라 짝 비교 구간을 낼 수 있다 (검토 뒤 추가, `b16_vs_b32_pairs.jsonl`, 판정 아님). Hailo B16 M2rot2 − B32 M = **+2.14 [+1.48, +2.84]%p** (CIFAR +0.5). ORT B16 M2rot_ort − B32 INT8 = **+2.34 [+1.68, +3.00]%p** (CIFAR −2.8).
- 정확도와 지연의 교환점은 팀이 정한다 (CLAUDE.md: 최종 트레이드오프 지점은 Claude가 정하지 않음).

## 8. 계획과 달라진 점

1. **context switch(bucket) 줄 없음:** Zoo B32 v5.4.0 .alls에는 bucket 줄이 없다 (S1B16에만 있음). M_rb·M_both에는 resources_param 한 줄만 옮겼다. 층 대응이나 자동 분할은 필요 없었다 (설계 2절에 결과 전 기록).
2. **ORT 지연 스크립트:** `rn50/bench_speed.py`가 모든 모델에 224 입력 하나를 넣어서, 측정 부분을 그대로 옮기고 입력만 모델별 크기인 `s2_g6b_bench.py`로 쟀다 (5절, 결과 전 커밋). `s2_bench_speed.py --size` 옵션은 만들지 않았다.
3. **B32 ONNX 파일 이름:** `s2_quantize.py --src`가 이름이 `siglip2_image_fp32`로 시작해야 한다고 assert해서 `siglip2_image_fp32_g6b_b32_m2rot.onnx`·`siglip2_image_fp32_g6b_b32.onnx`로 이름만 바꿨다 (sha256 `23236123…`·`be19a8ad…` 그대로).
4. **M_rb 200장 확인:** run.sh가 아니라 `g6_hailo_chip.py eval --k 200`을 직접 돌렸다. 냉각 대기는 하지 않았고, 정확도 출력은 온도와 무관하다 (끝난 뒤 68 °C).
5. **M_both 전체 eval:** 규칙상 M_both는 전체 eval 대상이라 그대로 쟀다. 결과는 M_z16과 비트 단위로 같았다.
6. 회귀 확인
   - quantize: 옵션을 끄고 다시 만든 B16 M2rot_ort가 배포 파일과 sha256 `e84c3727…`로 같다 (`regress/quantize_sha256.txt`).
   - export: G6 b1 ONNX는 다시 만들지 않았다. M의 ONNX `a0eb502b…`를 그대로 썼고, `--dynamic`은 b1 경로를 바꾸지 않는다 (코드 diff).
   - bench: `s2_g6b_bench.py`의 루프는 `rn50/bench_speed.main`과 입력 줄만 다르다 (파일 머리말).

## 9. 확인 못 한 것

- **ORT 지연 측정의 격리 (검토 지적):**
  - 데스크탑 쪽은 단독이었다. DFC는 16:39:59에 끝났고, 게이트는 16:40:36에 통과했다. 9회 모두 dirty false이고, 측정 전후 다른 python·hailo는 0개였다.
  - 다만 AGX의 Hailo 속도 측정(16:44~17:16, 다른 기계)이 ORT 세션 1~3과 같은 시간대에 돌았다. 데스크탑에서는 ssh·rsync 감시만 했다.
  - 세션 1은 compile 직후 37초 만에(1분 부하 1.41, 5분 부하 2.83) 시작했고, 스레드 1 B16 FP32가 0.5% 느렸다 (238.8 vs 237.6 ms). 표는 세 세션의 중앙값이라 이 영향을 흡수한다.

- M_both와 Z 사이에 남은 +11.35%의 원인 (회전·sink 그래프의 마스크 연산, 보정 256장·level 1, ONNX 출처 중 무엇인지)
- resources_param의 다른 값, 또는 Zoo S1B16식 bucket을 B32에 직접 정했을 때의 속도
- ORT 공식 수치 (정민 노트북), Windows 쪽 부하
- 빌드 사이 변동 (같은 레시피를 다시 빌드했을 때의 정확도·지연 차이)

## 10. 재현

```bash
# 데스크탑 (siglip2/, source ~/siglip2_env.sh)
RN50_RESULTS=../results/siglip2/G6b python s2_g6.py --model b32 export --cfg m2rot --dynamic   # + --cfg plain --dynamic
python s2_quantize.py --src siglip2_image_fp32_g6b_b32_m2rot.onnx --calib-npy ~/siglip2_work/g6pre/cache_zoo256/calib256_s0_x.npy --calib-source highres --calib-n 256 --method percentile --act-type u8 --percentile 99.999 --op-types MatMul --hist-chunk 16 --seed 0 --tag all --dump-ops --reduce-range --threads 4
PYTHONPATH=~/siglip2_work/g6pre/pylib python s2_g6.py --model b16 fp32
python s2_g6b_ort.py --models b16_fp32=…:224:b16 b16_int8=…:224:b16 b32_fp32=…:256:b32 b32_int8=…:256:b32 --pairs b16_int8-b16_fp32 b32_int8-b32_fp32
M="…:224 …:256" SZ="b16_fp32=224 …" bash ../results/siglip2/G6b/g6b_ort_speed.sh       # 단독으로
nohup bash ../results/siglip2/G6b/g6b_build.sh > ~/siglip2_work/g6/g6b_build.log 2>&1 &   # M_rb → M_z16 → M_both
# AGX (~/ssd/junyong/hailo_g6)
python3 g6_hailo_chip.py eval --data . --out out/eval200_b32_M_rb --k 200 --models b32_M_rb=b32_M_rb.hef
HEFS="b32_M_z16" SKIP_BENCH=1 bash run.sh; HEFS="b32_M_both" SKIP_BENCH=1 bash run.sh; HEFS=b16_M2rot2 SKIP_BENCH=1 bash run.sh
HEFS="b32_Z b32_M b32_M_rb b32_M_z16 b32_M_both b16_M2rot2" SKIP_EVAL=1 bash run.sh
# 정리
python g6b_summarize.py --chip <AGX out 사본> --ref ~/siglip2_work/g6/ref --s2dir ../../../siglip2 --out results_chip.jsonl
```
