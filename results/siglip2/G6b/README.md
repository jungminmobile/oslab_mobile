# G6b: B32 M HEF가 Zoo보다 느린 원인 나누기 + 지연을 Zoo 수준으로 + B16 vs B32 ORT 비교 (설계 2026-10-20)

> 상태: **설계·판정 고정 (결과 전).** 1~5절은 결과 전에 커밋한다. 결과는 6절부터 붙인다.
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
(측정 후)
