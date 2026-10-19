# G5: Hailo Model Zoo SigLIP B/16 HEF와 우리 Hailo 최종 HEF의 실칩 속도·구조 비교 (설계 2026-10-19)

> 상태: **완료 (10/19 11:30).** 설계 1~3절은 결과 전에 고정(커밋 `9c1ff74`), 결과 4~5절, 해석 6절. 결론: 지연 차이는 컨텍스트 수 차이로 **설명되지 않는다** → H1(bucket 분할) 제안하지 않음. 측정만 하는 실험이다. 모델·기존 결과 파일은 바꾸지 않는다. 채택 판정 없음.
> 브랜치 `siglip2/g5-zoo-compare` (origin/main `f969f83`에서 만듦 — main에는 아직 `siglip2/` 코드가 없어서, 측정 스크립트는 이 폴더에 따로 둔다). 담당: 용 (용 요청).

## 1. 질문과 대상
- 질문 (하나): **우리 HEF와 Zoo HEF의 실칩 지연 차이 ≈ (컨텍스트 수 차이 × 약 2 ms)로 설명되는가?**
  설명되면 다음 실험으로 "우리 .alls에 bucket 분할 적용(H1)"을 제안한다. 부수 목적: 보고서용 Hailo 속도 기준선.
- 정확도는 비교하지 않는다 (SigLIP 1세대와 SigLIP2는 가중치가 다르다). 보고서에는 각자 FP32 대비 손실로만 나란히 둔다.

| 이름 | 무엇 | 파일 (커밋하지 않음) | 크기 B | sha256 |
|---|---|---|---|---|
| **M2rot2_D256L1** | 우리 Hailo 최종 (D-043, 2차 빌드) | AGX `~/ssd/junyong/hailo_repro/M2rot2_D256L1.hef` (데스크탑 `~/siglip2_work/hailo_repro/`) | 110,436,352 | `040a3ae1ebd81d9979ec893b5a65cf1265370196e8cfd516350e016191752c6f` |
| rothad2_D256L1 | 참고: 이전 최종 (D-042 구성, 2차 빌드 — 1차 `7243b76c…`와 실칩 출력 비트 단위 같음, G3RQ repro 5절) | AGX `~/ssd/junyong/hailo_repro/rothad2_D256L1.hef` | 107,540,480 | `705c3a81417cae2c4cf0d9b5b359846f95626b1d296bdd0166de0d43480df218` |
| siglip_b_16_image_encoder | Zoo v5.4.0 hailo10h, SigLIP 1세대 B/16 224 | AGX `~/ssd/junyong/hailo_g5/` (10/19 10:25 AGX에서 직접 받음) | 99,061,760 | `e2198ed5975da551d9a807c3c22d694d7d3f1e9034b9fe3bb3562601f8f40650` |
| siglip2_b_32_256_image_encoder | 참고: Zoo SigLIP2 B/32 256 (구조표·`benchmark`만) | 같은 폴더 | 102,645,760 | `c16123297d83a36cc862839ba1b9984d1ab539abe3773f99a332acae39e318cc` |
| Zoo 프로파일러 html | `siglip_b_16_image_encoder_profiler_results_compiled_runtime_data.html` | 같은 폴더, 데스크탑 `~/siglip2_work/hailo_g5/zoo/` | 7,962,075 | `3c463d7951654ab57861f421c4b4d1abef3c6c07b365254b217da8edf7fefde2` |
| Zoo .alls | `hailo_model_zoo/cfg/alls/generic/siglip_b_16_image_encoder.alls` (GitHub 태그 v5.4.0) | 데스크탑 `~/siglip2_work/hailo_g5/zoo_alls/` | 1,145 | `8e31c5fb23e747de…` (사본 `structure/zoo_siglip_b_16_image_encoder.alls`) |

- 요청 문구와 기록이 다른 점: 요청에는 M2rot "컨텍스트 약 17개, 약 100 MB"로 적혀 있었지만, `040a3ae1` HEF의 기록(`G3RQ_m2rot/hailo/repro/run_info.txt`)과 이번 `parse-hef`는 **21 컨텍스트, 110.4 MB**다. 17 컨텍스트·107.5 MB는 rothad다. 이 README는 기록·실측값을 쓴다.
- Zoo 공식 수치 (Zoo 표, 이번에 재지 않음): BS1 33.0 FPS, BS8 80.7 FPS, CIFAR-100 float 72.2 → HW 71.6, 호스트 i5-9400 + PCIe Gen3 x4. 우리 호스트는 AGX Orin(aarch64)이라 호스트·PCIe가 다르다.

## 2. 측정 방법 (G3RQ repro `chip/run.sh`와 같음, `run.sh`)
- 장비: AGX Orin + Hailo-10H, HailoRT 5.3.0, 연구실 공용 → 시작 전과 세션마다 다른 Hailo 프로세스 수를 로그에 남기고, 0개가 될 때까지 기다린다 (최대 3시간, 넘으면 멈춤). tmux 안 씀, `nohup bash run.sh`.
- **온도 규칙:** 세션 시작 전 칩 온도 max(ts0, ts1) < **64.6 °C** (쉬는 온도 62.6 + 2, G3RQ 5절과 같은 값. 10/19 10:28 쉬는 온도 62.9 °C 확인). 30초마다 확인, 최대 90분.
- **주 측정 (지연):** `g5_hailo_bench.py` = `s2_hailo_chip.py bench`에 기본값 1인 `--batch`만 더한 것. 워밍업 20, 200회, perf_counter, monitor 끔, 모델마다 VDevice 새로 만듦.
  - 세션 = HEF 3개(M2rot2 → rothad2 → Zoo B/16)를 한 번씩. **배치 1 세션 3번, 배치 8 세션 3번**, 세션마다 냉각.
  - 배치 8의 한 번 = 8장을 run_async로 모두 넣고 8개 모두 wait한 시간 (set_batch_size(8)). 처리량 = 8000 / 평균 ms (장/초).
  - **보고 값 = 세션별 평균(mean_ms)의 3세션 중앙값.** 세션별 중앙값(median_ms)의 3세션 중앙값도 같이 적는다 (G3RQ는 이것을 주 수치로 썼다).
  - 입력: 우리 HEF는 호스트에서 SigLIP 정규화 → HEF quant_info(scale 1/127, zp 128)로 UINT8. Zoo HEF는 입력 quant_info가 scale 1·zp 0(원본 픽셀, 정규화는 HEF 안 `norm1`)이라 픽셀 0~255로 넣는다. **호스트 전처리(정규화·양자화)는 타이머 밖**이다 (두 쪽 모두).
- 보조 (주 수치로 쓰지 않음): `hailortcli run2 -t 10 set-net <hef> --batch-size {1,8}` FPS, `hailortcli benchmark <hef> --batch-size 1` 1회 (HEF 4개), `run2 measure-fw-actions --batch-size 1` 런타임 데이터 (데스크탑 프로파일러에서 우리 HEF의 컨텍스트별 시간을 보기 위한 기록용).
- 모든 결과 행에 커밋 해시·dirty·HailoRT 버전·세션 시작 온도를 남긴다 (`results.jsonl`). AGX에는 저장소가 없으므로 스크립트 sha256을 run.log에 남겨 커밋된 파일과 같은지 확인한다.

## 3. 해석 규칙 (결과 전 고정)
- 컨텍스트 수 C (parse-hef): M2rot2 21, rothad2 17, Zoo B/16 17.
- 비교 쌍: (가) **M2rot2 − Zoo** (주), (나) rothad2 − Zoo (참고, ΔC = 0).
- 예측 Δ = ΔC × 2.0 ms. 실측 Δ = 배치 1 보고 값의 차이.
- **"설명됨" = |실측 Δ − 예측 Δ| ≤ max(1.0 ms, 0.25 × |실측 Δ|).** (가)·(나) 둘 다 "설명됨"일 때만 "컨텍스트 수 차이로 설명된다"고 쓰고 H1을 제안한다.
  (가)만 설명되고 (나)가 안 되면 "컨텍스트 수 말고 다른 원인이 있다"고 쓰고 H1은 제안하지 않는다 (원인 가설 2개 이상을 함께 적는다).
- 배치 8 결과는 기록만 한다 (판정 없음).
- **설계 시점에 이미 알던 것 (숨기지 않음):** 이전 실칩 배치 1 지연 rothad 46.17 ms·M2rot 47.95 ms (G3RQ repro), Zoo 프로파일러 html의 런타임 데이터 지연 30.08 ms (Zoo 측 측정, 17 컨텍스트 합 29.99 ms). 그래서 (나)는 크게 어긋날 가능성이 높다는 것을 알고 기준을 정했다. 기준은 요청 문구(ΔC × 2 ms)를 그대로 숫자로 옮긴 것이다.

## 4. 결과: 구조 (`parse/`, `structure/`, 프로파일러 추출 `g5_profiler_extract.py`)
| HEF | 컨텍스트 | 크기 (Zoo 대비) | 입력 | 정규화 | 출력 | 최적화 수준·보정 | 층 수 (프로파일러) |
|---|---|---|---|---|---|---|---|
| M2rot2_D256L1 | **21** | 110.4 MB (+11.5%) | UINT8 NHWC 224×224×3, quant scale 1/127·zp 128 | **HEF 밖** (호스트에서 SigLIP 정규화 → 양자화) | UINT16 1×1×768 (`ew_add57`) | 1 · 256장 | 501 |
| rothad2_D256L1 | 17 | 107.5 MB (+8.6%) | 같음 | HEF 밖 | UINT16 1×1×768 (`ew_add26`) | 1 · 256장 | 343 |
| Zoo siglip_b_16 | 17 | 99.1 MB | UINT8 NHWC 224×224×3, quant scale 1·zp 0 (원본 픽셀) | **HEF 안** (`norm1`, 16비트) | UINT16 1×1×768 (`ew_add26`) | 0 · 64장 | 346 |
| (참고) Zoo siglip2_b_32_256 | 12 | 102.6 MB | UINT8 NHWC **256×256×3**, scale 1·zp 0 | HEF 안 | UINT16 1×1×768 | — | — |
- 모두 DFC 5.4.0 컴파일, "HAILO15H, HAILO10H 호환". Zoo 프로파일러 html의 hw_arch 표기는 `hailo15h`다 (Zoo가 받은 파일 그대로).
- **16비트 설정은 거의 같다.** 두 쪽 모두 conv1, 블록 0 fc1·fc2(conv4·5), 블록 5 fc1·fc2(conv24·25), LN 분해 층, (컴파일 뒤 남은) ew_add 전부가 16비트다. 우리 쪽만 conv41(블록 9 fc2)이 16비트이고, Zoo 쪽만 입력 정규화 `norm1`이 HEF 안에서 16비트다. M2rot는 마스크·캐시용 ew_add 30개가 8비트로 더 있다 (층 501개).
- **Zoo .alls에만 있는 것:** `resources_param(max_compute_utilization=1.0, max_control_utilization=0.95, max_memory_utilization=0.75)`, `bucket()` 3개, 최적화 수준 0·보정 64장 (뒤 두 가지는 정밀도·보정 설정이라 컴파일 배치에는 직접 관계없다고 보지만 확인하지 않음).
- **bucket은 컨텍스트 수를 줄이지 않았다** (Zoo 17 = rothad 17). bucket 경계는 컨텍스트 경계와 맞는다: b1의 끝 ew_add8은 컨텍스트 4 끝(5는 conv17 한 층), b2는 ew_add9로 컨텍스트 6을 시작해서 ew_add17(컨텍스트 10, conv33·ew_add17 두 층)에서 끝난다.
- **16비트 큰 FC층의 배치가 다르다** (`structure/*_layers.csv`의 context 열):

| 층 (16비트) | Zoo 컨텍스트 | rothad2 컨텍스트 | M2rot2 컨텍스트 |
|---|---|---|---|
| conv4·conv5 (블록 0 fc1·fc2) | **1·1** (같은 컨텍스트) | 1·2 (conv5 혼자) | 1·2 |
| conv24·conv25 (블록 5 fc1·fc2) | **7·7** | 7·8 | 8·9 |
| conv41 (블록 9 fc2, 우리만 16비트) | 13 (8비트) | 13 (혼자) | 16 |
- 층 1~6개짜리 작은 컨텍스트: Zoo 2개 (5, 10) / rothad2 5개 (1, 2, 5, 10, 13) / M2rot2 4개 (1, 2, 5, 8).
- **Zoo 컨텍스트별 실측 시간** (Zoo html 안 런타임 데이터, Zoo 측 호스트, 배치 1): 17개 합 29.99 ms (html 지연 30.08 ms, 33.24 FPS). 컨텍스트마다 0.53~4.26 ms로 고르지 않다 (평균 1.76 ms). FW 컨텍스트 전환 overhead는 컨텍스트당 0.04~0.09 ms, 합 1.0 ms (`structure/zoo_b16_contexts.csv`).
- 우리 HEF의 컨텍스트별 실측 시간은 **얻지 못했다** (7절 실패 기록). 데스크탑 프로파일러는 컴파일 정보만 있다.

## 5. 결과: 실칩 지연 (AGX Hailo-10H, HailoRT 5.3.0, 10/19 10:31~11:30, 커밋 `9c1ff74` dirty 0, `results.jsonl`, raw `chip/`)
- 시작 전 AGX 사용자 0, 다른 Hailo 작업 0 (세션마다 0). 세션 시작 온도 64.18~64.58 °C (모두 < 64.6). AGX의 `g5_hailo_bench.py`·`run.sh` sha256 = 커밋된 파일 (`4cea4d29…`·`e10c08e4…`, run.log 첫 줄).
- 첫 줄의 HEF sha256 칸은 비어 있다 (`sha256sum -L`이 잘못된 옵션). HEF 확인은 `bench.json` 행마다 남은 sha256(앞 16자)로 했다: `040a3ae1ebd81d99`·`705c3a81417cae2c`·`e2198ed5975da551`, 모두 1절과 같다.

### 배치 1 (보고 값 = 세션별 평균의 3세션 중앙값)
| HEF | C | 세션 1 / 2 / 3 평균 ms | **보고 ms** | 처리량 장/초 | 세션별 중앙값의 중앙값 ms | p90 ms | Zoo 대비 |
|---|---|---|---|---|---|---|---|
| **M2rot2_D256L1** | 21 | 47.561 / 47.545 / 47.583 | **47.561** | 21.03 | 47.553 | 47.69~47.76 | **+14.22 ms (+42.7%)** |
| rothad2_D256L1 | 17 | 45.908 / 45.929 / 45.957 | **45.929** | 21.77 | 45.934 | 46.03~46.07 | +12.59 ms (+37.8%) |
| **Zoo siglip_b_16** | 17 | 32.636 / 33.416 / 33.340 | **33.340** | 29.99 | 31.781 | 35.85~36.00 | — |
- 이전 측정과 맞는다: G3RQ repro(10/19 05:44) rothad2 중앙값 46.171·M2rot2 47.951 ms → 이번 45.934·47.553 ms (−0.5%·−0.8%).
- Zoo는 우리보다 흩어짐이 크다 (평균 > 중앙값, p90 − 중앙값 ≈ 4 ms; 우리는 0.2 ms).

### 배치 8 (기록만, 한 번 = 8장 넣고 8개 모두 기다린 시간)
| HEF | 보고 ms / 8장 | 처리량 장/초 | 세션별 중앙값의 중앙값 ms | p90 ms |
|---|---|---|---|---|
| M2rot2_D256L1 | 243.152 | 32.90 | 249.965 | 250.0 |
| rothad2_D256L1 | 235.096 | 34.03 | 234.997 | 235.1~236.0 |
| Zoo siglip_b_16 | 179.225 | 44.64 | 164.027 | 240.0 |
- M2rot2·Zoo는 배치 시간 분포가 한쪽으로 몰리지 않는다 (Zoo 중앙값 164 ms인데 p90 240 ms, M2rot2는 평균 < 중앙값). 원인은 보지 않았다.

### 보조 (주 수치 아님, 1회씩)
| HEF | `run2 -t 10` 배치 1 FPS | `run2 -t 10` 배치 8 FPS | `benchmark --batch-size 1` FPS |
|---|---|---|---|
| M2rot2_D256L1 | 21.58 | 41.66 | 21.62 |
| rothad2_D256L1 | 22.38 | 44.85 | 22.33 |
| Zoo siglip_b_16 | 32.87 | 76.02 | 32.97 |
| (참고) Zoo siglip2_b_32_256 | — | — | 64.54 |
- **Zoo 공식 수치(BS1 33.0, BS8 80.7 FPS, i5-9400 x86 호스트)와 이 AGX의 run2 값(32.87, 76.02)이 가깝다** → 호스트 차이는 배치 1에서 거의 없다. 배치 8은 −5.8%.
- 배치 8 처리량은 측정 방법에 따라 크게 다르다: 주 측정(8장씩 넣고 기다리기)은 Zoo 44.6 장/초, run2(계속 흘려 넣기)는 76.0 FPS. 보고서에 배치 8 처리량을 쓸 때는 어느 방법인지 밝혀야 한다.

### 보고서용 나란히 (정확도는 비교가 아님 — 서로 다른 가중치·데이터)
| HEF | 배치 1 지연 (이 AGX) | 크기 | 정확도 손실 (각자 FP32 대비, 각자 출처) |
|---|---|---|---|
| M2rot2_D256L1 (우리) | 47.56 ms (21.0 장/초) | 110.4 MB | 실칩 dev 고화질 5종 −0.41%p (`G3RQ_m2rot/hailo/README.md`, 1차 빌드 = 2차와 실칩 출력 같음) |
| Zoo SigLIP B/16 | 33.34 ms (30.0 장/초) | 99.1 MB | CIFAR-100 72.2 → 71.6 (−0.6%p, Zoo 표) |

## 6. 해석 (3절 규칙 그대로)
| 쌍 | ΔC | 예측 Δ (ΔC × 2 ms) | 실측 Δ (배치 1) | |실측 − 예측| | 허용 max(1, 0.25·|실측|) | 판정 |
|---|---|---|---|---|---|---|
| (가) M2rot2 − Zoo | +4 | +8.0 ms | **+14.22 ms** | 6.22 | 3.56 | **설명 안 됨** |
| (나) rothad2 − Zoo | 0 | 0.0 ms | **+12.59 ms** | 12.59 | 3.15 | **설명 안 됨** |
- **결론: 지연 차이는 컨텍스트 수 차이로 설명되지 않는다.** 컨텍스트 수가 같은 rothad2도 Zoo보다 12.6 ms(+38%) 느리다. 그래서 규칙대로 **H1(bucket 분할 적용)은 제안하지 않는다.** 4절에서 bucket이 컨텍스트 수를 줄이지 않았다는 것도 확인했다.
- (기록) M2rot2 − rothad2 = +1.63 ms, 늘어난 컨텍스트 4개 → 컨텍스트당 약 0.41 ms. "컨텍스트당 약 2 ms"보다 작다 (늘어난 컨텍스트에 든 일의 양도 함께 늘었으므로 순수 고정 비용은 아님).
- **차이의 원인 가설 (확인 안 함, 확인하려면 컴파일 실험이 필요):**
  1. **컴파일 자원 한도 `resources_param`**: Zoo는 compute 1.0·control 0.95·memory 0.75로 두었고 우리는 기본값이다. 4절 표처럼 Zoo는 16비트 fc1·fc2 쌍(conv4·5, conv24·25)을 한 컨텍스트에 넣었고, 우리는 둘로 나눴다. 컨텍스트 수가 같아도 컨텍스트마다 쓰는 자원과 데이터 이동량이 다를 수 있다.
  2. **컨텍스트 나누는 위치(bucket 포함)**: 우리 rothad2에는 1~6층짜리 작은 컨텍스트가 5개 있다 (Zoo 2개). 작은 컨텍스트도 가중치·설정을 불러오는 시간은 든다 (Zoo 런타임 데이터의 시간 비율: infer 28.6%, drain 57.9%, configs 5.6%, sequencers 4.6%, overhead 3.4%; 대역폭 비율: 설정 불러오기 81.2%, 컨텍스트 사이 데이터 17.3%). bucket은 컨텍스트 수는 그대로여도 경계 위치를 바꾼다.
  3. (작은 요인) 우리만 conv41이 16비트 (rothad2에서 혼자 컨텍스트 13을 차지).
  - 가설 1·2를 따로, 그리고 함께 확인하는 방법 (팀 결정 사항, 이번에 하지 않음): rothad 또는 M2rot `.alls`에 (A) Zoo `resources_param`만 / (B) bucket만 / (A+B)를 넣어 다시 컴파일 → 컨텍스트 배치·실칩 지연 비교. 정밀도·보정은 그대로라 실칩 출력이 바뀌는지도 함께 확인해야 한다.
- 입력 처리 차이: Zoo는 정규화가 HEF 안에 있어 호스트는 픽셀 uint8을 그대로 넘긴다. 우리는 호스트에서 정규화·양자화가 필요하다. 이번 지연은 두 쪽 모두 이 호스트 처리를 **뺀** 값이다.

## 7. 실패·이상 기록
- **런타임 데이터 수집 실패 (기록용 항목):** `hailortcli run2 -t 5 measure-fw-actions <hef> --batch-size 1 --output-path …`가 HEF 3개 모두 rc=111, `Only one model is allowed when measuring fw actions` (`chip/rtdata_*.log`). 같은 오류가 3번 나서 다시 시도하지 않았다. 명령 형식(하위 명령 위치) 문제로 보이지만 확인하지 않았다. 우리 HEF의 컨텍스트별 실측 시간이 있으면 6절 가설 2를 더 좁힐 수 있다.
- `run.log` 첫 줄 HEF sha256 칸이 비어 있음 (5절).

## 8. 재현
```bash
# AGX (~/ssd/junyong/hailo_g5): HEF 4개(우리 둘은 ../hailo_repro 심볼릭 링크), g5_hailo_bench.py, run.sh, COMMIT
nohup bash run.sh > run.out 2>&1 &
# 데스크탑: raw를 chip/로 가져온 뒤
python3 -I g5_summarize.py chip 9c1ff74 0
python3 -I g5_profiler_extract.py <profiler.html> structure/<접두어>
```
- 우리 프로파일러 html: `~/siglip2_work/hailo_repro/comp2_{M2rot,rothad}/profiler.html` (G3RQ repro 2차 빌드 때 생성). Zoo html·HEF는 커밋하지 않는다 (경로·sha256은 1절).
