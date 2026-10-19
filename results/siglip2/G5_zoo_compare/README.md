# G5: Hailo Model Zoo SigLIP B/16 HEF와 우리 Hailo 최종 HEF의 실칩 속도·구조 비교 (설계 2026-10-19)

> 상태: **설계 고정 (1~3절, 결과 전).** 측정만 하는 실험이다. 모델·기존 결과 파일은 바꾸지 않는다. 채택 판정 없음.
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
| Zoo .alls | `hailo_model_zoo/cfg/alls/generic/siglip_b_16_image_encoder.alls` (GitHub 태그 v5.4.0) | 데스크탑 `~/siglip2_work/hailo_g5/zoo_alls/` | 1,023 | `8e31c5fb23e747de…` |

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
