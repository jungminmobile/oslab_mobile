# G6e: B16에서 conv41을 8비트로 내리면 실칩 출력만 무너지는 원인 찾기

브랜치 `siglip2/g6e-b16-g` (`siglip2/g6d-speed2` 기준). 목표는 B16에 G 방식을 안전하게 쓸 수 있는지 판단하는 것이다. G 방식은 B32 G6d의 G처럼 16비트 conv는 패치 임베딩만 남기고, 16비트는 잔차 덧셈 26개에만 쓰는 구성이다. test 세트는 쓰지 않았다.

> **요약**
> - **원인은 찾지 못했다.** 그래서 3단계 B16_G는 하지 않았다 (용 지시: 셋 다 붕괴가 그대로면 3단계 생략).
> - **H1(분리된 16비트 잔차 덧셈 층)만으로는 설명이 안 된다.** 같은 구조가 정상 HEF(B16_rb 19개, B32 G)에도 흔하다.
> - **H3(활성값 범위·포화)는 근거가 없다.** conv41과 ew_add38의 범위·스케일이 B32 G의 같은 위치 층과 거의 같다.
> - **H2(컨텍스트 경계)는 사실상 배제한다.** E1은 bucket 경계를 옮겨 컨텍스트 수(16 → 17)와 ew_add38의 배치를 바꿨다. 그런데 실칩 출력이 B16_both_bk와 **dev 6×200 전부 비트 단위로 같았다** (maxabs 0).
>   - 그래서 붕괴는 층이 어디에 배치되느냐와 상관없이, compile된 계산 자체가 같은 방식으로 틀린 것으로 본다.
> - E3(bucket + F 두 줄)과 E2(bucket 없음 + F 두 줄)는 **compile에 실패**했다.
> - 다음에 볼 것 1순위는 DFC의 `SDK_BIT_EXACT` 에뮬레이터로 붕괴를 재현해 층별로 비교하는 것이다 (5절).

## 1. 1단계: 빌드 없는 진단

- 스크립트: `g6e_diag.py` (compiled HAR의 `.hn`·`.auto.alls`·`.q.npz`를 읽는다). 결과: `diag.json`.
- 대상: B16_rb_bk(정상), B16_both_bk(붕괴), B32 G(정상). 모두 G6c·G6d에서 만든 compiled HAR이다.
- **경계 홉:** hn 그래프에서 다른 컨텍스트와 맞닿은 층까지의 최소 홉 수다. 0이면 경계 층이다.

### 1-1. ew_add38과 이웃 층 (B16_both_bk, 컨텍스트 16개)

| 층 | 정밀도 | 컨텍스트 | 경계 홉 | 출력 범위 (limvals_out) |
|---|---|---|---|---|
| ew_add36 (블록 9 attention 잔차, R19) | a16_w16_a16 | 12 (입력 ew_add31은 11) | 0 | −8.80~8.47 |
| conv40 (블록 9 fc1) | a8_w8_a8 | 12 | 2 | −0.27~34.35 |
| conv41 (블록 9 fc2) | a8_w8_a16 | 12 | 1 | −4.68~3.92 |
| **ew_add38** (블록 9 MLP 잔차, R20, `/trunk/Add_27`) | a16_w16_a16 | 12 | **0** (출력 → 13의 ew_add43) | −10.03~9.33 |
| LN21 분해 (reduce_mean·ew_sub·square·ew_mult) | a16_w16_a16 | 12 | 1~2 | — |
| ew_add43 (블록 10 attention 잔차) | a16_w16_a16 | 13 | 0 | −8.70~8.62 |

- **B16_rb_bk에는 ew_add38이 없다.** `/trunk/Add_27`이 16비트 conv41 안에 합쳐져 있다(conv41 입력 = conv40, ew_add36). conv41은 컨텍스트 13에 있다.
- 두 `.alls`에서 정밀도 줄의 차이는 conv41이 16비트 목록에 있느냐뿐이다. bucket 3줄은 같다.
- 컨텍스트 0~10은 두 HEF가 같다. 차이는 bucket 3 구간(컨텍스트 11~15)에서만 생겼고, 72개 층의 컨텍스트가 바뀌었다.
- bucket 3은 블록 8 norm1에서 시작한다. ew_add38은 bucket 경계에서 약 한 블록 뒤이고, 자동으로 정해진 컨텍스트 12/13 경계에 붙어 있다.

### 1-2. 가설별 판정 (1단계)

| 가설 | 판정 | 근거 |
|---|---|---|
| H1: 분리된 16비트 잔차 덧셈 층이 실칩에서 잘못 계산된다 | **이것만으로는 설명이 안 된다** | "8비트 conv(a8_w8_a16) + 16비트 잔차 → 16비트 ew_add" 구조가 B16_both에 20개, 정상인 B16_rb에 19개 있다. B32 G에서도 conv3·5·25가 8비트가 되면서 ew_add2·3·14가 떨어져 나왔고(471 → 475층) 실칩은 정상이다. ew_add38의 `compilation_param`도 ew_add29·31·36·43·45와 같다 |
| H2: 컨텍스트 경계 근처라서 경계를 넘을 때 값이 잘못 넘어간다 | 근거가 약하다 (2단계에서 사실상 배제) | 경계 홉 0인 분리 덧셈은 정상 HEF에도 있다 (B32 G ew_add3·14, B16_rb ew_add31·36) |
| H3: B16 활성값 범위가 커서 포화된다 | **근거가 없다** | 아래 표 |

| 층 | 입력 범위 | 출력 범위 | 출력 스케일 (채널별) | mult_shift / slope_e |
|---|---|---|---|---|
| B16_both conv41 | −0.27~34.35 | −4.68~3.92 | 1.29e-4~2.63e-4 | 1 / 3 |
| B32 G conv41 | −0.27~34.21 | −4.89~3.80 | 1.29e-4~2.65e-4 | 1 / 3 |
| B32 G conv5 (8비트로 내린 층) | −0.22~28.17 | −5.09~4.38 | 1.15e-4~2.89e-4 | 1 / 3 |
| B16_both ew_add38 | — | −10.03~9.33 | 5.91e-4 | 0 / 11 |
| B32 G ew_add34 | — | −8.27~8.49 | 5.12e-4 | 0 / 11 |
| B16_both ew_add45 (정상 블록 비교) | — | −10.27~9.12 | — | 0 / 11 |

- conv40부터 출력층까지 양자화 값이 바뀌었다. 보정 통계가 아래쪽 층으로 전파된 결과다.
- 하드웨어 설정(mult_shift, slope, zero point, bias_factor)만 보면, 바뀐 층은 conv40·conv41·ew_add38과 LN21~25의 일부(slopes_m, zero point, bias_factor)뿐이다. 모두 정상 동작하는 다른 블록과 같은 범위다.
- ew_add38의 bias가 768채널 모두 int16 범위를 넘는다. 하지만 정상인 ew_add43·45·50·52·57도 같아서(ew_add의 bias는 원래 큰 정수) 이상 신호로 보지 않는다.

## 2. 2단계: 원인을 가르는 빌드

모두 **B16_both의 같은 optimized HAR**(`7d22df79…`, G6c)에서 compile만 다시 했다. 상한 3시간, dfc.lock을 따랐다(G6f와 공유).

| 실험 | model script (B16_both_bk 대비 바뀐 것) | compile | 컨텍스트 | ew_add38 위치 | 실칩 6×200 |
|---|---|---|---|---|---|
| E3 | bucket 3줄 유지 + G6d F의 `allocator_param`·`context_switch_param` 두 줄 | **실패** (74분, rc=1): 7개 컨텍스트로 나눈 뒤 `siglip2_m2rot_b1 - Partition and Allocation Failed (Timeout)`, `Watchdog expired after 1h 1m` | — | — | — |
| E1 | bucket 3의 시작을 블록 9 norm1로 (LN17 ×5 → LN19 ×5, ew_add29 → ew_add36, precision_change16 → precision_change18) | 성공 (82분) | 17 | 컨텍스트 13, **경계 홉 0** (입력 ew_add36이 12에서 옴. 출력 ew_add43은 같은 13) | **붕괴 그대로.** B16_both_bk와 **비트 동일** |
| E2 | bucket 3줄 삭제 + F 두 줄 | **실패** (168분, rc=1): `Compilation failed with unexpected crash`, `Error has occured during the layer latency calculation` | — | — | — |
| E1c | (E1이 붕괴를 고쳤을 때만) | 하지 않음 | | | |

- model script: `alls/g6e_both_E3.alls`, `alls/g6e_both_E1.alls`, `alls/g6e_both_E2.alls`. compile 기록은 `build.jsonl`, `logs/<tag>/`.
- E1 배치: `diag_g6e_both_E1.json`.
- E1 HEF: sha256 `08ccc3edbe893efe…`, 108,216,320 B.

### 2-1. E1 실칩 판정 (`g6e_check200.py`, `check200.jsonl`)

판정 규칙(결과 전에 고정): 데이터셋 6개 모두 |cos(실칩, FP32) − cos(에뮬레이터, FP32)| ≤ 0.005이면 정상이다. 에뮬레이터는 같은 optimized HAR의 SDK_QUANTIZED 출력(G6c `emu_b16_both.npz`)이다.

| 데이터 | cos(실칩, FP32) | cos(에뮬레이터, FP32) | Δcos | 실칩 top-1 | 에뮬레이터 top-1 | FP32 top-1 | B16_both_bk 실칩과 비트 동일 |
|---|---|---|---|---|---|---|---|
| cifar_A | 0.3072 | 0.9607 | −0.654 | 35.0 | 69.0 | 74.0 | 예 |
| pets_dev | 0.4409 | 0.9897 | −0.549 | 53.5 | 92.5 | 93.5 | 예 |
| flowers_dev | 0.3680 | 0.9882 | −0.620 | 40.5 | 86.5 | 85.0 | 예 |
| cars_dev | 0.4168 | 0.9868 | −0.570 | 49.5 | 93.5 | 93.5 | 예 |
| food_dev | 0.4036 | 0.9838 | −0.580 | 43.5 | 87.0 | 87.5 | 예 |
| caltech_dev | 0.3910 | 0.9850 | −0.594 | 78.5 | 95.0 | 95.0 | 예 |

- **판정 스크립트 점검:** G6c 결과로 먼저 돌려 봤다.
  - B16_rb_bk: 정상. |Δcos|는 최대 0.0013이고, M2rot2 실칩과 6×200 비트 동일.
  - B16_both_bk: 붕괴. Δcos −0.55~−0.65.
- 실칩 조건: AGX Hailo-10H, HailoRT 5.3.0, `g6e_chip200.sh`. 시작 전 다른 Hailo 작업 0개, 48.2 °C에서 시작해 측정 직후 76.4 °C.

### 2-2. 원인 판정

- **원인 확정 조건**(E1 또는 E2가 붕괴를 없앰)을 **만족하지 못했다.** E3·E2는 compile이 실패했고, E1은 붕괴가 그대로였다.
- **H2 (위치·컨텍스트 경계): 사실상 배제한다.**
  - E1에서는 컨텍스트 수, bucket 3 시작점, ew_add38의 컨텍스트(12 → 13), 경계를 넘는 쪽(출력 → 입력)이 모두 바뀌었다.
  - 그런데도 실칩 출력이 비트 단위로 같았다.
  - 정상 쪽에서도 B16_rb_bk와 M2rot2는 compile 설정이 다른데 비트 동일했다(G6c). **이 칩에서는 출력 비트가 배치와 무관하게 정해진다**고 본다.
- **남은 설명:** HEF가 계산하는 내용 자체가 같은 optimized HAR의 에뮬레이터(SDK_QUANTIZED)와 다르다. 차이는 B16_both의 양자화 값과 그래프(conv41 8비트, ew_add38 분리)에 고정되어 있다.
  - 이 차이가 어느 층에서 생기는지는 확인하지 못했다.
  - conv40·conv41·ew_add38의 하드웨어 설정은 정상 층과 같은 범위였다. 그래서 그 층 하나의 설정값만 보고는 원인을 지목할 수 없다.

## 3. 3단계 (B16_G): 하지 않음

- 원인을 피하는 설정을 찾지 못했으므로 하지 않았다 (용 지시).
- **그래서 지금은 B16에 G 방식을 안전하게 적용할 수 있다고 판단할 수 없다.** B16에서 conv41을 8비트로 내린 HEF는 compile 설정을 바꿔도(E1) 같은 방식으로 무너졌다.
- B32 G는 정상이다(G6d). 그래서 이 문제는 "8비트 conv + 분리된 잔차 덧셈"이라는 일반 구조가 아니라, B16_both의 특정 양자화 모델과 관련된 것으로 보인다.

## 4. 계획에서 달라진 점

1. E1 해석(1단계 보고 때 용 확인): ew_add38은 bucket 경계가 아니라 자동 컨텍스트 경계에 붙어 있었다. 그래서 E1은 "bucket 3 시작을 블록 9 norm1로 옮기고, compile 뒤 경계 홉을 기록"으로 했다.
   - 결과적으로 경계 홉은 0 그대로였다. 다만 컨텍스트와 경계를 넘는 쪽은 바뀌었다.
2. E3(bucket + F 두 줄)을 추가하고 순서를 E3 → E1 → E2로 했다 (용 지시).
3. E1c는 조건(E1이 붕괴를 고침)을 만족하지 못해 하지 않았다.
4. "설정이 B16_rb HAR에서도 출력을 바꾸지 않는지" 확인은 붕괴를 고친 설정이 없어서 하지 않았다.

## 5. 확인 못 한 것과 다음에 볼 것

1. **(1순위, 빌드 없음) `SDK_BIT_EXACT` 에뮬레이터로 재현.**
   - DFC 5.4의 `InferenceContext`에는 `SDK_BIT_EXACT`와 `SDK_HAILO_HW`가 있다 (확인함, 아직 돌리지 않음).
   - B16_both optimized HAR을 BIT_EXACT로 몇 장 돌려 실칩 출력(비트 동일한 B16_both_bk·E1)과 같아지는지 본다.
   - 같아지면 층별 출력을 SDK_QUANTIZED와 비교해 처음 갈라지는 층을 찾을 수 있다. 칩도 빌드도 필요 없다.
   - 대조로 B16_rb HAR도 같이 돌린다(실칩이 SDK_QUANTIZED와 cos 0.98~0.99로 맞는 쪽).
2. **(BIT_EXACT가 재현하지 못할 때) 중간 출력이 있는 HEF.**
   - parse 때 end node를 여러 개 둔다(예: 블록 8·9·10의 잔차 출력 `/trunk/Add_25`·`/trunk/Add_27`·`/trunk/Add_32`, 그리고 최종 출력).
   - 같은 .alls로 optimize·compile해서 실칩 중간 출력과 에뮬레이터 중간 출력을 블록 단위로 비교한다.
   - 비용: optimize 약 30분 + compile 약 80분 + 칩 수 분. 출력이 늘어 보정 통계가 조금 달라질 수 있다.
3. **Zoo S1B16과 비교.**
   - Zoo S1B16도 16비트 conv가 conv4·5·24·25뿐(conv41은 8비트)인데 실칩에서 정상이다.
   - 우리 B16 M2rot 그래프에만 있는 연산(sink 마스크 등)이 블록 9 이후에서 8비트 conv41과 만나는 지점이 후보다.
4. **보정 다시 하기.** 같은 .alls로 보정 시드나 장수를 바꿔 optimize하면 붕괴가 계속되는지 본다. 특정 양자화 값의 우연인지, 구성 자체의 문제인지를 가른다.
5. 시간 제약: E3·E2와 같은 compile 실패를 피하려면 B16에서는 수동 bucket만 쓰고 F의 context_switch_param은 쓰지 않는 것이 안전하다 (G6c·G6e 실패 3건).

## 6. 재현

```bash
# 1단계 (siglip2 venv)
python g6e_diag.py --har rb_bk=<b16_rb_bk compiled.har> both_bk=<b16_both_bk compiled.har> b32_G=<b32_G compiled.har> \
  --focus rb_bk=conv41 both_bk=ew_add38 b32_G=ew_add3,ew_add14,ew_add34 --hops 2 --qdiff rb_bk,both_bk --out diag.json
# 2단계 compile (E3 → E1 → E2, dfc.lock)
nohup bash g6e_chain.sh > ~/siglip2_work/g6/g6e_chain.log 2>&1 < /dev/null &
# 실칩 6×200 (AGX ~/ssd/junyong/hailo_g6)
HEFS=b16_g6e_E1 nohup bash g6e_chip200.sh > g6e_chip200_E1.out 2>&1 < /dev/null &
# 판정
python g6e_check200.py --chip emb_b16_g6e_E1.npz --emu emu_b16_both.npz --ref ~/siglip2_work/g6/ref --tag g6e_both_E1 --out check200.jsonl
python g6e_check200.py --chip emb_b16_g6e_E1.npz --ref ~/siglip2_work/g6/ref --bitref emb_b16_both_bk.npz --tag g6e_both_E1_vs_both_bk --out check200.jsonl
```

- 실칩 임베딩 사본: `~/siglip2_work/g6/g6e_chip/eval200_b16_g6e_E1/` (큰 파일이라 커밋하지 않음). `chip/`에는 AGX 스크립트와 기록만 둔다.
- 칩 최고 온도: 76.37 °C (E1 6×200 직후). 95 °C 중단은 없었다.
