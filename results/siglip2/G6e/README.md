# G6e: B16에서 conv41을 8비트로 내리면 실칩 출력만 무너지는 원인 찾기

브랜치 `siglip2/g6e-b16-g` (`siglip2/g6d-speed2` 기준). 목표는 B16에 G 방식을 안전하게 쓸 수 있는지 판단하는 것이다. G 방식은 B32 G6d의 G처럼 16비트 conv는 패치 임베딩만 남기고, 16비트는 잔차 덧셈 26개에만 쓰는 구성이다. test 세트는 쓰지 않았다.

> **요약**
> - **원인은 찾지 못했다.** 그래서 3단계 B16_G는 하지 않았다 (용 지시: 셋 다 붕괴가 그대로면 3단계 생략).
> - **H1(분리된 16비트 잔차 덧셈 층)만으로는 설명이 안 된다.** 같은 구조가 정상 HEF(B16_rb에 약 20개, B32 G)에도 흔하다.
> - **H3(활성값 범위·포화)는 정적 양자화 통계상 근거가 없다.** conv41과 ew_add38의 범위·스케일이 B32 G의 같은 위치 층과 거의 같다.
> - **H2(컨텍스트 경계): 특정 배치는 배제했다.** E1은 bucket 경계를 옮겨 컨텍스트 수(16 → 17), ew_add38의 컨텍스트, 경계를 넘는 텐서를 바꿨다. 그런데 실칩 출력이 B16_both_bk와 **dev 6×200 전부 비트 단위로 같았다** (maxabs 0).
>   - 다만 ew_add38은 두 HEF 모두 경계 층(홉 0)이어서, "경계에 붙어 있다"는 조건 자체는 직접 바꿔 보지 못했다. 비트가 같으므로 배치에 따라 생기는 오류일 가능성은 매우 낮다고 본다.
>   - 그래서 붕괴는 배치와 상관없이, compile된 계산 자체가 같은 방식으로 틀린 것으로 본다.
> - E3(bucket + F 두 줄)과 E2(bucket 없음 + F 두 줄)는 **compile에 실패**했다.
> - **(10/22 추가) 4단계 SDK_BIT_EXACT 진단 (8절):** 규칙상 "판정 불가"다.
>   - bit-exact도 붕괴를 재현하지 못했다 (B16_both 실칩과 cos 0.15~0.46).
>   - 정상인 B16_rb에서도 실칩과 비트 단위로 맞지 않았다 (0.97~0.99).
>   - DFC 에뮬레이터 두 종류 모두 붕괴를 재현하지 않는다. 원인이 compile 결과물(HEF)인지, 에뮬레이터가 실칩 연산을 따라가지 못하는 부분(예: 실제 누산·포화)인지는 이 결과로 가를 수 없다.

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
- B16_rb_bk와 B16_both_bk는 컨텍스트 0~10의 배치가 같다. 차이는 bucket 3 구간(컨텍스트 11~15)에서만 생겼고, 72개 층의 컨텍스트가 바뀌었다. (E1과 B16_both_bk는 컨텍스트 8부터 166개 층이 다르다.)
- bucket 3은 블록 8 norm1에서 시작한다. ew_add38은 bucket 경계에서 약 한 블록 뒤이고, 자동으로 정해진 컨텍스트 12/13 경계에 붙어 있다.

### 1-2. 가설별 판정 (1단계)

| 가설 | 판정 | 근거 |
|---|---|---|
| H1: 분리된 16비트 잔차 덧셈 층이 실칩에서 잘못 계산된다 | **이것만으로는 설명이 안 된다** | "8비트 conv(a8_w8_a16) + 16비트 잔차 → 16비트 ew_add" 구조가 B16_both에 약 20개, 정상인 B16_rb에는 하나 적게 있다 (세는 규칙에 따라 다르다: 26개 잔차 목록 기준 20/19, 입력에 a8_w8_a16 conv가 있는 a16 ew_add 전부 23/22, 잔차 입력이 ew_add인 것만 19/17 — 어느 규칙이든 차이는 ew_add38 하나). B32 G에서도 conv3·5·25가 8비트가 되면서 ew_add2·3·14가 떨어져 나왔고(471 → 475층) 실칩은 정상이다. ew_add38의 `compilation_param`도 ew_add29·31·36·43·45와 같다 |
| H2: 컨텍스트 경계 근처라서 경계를 넘을 때 값이 잘못 넘어간다 | 근거가 약하다 (2단계에서 사실상 배제) | 경계 홉 0인 분리 덧셈은 정상 HEF에도 있다 (B32 G ew_add3·14, B16_rb ew_add31·36) |
| H3: B16 활성값 범위가 커서 포화된다 | **정적 양자화 통계상 근거가 없다** (실행 중 포화를 직접 잰 것은 아니다) | 아래 표 |

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
| E1 | bucket 3의 시작을 블록 9 norm1로 (LN17 ×5 → LN19 ×5, ew_add29 → ew_add36, precision_change16 → precision_change18). **실수로 16비트 `quantization_param` 줄의 ew_add29도 ew_add36으로 바뀌었다** (아래 참고) | 성공 (82분) | 17 | 컨텍스트 13, **경계 홉 0** (입력 ew_add36이 12에서 옴. 출력 ew_add43은 같은 13) | **붕괴 그대로.** B16_both_bk와 **비트 동일** |
| E2 | bucket 3줄 삭제 + F 두 줄 | **실패** (168분, rc=1): `Compilation failed with unexpected crash`, `Error has occured during the layer latency calculation` | — | — | — |
| E1c | (E1이 붕괴를 고쳤을 때만) | 하지 않음 | | | |

- model script: `alls/g6e_both_E3.alls`, `alls/g6e_both_E1.alls`, `alls/g6e_both_E2.alls`. compile 기록은 `build.jsonl`, `logs/<tag>/`.
- E1 배치: `diag_g6e_both_E1.json`.
- E1 HEF: sha256 `08ccc3edbe893efe…`, 108,216,320 B.
- **E1 model script의 실수 (독립 검토로 발견):**
  - bucket 줄을 고치는 치환이 16비트 `quantization_param` 목록에도 걸렸다. 그래서 그 목록에서 ew_add29가 빠지고 ew_add36이 두 번 들어갔다.
  - compile만 했으므로 영향은 없다. E1 compiled HAR의 `.hn`과 `.q.npz`는 B16_both_bk와 **바이트 단위로 같다** (`cmp`로 확인). 즉 양자화 모델은 그대로이고, 바뀐 것은 bucket 줄뿐이다.
  - E2·E3는 의도한 줄만 다르다.
- **E1이 실제로 바꾼 것과 못 바꾼 것:**
  - 바꾼 것: 컨텍스트 배치(컨텍스트 8부터 166개 층), ew_add38의 컨텍스트와 경계를 넘는 쪽, ew_add38의 `balance_output_multisplit` (disabled → enabled).
  - 못 바꾼 것: conv41의 `compilation_param` (4갈래 분할 d0~d3+dc). 이 설정은 정상 블록의 8비트 fc2인 conv45와 같다.

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

- cos(에뮬레이터, 실칩)은 0.31~0.44다 (`check200.jsonl`).
- **판정 스크립트 점검:** G6c 결과로 먼저 돌려 봤다 (`check200.jsonl`의 control 줄. 모든 줄에 커밋 해시가 있고 dirty=false).
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
4. **보정 다시 하기** (빌드 1회로 두 설명을 가를 수 있어서 2번보다 먼저 해도 된다). 같은 .alls로 보정 시드나 장수를 바꿔 optimize하면 붕괴가 계속되는지 본다. 특정 양자화 값의 우연인지, 구성 자체의 문제인지를 가른다.
5. compile 실패를 피하려면 B16에서는 수동 bucket만 쓰고 F의 두 줄은 쓰지 않는 것이 안전하다.
   - B16 compile 실패는 4건이다. G6c B16_rb·B16_both는 bucket 없음, E3는 bucket + F, E2는 bucket 없음 + F였다.
   - 성공한 것(B16_rb_bk, B16_both_bk, E1)은 모두 "bucket 있음, F 없음"이었다.

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

## 7. 확인 기록

1. **자기 검증**
   - 판정 스크립트를 G6c 결과(정상 대조, 붕괴 대조)로 먼저 돌렸다.
   - E1 HAR의 `.hn`·`.q.npz`가 B16_both_bk와 같은지 `cmp`로 확인했다.
   - optimized HAR이 G6c와 같은 파일인지 sha256으로 확인했다.
2. **독립 검토** (맥락 없는 서브에이전트가 compiled HAR과 임베딩 원본에서 직접 다시 계산)
   - 2-1 표의 값, 비트 동일, 컨텍스트·경계 홉, H3 표, 로그 값(시간·오류 문구·온도)이 모두 일치했다. Δcos 두 개는 넷째 자리 반올림만 다르다.
   - 반영한 지적:
     - (a) E1 model script 실수와 그 영향 없음 근거
     - (b) H2 문구를 "특정 배치 배제"로 좁힘
     - (c) "컨텍스트 0~10 같음"이 어떤 두 HEF 사이의 비교인지 명시
     - (d) H3를 "정적 양자화 통계상"으로 한정
     - (e) H1의 개수를 세는 규칙 명시
     - (f) compile 실패 건수를 4건으로 정정
     - (g) check200 줄에 커밋 해시 추가
     - (h) conv41 compile 설정이 E1에서 바뀌지 않았다는 점, cos(에뮬레이터, 실칩) 추가

## 8. 4단계: SDK_BIT_EXACT 진단 (10/22, 칩·compile 안 씀, 상한 2시간)

> **요약**
> - 미리 고정한 규칙으로는 **"판정 불가"**다.
> - SDK_BIT_EXACT는 B16_both의 **실칩 붕괴를 재현하지 못했다** (cos(bit-exact, 실칩) 0.15~0.46). bit-exact 출력은 SDK_QUANTIZED 쪽에 가깝다 (0.96~0.99).
> - 정상인 B16_rb에서도 bit-exact와 실칩은 cos 0.97~0.99에 그쳤다. **이 DFC·칩 조합에서는 bit-exact 모드도 실칩과 비트 단위로 맞지 않는다.**
> - 그래서 "재현됨" 조건(두 HAR 모두 실칩과 ≥ 0.999)도, "재현 안 됨" 조건(B16_both가 QUANT와 ≥ 0.999)도 만족하지 못했다.
> - 내용상으로는 **DFC 에뮬레이터 두 종류(SDK_QUANTIZED, SDK_BIT_EXACT) 모두 붕괴를 재현하지 않는다.** 원인이 compile 결과물(HEF)인지, 에뮬레이터가 실칩 연산을 따라가지 못하는 부분(예: 16비트 경로의 실제 누산·포화)인지는 이 결과로 가를 수 없다.
> - 재현되지 않았으므로 층별 비교(4번)는 하지 않고 멈췄다. 걸린 시간은 33분(02:18~02:51)이다.

### 8-1. SDK_BIT_EXACT가 받는 것과 지원 범위 (출처)

DFC 5.4.0(`hailo-dataflow-compiler 5.4.0`, `~/hailo10_venv`)의 설치본에는 별도 문서 파일이 없다. 그래서 설치된 소스로 확인했다.

- **정의:** `hailo_sdk_client/exposed_definitions.py:133` — `#: SDK_BIT_EXACT (preview) bit exact emulation. Currently not all layers and mode are supported`
  - 미리보기 기능이고, 모든 층을 지원하지는 않는다고 적혀 있다.
- **받는 HAR:** `hailo_sdk_client/runner/client_runner.py` `_infer_emulator`
  - SDK_QUANTIZED·SDK_BIT_EXACT는 상태가 `QUANTIZED_MODEL` 또는 `COMPILED_MODEL`이어야 한다. 즉 optimized HAR과 compiled HAR 둘 다 받는다.
  - 이 상태 검사는 `requires_quantized_weights`가 참일 때만 적용된다.
  - 우리는 **optimized HAR**(G6c `b16_both`·`b16_rb`의 `siglip2_m2rot_optimized.har`)을 썼다.
  - 기본 경로는 acceleras(`sdk_backend.acceleras_inference`)다. `experimental_torch_infer`일 때만 saitama 경로를 쓴다.
- **사용 데이터:** `sdk_backend.build_acceleras_model`(sdk_backend.py:988)은 SDK_BIT_EXACT와 SDK_QUANTIZED에 같은 분기를 쓴다: hn `get_hn_dict`, 값 `get_params_hailo_optimized`·`get_params_translated`.
  - 정수 연산 모드는 그 뒤 `acceleras_inference`(sdk_backend.py 약 1472~1476)에서 `set_quantized(); set_native(False); set_bit_exact(True)`로 켠다.
  - 따라서 compile 결과물(HEF의 배치·분할)은 반영되지 않는다.
- **층별 지원:** `acceleras/hailo_layers/base_hailo_layer.py`의 `bit_exact_supported`(모든 atomic op가 지원해야 True. 아니면 `cant support bit exact` 오류)를 B16_both HAR의 489층에 대해 조회했다 (`g6e_bitexact.py probe`).
  - **지원하지 않는 층은 없었다.**
  - **실제로 bit-exact 모드로 돌았다는 근거:** 실행 경로에서 판정하는 곳은 `hailo_model_optimization/flows/inference_flow.py:361~363`의 `set_bit_exact`다.
    - 모든 op의 `bit_exact_emulation_supported`(`bit_exact_supported and not fully_native`, `base_atomic_op.py:343`)가 참이 아니면 `AccelerasImplementationError("Not all layers in the models support bit exact")`로 멈춘다.
    - 실행이 오류 없이 끝났으므로, 모든 op가 bit-exact 모드로 돌았다. 지원하지 않는 op를 조용히 일반 계산으로 대체하는 줄(`base_hailo_layer.py:378`)이 있지만, 이 경로에서는 그 전에 오류가 난다.
    - probe(`bit_exact_supported`, `set_native(False)` 전)는 이보다 약한 확인이다.
    - 다만 "지원"은 op 스스로의 선언이다 (예: `GenericNativeOp`는 항상 True). 기능도 preview이고, 대조군 결과는 이 모드가 실칩과 비트 단위로 맞지 않음을 보여 준다.
  - 우리 층 종류도 모두 지원한다: 16비트 잔차 덧셈 `HailoElementwiseAdd` 54, LN 분해 `HailoReduceMean` 26·`HailoElementwiseSub` 39·`HailoFeatureMultiplierOnMac` 26·`HailoElementwiseMultOnMac` 26, 마스크 연산 `HailoElementwiseMult` 44·`HailoConst` 33, 그 밖에 `HailoConv` 75·`HailoConvAdd` 3·`HailoMatmul` 26·`HailoReduceSum` 47·`HailoReduceMax` 13·`HailoNormalization` 31·`HailoStandaloneActivation` 34 등.
  - (참고) saitama 경로의 bit-exact 등록표에는 concat·conv·dw·ew_add·ew_mult·feature_shuffle·input·matmul·norm·output·shortcut·slice만 있다. 기본 경로가 아니라서 쓰지 않았다.

### 8-2. 사진과 방법

- **사진 8장:** CIFAR `cifar_A` 0~3번, 고화질 `pets_dev`·`flowers_dev`·`cars_dev`·`food_dev` 각 0번.
  - G6c에서 실칩 출력을 이미 저장한 사진이다. 실칩 출력은 B16_both_bk(G6c dev 전체 eval)와 B16_rb_bk(G6c eval200, M2rot2와 비트 동일)에서 가져왔다.
- **입력:** `g6c_emulate.py`와 같다 (Zoo 전처리 픽셀 x/127.5 − 1).
  - 실칩 eval(`G6/chip/g6_hailo_chip.py`)도 같은 float 값을 보낸다. HailoRT는 이것을 HEF의 입력 양자화(scale 1/127, zp 128)로 uint8로 바꾼다.
  - 픽셀 값 256개 중 반올림 경계(±0.002 안)에 걸리는 값은 없다 (독립 검토). 그래서 반올림 방식이면 양쪽 uint8 입력이 같다.
  - HailoRT가 반올림 대신 버림을 쓰거나, 에뮬레이터의 입력 스케일이 HEF와 다르면 1 LSB 차이가 날 수 있다. 이것은 확인하지 못했다.
  - 다음에 할 때는 실칩에는 미리 양자화한 uint8을, 에뮬레이터에는 격자 값 (q − 128)/127을 넣어 입력 문제를 없앤다.
- **실행 조건:** `g6e_bitexact.py run --ctx bit_exact`, 배치 1, CPU(`CUDA_VISIBLE_DEVICES=-1`, OpenBLAS 6스레드), dfc.lock 안에서 돌렸다.
- **QUANT:** G6c `emu_b16_both.npz`·`emu_b16_rb.npz` (SDK_QUANTIZED, 같은 HAR).
- **한 장당 시간:** 86~106초 (첫 장은 준비 포함 약 100초). 8장 × 2 HAR = 약 26분으로 1시간보다 짧아 8장 그대로 했다.

### 8-3. 결과 (`bitexact.jsonl`, `g6e_bitexact_cmp.py`)

| 사진 | both: BE–실칩 | both: BE–QUANT | both: QUANT–실칩 | rb: BE–실칩 | rb: BE–QUANT | rb: QUANT–실칩 | 시간(초) both / rb |
|---|---|---|---|---|---|---|---|
| cifar_A:0 | 0.3175 | 0.9682 | 0.3397 | 0.9658 | 0.9727 | 0.9807 | 100 / 101 |
| cifar_A:1 | 0.3211 | 0.9886 | 0.3215 | 0.9826 | 0.9884 | 0.9813 | 90 / 88 |
| cifar_A:2 | 0.3411 | 0.9841 | 0.3355 | 0.9868 | 0.9840 | 0.9801 | 89 / 99 |
| cifar_A:3 | 0.1459 | 0.9609 | 0.1686 | 0.9761 | 0.9553 | 0.9742 | 88 / 106 |
| pets_dev:0 | 0.3720 | 0.9822 | 0.3684 | 0.9889 | 0.9817 | 0.9789 | 89 / 97 |
| flowers_dev:0 | 0.3319 | 0.9841 | 0.3481 | 0.9871 | 0.9866 | 0.9859 | 89 / 100 |
| cars_dev:0 | 0.4584 | 0.9935 | 0.4520 | 0.9934 | 0.9938 | 0.9922 | 89 / 98 |
| food_dev:0 | 0.4516 | 0.9762 | 0.4614 | 0.9867 | 0.9754 | 0.9821 | 86 / 99 |

(BE = SDK_BIT_EXACT 출력. 값은 모두 cos. 실칩과 비트 단위로 같은 사진은 두 HAR 모두 0장)

- **B16_both:** bit-exact는 실칩과 0.15~0.46으로 다르다. QUANT와는 0.96~0.99로 비슷하다. 즉 bit-exact도 "붕괴하지 않는 양자화 모델"을 계산한다.
- **B16_rb (대조군):** bit-exact–실칩 0.966~0.993, QUANT–실칩 0.974~0.992, bit-exact–QUANT 0.955~0.994로 셋이 서로 비슷한 수준이다. bit-exact가 QUANT보다 실칩에 더 가깝지 않다.
- **두 bit-exact 출력끼리:** cos(BE_both, BE_rb) = 0.996~0.998. 두 HAR의 bit-exact 출력은 각자의 실칩 출력보다 서로 훨씬 가깝다.
  - 정상 모델에서도 bit-exact가 실칩을 비트 단위로 재현하지 못한다. 그래서 이 도구로는 "실칩이 계산하는 것"을 따라갈 수 없다.

### 8-4. 판정과 해석

- **규칙상 판정: 판정 불가.**
  - 재현됨: 아니다. rb에서도 BE–실칩이 0.999 아래다.
  - 재현 안 됨: 아니다. both BE–QUANT가 0.999 아래다.
  - 판정 규칙을 정할 때 bit-exact가 실칩과 비트 단위로 맞는다고 가정했는데, 대조군에서 그 가정이 성립하지 않았다.
- **내용상 해석:**
  - DFC의 두 에뮬레이터(SDK_QUANTIZED, SDK_BIT_EXACT)는 B16_both를 모두 정상(FP32에 가까운 쪽)으로 계산한다. 붕괴는 실칩에서만 나타난다.
  - 그러나 bit-exact는 정상 모델(B16_rb)에서도 실칩에 QUANT보다 더 가깝지 않다. 그래서 **원인이 compile 결과물(HEF)인지, 양자화 값에 있지만 실제 하드웨어 연산(누산·포화 등)에서만 드러나는 문제인지는 가를 수 없다.** 용의 지시 5번 문구("컴파일러 계산은 맞고 실칩·런타임 쪽 문제로 보임")는 이 한계와 함께 적는다.
  - HEF 안에서 어느 층이 틀리는지는 이 방법으로 알 수 없다.
- 지시대로 층별 비교와 우회 방법 시험은 하지 않았다.

### 8-5. 다음에 볼 것 (5절 갱신)

- 5절 1번(SDK_BIT_EXACT)은 위 결과로 끝났다.
- 남은 방법은 5절 2~4번이다. 그중 **2번(중간 출력이 있는 HEF로 실칩 중간 출력을 직접 보기)**이 HEF 쪽 문제를 직접 볼 수 있는 유일한 방법이 됐다.
- 4번(보정 다시 하기)은 빌드 1회로 "특정 양자화 값의 우연인지"를 가를 수 있다.

### 8-6. 확인 기록 (4단계)

- **자기 검증:** 판정 스크립트를 결과 전에 커밋했다(`0efaf9a`). 사진 태그 순서가 두 HAR에서 같은지 assert했다.
- **독립 검토** (맥락 없는 서브에이전트, raw npz에서 다시 계산)
  - 표의 48개 cos와 범위가 일치했다. cifar_A:0 rb BE–실칩 한 칸만 반올림 오류(0.9657 → 0.9658)여서 고쳤다.
  - 반영한 지적:
    - (a) `set_bit_exact` 호출 위치와 실제 판정 위치(`inference_flow.py:361~363`) 출처 정정, `get_params_hailo_optimized` 추가
    - (b) 조용한 대체가 없었다는 근거 추가
    - (c) 해석 문구를 "HEF인지 에뮬레이터 한계인지 가를 수 없음"으로 좁히고, 맨 위 요약에도 같은 한정을 붙임
    - (d) 입력 양자화 경로와 확인 못 한 점 추가
    - (e) cos(BE_both, BE_rb) 추가
