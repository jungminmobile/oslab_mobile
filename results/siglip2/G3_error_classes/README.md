# 양자화로 생긴 새 오답은 어느 클래스에 몰리나 (2026-10-15 밤)

**결론: 몰리지 않는다. 새 오답은 여러 클래스에 1~4장씩 흩어져 있고, 대부분 원래도 헷갈리는 아주 비슷한 클래스 쌍에서 뒤집힌다. ORT 최종과 DFC는 같은 사진에서 틀리는 경우가 적다.** → 양자화가 특정 클래스를 망가뜨린 것이 아니라, 작은 임베딩 오차가 경계 근처 사진의 순위를 바꾼 것으로 본다.

## 왜 했나
보고서 "결과 해석"용. DFC D256L1은 Food에서 −2.64%p로 가장 많이 떨어졌다 (`../G4_dfc_full`). 손실이 일부 클래스에 몰리면 그 클래스를 보호하는 대책을 생각할 수 있고, 흩어져 있으면 전체 오차 크기 문제다.

## 방법
- `siglip2/s2_error_classes.py`: dev 전체 사진별 예측(이미 저장된 npz)만 사용, 모델 실행 없음. 커밋 `061b001`, 결과 `error_classes.json`.
- 대상: ORT 최종 fc2L2 U8 + reduce_range (`../G3_cumulative/qkv_precision/full`), DFC D256L1 (`../G4_dfc_full`). 기준은 같은 사진의 FP32 예측.
- 새 오답 = FP32 정답 → 양자화 오답, 새 정답 = 그 반대. 순손실 = (새 정답 − 새 오답) / 장수.

## 결과 (dev 전체)

| 데이터셋 | ORT U8+rr 새 오답 / 새 정답 (순 %p) | DFC D256L1 새 오답 / 새 정답 (순 %p) | 둘 다 틀린 사진 |
|---|---|---|---|
| Pets (990) | 16 / 10 (−0.61) | 21 / 17 (−0.40) | — |
| Flowers (957) | 28 / 13 (−1.57) | 23 / 9 (−1.46) | 8 |
| Cars (967) | 17 / 13 (−0.41) | 27 / 16 (−1.14) | 4 |
| Food (985) | 27 / 10 (−1.73) | 39 / 13 (−2.64) | 11 |
| Caltech (1024) | 6 / 10 (+0.39) | 15 / 12 (−0.29) | — |
| CIFAR A (1000) | 90 / 48 (−4.20) | 102 / 41 (−6.10) | — |

- 한 클래스에서 순손실이 가장 큰 경우도 2~4장이다 (Food: hummus·grilled cheese·chicken wings·tacos 각 2장).
- 뒤집힌 쌍의 예 — 원래 비슷한 클래스끼리다:
  - Flowers: marigold → english marigold (ORT·DFC 모두 3장), geranium → pelargonium
  - Food: beef tartare → tuna tartare, tacos → chicken quesadilla, chicken wings → pork chop
  - Cars: Acura TL Sedan 2012 → Acura TL Type-S 2008, Corvette ZR1 → Corvette Z06, Spyker C8 Convertible → Coupe
- 양자화가 FP32 오답을 고친 사진(새 정답)도 적지 않다. 순손실은 둘의 차이다.

## 해석
- 손실이 넓게 흩어져 있고 비슷한 클래스 사이에서만 뒤집히므로, **특정 클래스 보호보다 전체 임베딩 오차를 줄이는 것**(보호 층 선택·보정)이 맞는 방향이다.
- ORT와 DFC가 틀리는 사진이 많이 다르다 (Food 27·39장 중 11장만 겹침). 오답이 경계 근처의 우연한 뒤집힘이라는 해석과 맞고, 200장 표본 값이 흔들렸던 이유(`../G4_dfc_full` 4절)와도 맞는다 (추정).
- Food·Flowers가 많이 떨어지는 것은 세밀한 구분이 필요한 클래스 쌍이 많기 때문으로 보인다 (추정).
