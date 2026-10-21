# 보고서 v4 검토에서 나온 추가 확인 (2026-10-21 17:51~18:35)

실행: `bash siglip2/scripts/g8_final_checks.sh` (커밋 0dacc5b, dirty 아님), 이 PC i7-1165G7, ORT 1.25.0

## E1. 최종 배포 파일(e84c3727…) 오류 클래스 분석
- dev 전체 다시 실행: 고화질 Δ **−0.23 [−0.64, +0.18]**, cos 0.9757 → 10/19 기록(`G6_m2rot_ort/dev_full`)과 같음 (`E1_devfull/paired_ci.jsonl`, 사진별 예측 `pred_*.npz`)
- 오류 클래스 (`E1_error_classes/error_classes.json`, FP32는 맞고 INT8은 틀린 사진 = 잃음, 반대 = 얻음)

| 데이터셋 | 장수 | 잃음 | 얻음 | 순변화 %p | 잃은 클래스 수 | 자주 뒤집힌 쌍 (정답 → 오답, 장수) |
|---|---|---|---|---|---|---|
| Pets | 990 | 15 | 11 | −0.40 | 10 | american bulldog → staffordshire bull terrier (2) |
| Flowers | 957 | 13 | 9 | −0.42 | 9 | petunia → mexican petunia (2), marigold → english marigold (2) |
| Cars | 967 | 8 | 8 | 0.00 | 7 | Bentley Continental GT Coupe 2012 → 2007 (2), Spyker C8 Convertible → Coupe (1) |
| Food | 985 | 16 | 12 | −0.41 | 14 | guacamole → ceviche·hummus (각 1) |
| Caltech | 1024 | 9 | 10 | +0.10 | 7 | wild cat → leopards (2), electric guitar → mandolin (2) |
| CIFAR A | 1000 | 73 | 30 | −4.30 | 42 | (클래스 번호) 64 → 36 (3), 59 → 52 (3) |

- 결론: R0 때와 같은 모양이다. 고화질 5종에서 잃은 사진은 클래스마다 1~2장씩 흩어지고, **원래 비슷한 쌍(같은 차종의 연식, 비슷한 꽃·견종)**에서 뒤집힌다. 잃은 만큼 얻기도 한다(잃음 61 / 얻음 50). CIFAR는 잃은 사진이 42개 클래스에 걸쳐 많다(저해상도).

## E2. 준용 Release 파일로 재현 (준용 PC 빌드와 같은 파일이 나오는가)
준용이 Release `siglip2-m2rot-ort-20261019`에 올린 파일 4개(sha256 기록과 일치 `E2_jy_files.txt`)로:

| 시험 | 결과 | 뜻 |
|---|---|---|
| ① 준용 M2rot FP32(16ef696e…) → 이 PC에서 INT8 양자화만 | `f99b652b…` ≠ 배포 `e84c3727…` | **양자화(보정) 단계도 PC마다 다르다** — 보정 때 CPU로 FP32 추론해 활성값 범위를 재기 때문 (준용 예상과 같음) |
| ② 준용 회전 행렬·캐시 → 이 PC에서 M2rot FP32 export | `f72cc6bc…` ≠ 준용 `16ef696e…` | 회전을 가중치에 곱하는 torch 계산도 PC마다 다르다 |

- 결론: 빌드 단계 어디에서 시작해도 다른 PC에서는 같은 파일이 나오지 않는다 → **채점·재현은 배포한 INT8 파일로 한다** (보고서 9.4). 배포 파일 자체는 x86 세 세대에서 같은 출력을 낸다(G6_m2rot_ort).
- (참고) 같은 PC 안에서는 비트 단위로 같은 파일이 나온다 (A2).
