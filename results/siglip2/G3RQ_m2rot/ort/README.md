# M2rot ONNX를 ORT(CPU)에서: 예선 후보 확인 (설계 2026-10-19)

> **용 제안, 준혁 동의 전, 결과 보기 전 고정.** 결과를 본 뒤에는 이 기준을 바꾸지 않는다. **dev만 쓴다 (test 사용 금지).**

## 1. 배경
- 공지상 본선은 "예선 모델을 Hailo로 컴파일"한다. 예선 ONNX도 NPU에서 검증된 M2rot(D-043, Hailo 최종)으로 맞출 수 있는지 사전 측정한다.
- 원칙 (용 10/19): 기존 "R0보다 좋아야 교체"가 아니라 **"CPU에서 크게 나쁘지 않으면 M2rot"**.

## 2. 구성
| 이름 | 만드는 법 |
|---|---|
| **M2rot_ort** (주 구성) | M2rot FP32 ONNX `siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad.onnx` (sha256 `16ef696e…`, G3RQ `export.jsonl`)를 `s2_quantize.py --src`로 **MatMul 전부 INT8 (fc2 예외 없음)**, U8 + reduce_range, highres 256장 시드 0, percentile 99.999, hc16. R0와 같은 양자화 옵션 (`--exclude`만 없음) |
| **R0** (비교 기준) | Release ORT 최종 fc2L2 U8+rr, sha256 `5eee7839…` (D-030). 파일은 Release `siglip2-xplat-20261015`의 것 (`~/siglip2_work/xplat/models/`, sha256 확인). x86 VNNI 두 세대에서 출력이 비트 단위로 같아(G3_xplat) 이 PC에서 그대로 돌린다 |
| FP32 | Release `siglip2_image_fp32.onnx` (`333cad08…`) |

## 3. 판정 (예선 후보 조건, 모두 만족)
- **① M2rot_ort − R0 고화질 Δtop-1 95% 구간 하한 ≥ −0.5%p** (dev 전체, `s2_paired_ci.py`, 층화 짝 부트스트랩 시드 0·1000회)
- **② 지연이 FP32보다 느리지 않음** — 측정한 모든 스레드 수(1·2·4)에서, 3세션 중앙값 기준
- 기록만 (판정에 안 씀): 크기·지연·메모리가 R0보다 좋은지, FP32 대비 Δtop-1, 고화질 cos, CIFAR A Δ.
- 둘 다 만족하면 "예선 후보" → 팀(준혁) 결정. 교체 시 최종 test 평가를 다시 해야 한다 (test는 이번에 쓰지 않음).

## 4. 순서와 관문
| 단계 | 내용 | 관문 |
|---|---|---|
| 0 | `--src` 옵션은 이미 있음 (`707c031`). 옵션을 끈 기본 경로로 R0 재생성 (태그 `idcheck_ort`, 기존 파일 덮어쓰지 않음) | 이 PC의 기존 재생성 파일 `410d7483…`와 sha256이 같으면 코드 동일성 통과. Release `5eee…`와는 이 PC에서 원래 다르다 (G3R identity: 활성 눈금 미세 차이, CPU 차이) — 기록만 |
| a | M2rot FP32 ONNX vs 원래 FP32 임베딩 cos 최소 (ORT, dev 64장 + dev 6×200) | **≥ 0.999** (M2 근사 확인). 미달이면 멈춤 |
| b | 양자화 → 스크리닝 (dev 6×200, 정수 커널, R0·FP32와 같은 실행) | **고화질 cos ≥ 0.99**. 미달이면 멈추고 원인 가설 2개 이상과 함께 보고 |
| c | dev 전체 짝 비교 (FP32, R0, M2rot_ort) | 3절 ① |
| d | 지연·크기·메모리: FP32·R0·M2rot_ort를 같은 세션에서 번갈아 (`s2_bench_speed.py --warmup 20 --n 200 --repeats 3`, 배치 1, 스레드 1·2·4 × 3세션, WSL 안 게이트: 다른 python·hailo 0 + 1분 평균 부하 < 2, Windows 부하 확인 못 함), peak 메모리 `s2_model_stats.py` | 3절 ②. **비공식** (이 PC) |
- 모든 결과 행에 커밋 해시·dirty를 남긴다.
