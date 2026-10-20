# E11 사전 등록 — 교차 기법 적용 (결과 보기 전 고정, 2026-10-21 06:40 KST)

- 계획서: Claude Docs "E11 교차 기법 적용 실험계획서" (https://claude.ai/artifact/UbR8Nz9iKfz3RTU6skVKc8)
- 결과 기록: Claude Docs "E11 교차 기법 실험 결과 기록" (https://claude.ai/artifact/K9TNxYwJzpWM4EdF74ZMVx)
- 브랜치 `jungmin/e11-cross` (511de3c에서 분기). 이 파일이 들어간 커밋이 사전 등록 해시다.
- 측정 기기: 정민 노트북 Core Ultra 7 258V, 충전 중, 전원 구성표 SAMSUNG MODE(전원 모드 덧씌우기 없음 = 균형), ORT 1.25.0, 정수 커널.

## 1. 실험 (기준 모델에서 한 가지만 바꾼다)
| # | 기법 | 기준 | 바꾸는 것 |
|---|---|---|---|
| S1 | 보정 데이터 출처·장수 | SigLIP2 B/16 M2rot_ort | cifar256 · mix256 · highres64 · highres512 |
| S2 | Bias Correction | SigLIP2 B/16 M2rot_ort | MatMul 뒤 Add bias에 채널별 평균 쏠림 보정 (+ 다른 사진 BC′) |
| S3 | 정수계획 재배분 | SigLIP2 B/16 M2rot_ort | MatMul 단위 INT8/FP32를 지연 예산별로 선택 |
| R1 | reduce_range | RN50 B2 | 가중치 7비트 범위 |
| R2 | 동적 양자화 | RN50 FP32 q1 | ORT quantize_dynamic: 기본(Ddef: per_channel 0) · 채널별+rr(Dpc) |
| R3 | MatMulNBits 4비트 | RN50 B2 | AttnPool q·k·v MatMul 가중치 4비트(블록 32, accuracy_level 4) |
| R4 | 회전 | RN50 B2 | AttnPool v→c_proj 오프라인 회전 (+ 입력 하다마드) |

## 2. 측정
- 정확도: dev 전체(고화질 5종 + CIFAR A), 정수 커널 4T. 기준 대비 짝 부트스트랩 1000회, 데이터셋별 층화, 95% 구간.
- 지연: bench_speed n=200, warmup 20, repeats 3, 1·2·4T × 3세션(세션 사이 2분), FP32·기준·새 모델을 한 실행에서 번갈아. 3세션 중앙값.
- 메모리: bench_mem 4T peak RSS. 크기: 파일 크기.
- 재현: 기준 모델을 같은 코드로 다시 만들어 sha256을 확인한다.

## 3. 판정 규칙 (계획서 그대로)
- **채택 후보 (속도·크기 이득):** ① 기준 대비 95% 구간이 0을 포함하거나 0보다 큼 ② 1·2·4T 모두 기준보다 느리지 않음(3세션 중앙값) ③ 4T −5% 이상, 또는 peak RSS −5% 이상, 또는 파일 −10% 이상.
- **채택 후보 (정확도 이득):** ① 기준 대비 하한 > 0 ② 4T 지연이 기준의 +5% 이내 ③ 모든 스레드에서 FP32보다 빠름.
- **효과 없음 / 해로움:** 위 둘 다 아님. 상한 < 0이면 "해로움".
- cos·CIFAR A는 기록만 한다. 여러 설정을 재면(S1, S3) 조건을 만족하는 것 중 정확도 점추정이 가장 높은 하나만 후보로 올린다.
- R2(동적)는 Hailo 미지원이라 기준선으로만 기록하고 판정하지 않는다.
- SigLIP2 채택 후보는 팀에 교체를 제안한다(교체 시 test 재사용은 팀 결정). RN50 결과는 비교 자료로만 기록한다.
- 결과 후 기준 변경은 이유를 적고 정민의 승인을 받는다.

## 4. 계획서와 다른 점 (결과 전 기록)
1. **순서:** SigLIP2 B/16 M2rot **FP32** ONNX(`16ef696e…`)가 노트북과 GitHub Release에 없다(M2rot_ort INT8만 있음). S1·S2·S3는 이 FP32 파일이 필요하다. 그래서 RN50(R1~R4)을 먼저 하고, SigLIP2는 파일을 받은 뒤 한다. 노트북에서 s2_regcache·s2_rotate로 직접 만드는 것은 용의 이전 요청(노트북에서 SigLIP2 모델을 새로 만들지 말 것)이 있어 정민 확인 전에는 하지 않는다.
2. **R3 범위:** c_proj는 Gemm이라 MatMulNBits 대상이 아니다. q·k·v MatMul 3개만 4비트로 하고 c_proj는 B2처럼 INT8로 둔다.
3. **R2 연산 범위:** "한 줄" 기준선이라 연산 종류를 제한하지 않는다(ORT 기본, Conv 포함). 제외 노드도 없다.
