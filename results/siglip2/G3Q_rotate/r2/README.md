# G3Q-R2: 헤드 안쪽 회전(R2)으로 attn proj 입력의 INT8 손실 줄이기 (ORT, 설계 2026-10-19 — 1~4절은 결과 전에 고정)

## 1. 근거
- 진단 2(상위 README 10-8): 블록 attn proj 12개(E3)가 남은 cos 손실의 39%를 가진다 (FP32로 두면 cos +0.0175, CIFAR A −4.0 → −0.5). FP32로 남기면 4스레드 지연 +9.8%, 크기 +19% (10-10).
- attn proj의 입력은 헤드 출력을 이어붙인 768차원(12헤드 × 64)이고, 기존 회전(잔차 768, Q)이 돌리지 않는 공간이다. 이 공간을 헤드별 직교 행렬로 돌려 채널 쏠림을 퍼뜨린다 (SpinQuant의 R2). 연산 종류·개수는 그대로라 비용 없음.

## 2. 변환 (`siglip2/s2_rotate.py --r2 had64`, 기본 꺼짐)
- R (64×64) = diag(s)·H₆₄/8, H₆₄ = Sylvester Hadamard, s = ±1 부호 64개 (torch 생성기 시드 0). 모든 블록·모든 헤드에 같은 R. 직교 (RᵀR = I).
- timm `Attention` (코드로 확인): `qkv(x).reshape(B, N, 3, H, d)` → qkv 출력 열 = `[q | k | v] × 헤드 h × 64`, V 헤드 h = 열 1536 + 64h … 1536 + 64h + 63. 출력 `(P·V).transpose(1,2).reshape(B, N, H·d)` → proj 입력 열 64h … 64h + 63 = 헤드 h. `attn.norm` Identity, gate 없음 (assert).
- 블록마다 헤드 h마다: **qkv의 V 부분** `W_v,h ← Rᵀ·W_v,h`, `b_v,h ← Rᵀ·b_v,h` (출력 v_h → v_h·R), **attn.proj** 입력 열 `W_o[:, h] ← W_o[:, h]·R`. P·(V·R)·Rᵀ·W_oᵀ… 즉 (P·V·R)·(W_o,h·R)ᵀ = P·V·W_o,hᵀ로 정확히 동치 (R·Rᵀ = I). q·k는 건드리지 않는다.
- 풀링 헤드(MAP) attention은 이번에 건드리지 않는다.
- 두 구성: **rothad_r2** = rothad(잔차 회전) 위에 R2, **r2** = 회전 없는 원래 모델 위에 R2. 파일 `siglip2_image_fp32_rothad_r2had64.onnx`, `siglip2_image_fp32_r2had64.onnx` (+ `_b1`).

## 3. 확인 (INT8 전에)
| 항목 | 기준 |
|---|---|
| (a) FP32 동치: ORT 회전 ONNX vs 원래 FP32 임베딩(dev 캐시 ref_fp32), dev 64장 (Pets·Caltech 앞 32장씩) | cos 최소 ≥ **0.99999** (두 구성, 배치 가변·b1). 미달이면 구현 버그 → 멈춤 |
| (b) 연산 종류·개수, 연산 순서·범위 이름·initializer 이름 | 원래 FP32 export와 같음 |
| (c) attn proj 입력(=헤드 출력 이어붙인 것)의 채널별 max의 최대/중앙값, max/p99.9 | 기록 (R2 전보다 줄었는지) |
- 무작위 가중치 selftest(`s2_rotate.py selftest --r2 had64`)로 먼저 동치·ORT·연산 개수를 확인한다.

## 4. INT8과 판정 규칙
- 양자화: R0와 같은 명령 + `--src` (fc2L2 U8+rr, percentile 99.999, highres 256 s0, hc16, `--exclude /blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/`).
- 스크리닝: dev 6×200, 정수 커널 4스레드, R0·rothad·E3를 같은 실행에 넣는다. 표: 구성 / 정수 MatMul / 고화질 cos / Δtop-1 / CIFAR A Δ.
- **판정: rothad_r2의 고화질 cos가 rothad보다 +0.005 이상 높거나, CIFAR A top-1이 rothad보다 +2%p 이상 좋으면** → dev 전체 짝 비교 1회 (rothad_r2 vs R0, 층화 짝 부트스트랩 시드 0·1000회). **하한 > 0이면 "예선 ONNX 후보"로 팀에 제안** (교체는 팀 결정, 최종 test 평가를 다시 해야 함을 같이 적는다). 아니면 진단 기록만 하고 끝.
- r2(회전 없는 R2)는 기록만 한다 (판정은 rothad_r2로).
- 연산 구성이 같으므로 지연은 재지 않는다. 양자화·스크리닝 전 `free -g` 여유 ≥ 10 GB 확인.
