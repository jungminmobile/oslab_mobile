# G4 첫 구성: Hailo DFC 16비트 보호 (2026-10-13)

DFC 5.4.0, 최적화 레벨 0(GPU 없음), 보정 64장(`calib_highres64_s0_nhwc.npy`, G1과 같음), 입력은 G1 HAR(`siglip2_b16_224.har`).
평가: **SDK_QUANTIZED 에뮬레이션**, Pets dev 앞 64장, 기준 = FP32 ONNX 임베딩(top-1 95.31). 컴파일(HEF)은 아직 안 함.
**결론: 잔차 합(`ew_add`)만 16비트로는 부족(cos 0.72)하고, 블록 0·5의 MLP(공식 패턴, C)나 블록 5~11 fc2(B)까지 16비트로 두면 cos 0.94~0.96으로 회복된다 — ORT(G3)와 같은 위치가 민감하다.**

## 실행 명령 (WSL Ubuntu-22.04)
```bash
bash ~/g4_run.sh     # siglip2/hailo/g4_run.sh: 구성마다 optimize(--model-script) → s2_dfc_emulate.py --quantized-only
```
모델 스크립트: `siglip2/hailo/g4_*.alls` (이 폴더에도 사본). 층 이름 ↔ ONNX 노드: `siglip2/hailo/layer_map_siglip2.csv`.
공통 머리: `optimization_level=0`, `layer_norm_decomposition`, `equalization=disabled`, `bit_decomposition_mode=uniform_precision`, `calibset_size=64` (공식 SigLIP `.alls`와 같음, D-018)

## 결과 (`emulate.jsonl`)
| 구성 | 16비트(a16_w16)로 둔 층 | cos 평균 | cos 최소 | top-1 | 에뮬레이션 시간 |
|---|---|---|---|---|---|
| 기본 (G1, 모델 스크립트 없음) | 없음 | 0.121 | −0.161 | 3.12 | 1454 s |
| A | `conv1`(패치) + `ew_add*`(잔차 합 26개) | 0.717 | 0.549 | 40.62 | 2553 s |
| **C (공식 패턴)** | A + `conv4, conv5, conv24, conv25` (블록 0·5의 fc1·fc2) | **0.963** | 0.888 | 92.19 | 3771 s |
| B | A + 블록 5~11 fc2 (`conv25…conv49`, 7개) | 0.936 | 0.837 | **93.75** | 6775 s (메모리 부족·스왑) |

- C의 층 목록은 Hailo 공식 `siglip_b_16_image_encoder.alls`(SigLIP 1세대)에서 그대로 가져왔다. 우리 층 번호 규칙(conv 2+4b=qkv, 3+4b=proj, 4+4b=fc1, 5+4b=fc2)에서는 블록 0·5의 MLP에 해당한다 — **공식 모델의 번호 규칙이 같다는 것은 가정**이다.
- 블록 5는 G1·G3에서 massive activation이 처음 나타나는 블록이다. 공식 설정과 우리 ORT 민감도 분석이 같은 블록을 가리킨다.
- 64장이라 top-1(C 92.19 vs B 93.75 = 1장 차이)로는 순위를 가릴 수 없다. cos는 C가 높다.
- B 에뮬레이션은 메모리 11.6 GB + 스왑 12 GB로 매우 느렸다(WSL 11 GB). 다음에는 배치를 줄인다.

## 3중 확인
1. 자기 검증: optimize rc=0 ×3, 출력 shape (64, 768) assert, HAR sha256(`g4_20261013_165132/har_sha256.txt`)
2. 독립 재계산: 같은 기준 임베딩으로 G1의 float 에뮬레이터가 cos 1.0 → 차이는 양자화에서만 온다
3. 독립 검토: PR에서 예정

## 다음
- A·B·C 조합(블록 0·5 MLP + 블록 9 fc2 등)과 컨텍스트 수·HEF 크기 (컴파일 2시간+/구성)
- 표본을 늘린 에뮬레이터 정확도(배치 축소)
- D-027: 최종 후보 `.alls`로 정민 노트북 교차 확인
