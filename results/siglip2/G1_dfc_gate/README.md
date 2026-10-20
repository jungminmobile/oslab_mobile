# G1 컴파일 게이트: SigLIP2 B/16 224 → Hailo-10H (DFC 5.4.0) (2026-10-13)

**결과: 전 구간 컴파일 성공 (HEF 생성, 입력 `image`부터 pooling head 출력까지 모두 칩 안, 호스트로 떨어진 구간 없음).**
ROADMAP G1 판정표의 "전 구간 컴파일 성공 → G2 진행"에 해당한다. **판정은 사람(쭈리)이 한다** (CLAUDE.md 7절).
단, DFC 기본 양자화(모델 스크립트 없음, 최적화 레벨 0)의 정확도는 붕괴했다 → G4에서 16비트 보호가 필수.

## 실행 명령
```bash
# Windows (siglip2 폴더): 보정 사진 (ORT 보정과 같은 풀·같은 뽑기, highres 64장 시드 0, NHWC)
"$SIGLIP2_PYTHON" hailo/s2_dfc_calib.py --n 64
# WSL Ubuntu-22.04 (D-011)
wsl -d Ubuntu-22.04 -e bash -lc "tr -d '\r' < /mnt/d/mokerton/oslab_mobile/siglip2/hailo/g1_dfc.sh | bash"
# 에뮬레이터 확인 (WSL, hailo10_venv)
python s2_dfc_emulate.py --har siglip2_b16_224_optimized.har --data pets_dev --k 64 --out .../G1_dfc_gate/emulate.jsonl
hailo profiler siglip2_b16_224_compiled.har --hef siglip2_b16_224.hef --out-path profiler.html --csv profiler.csv
```

## 환경 (`20261013_124213/env.txt`)
DFC **v5.4.0**, Python 3.10.12, WSL2 Ubuntu-22.04 (커널 6.6.87.2), 8스레드, WSL 메모리 11 GB, **GPU 없음**, 커밋 `e519ef0`.
입력 ONNX `siglip2_image_fp32_b1.onnx` sha256 `ce2b66b9…6fd8a70` (G1 export와 같은 파일), 보정 `calib_highres64_s0_nhwc.npy` sha256 `9d7ce83b…c535bc`.

## 단계별 결과 (`20261013_124213/steps.txt`)
| 단계 | 결과 | 시간 | 비고 |
|---|---|---|---|
| parse | 성공 | 139초 | 원본 그래프로는 실패 → DFC가 자동으로 onnxsim 단순화 후 재시도해 성공. 실패 원인은 로그에 없음 (추정: pooling head의 `Shape`·`Where`·`Expand` 동적 연산, G1 export README). 393층, 끝 노드 `/trunk/attn_pool/Gather` |
| optimize | 성공 | 1121초 | **GPU 없음 + 보정 1024장 미만 → 최적화 레벨 0으로 자동 하향**. LayerNorm Decomposition, Statistics Collector, Matmul Equalization 적용. 모델 스크립트 없음(기본) |
| compile | 성공 | 벽시계 8341초(2시간 19분), 컴파일러 보고 2시간 15분 56초 | 단일 컨텍스트 불가(자원 123/80) → 다중 컨텍스트. **컨텍스트 18개** (프로파일러). HEF 69.2 MB, sha256 `6733511…3a17540` |

산출물: `$SIGLIP2_ARTIFACTS/dfc/20261013_124213/` (HAR 3개·HEF·전체 로그·profiler.html). 이 폴더에는 작은 텍스트만(큰 로그는 앞뒤만).

## 에뮬레이터 확인 (`emulate.jsonl`, Pets dev 앞 64장, 기준 = FP32 ONNX 임베딩)
| 실행 | cos 평균 | cos 최소 | top-1 |
|---|---|---|---|
| FP32 ONNX (기준) | 1 | 1 | 95.31 |
| DFC float (SDK_FP_OPTIMIZED) | **1.00000** | **1.00000** | 95.31 |
| DFC 8비트 에뮬레이션 (SDK_QUANTIZED, 기본) | **0.121** | −0.161 | **3.12** |

- float 그래프가 ONNX와 같다 → 파싱·단순화·그래프 변환이 계산을 바꾸지 않았다 (VERIFICATION D-G1 기능 확인).
- 기본 8비트는 붕괴: ORT INT8 전부(고화질 cos 0.445)보다도 나쁘다. 양자화 오차와 최적화 오차가 분리됐다(PRD E8a 방식): 오차는 전부 양자화에서 온다.
- Hailo 공식 SigLIP B/16·SigLIP2 B/32 `.alls`는 잔차 합(`ew_add*`)과 입력 쪽 conv를 `a16_w16`으로 둔다(D-018). G3에서 ORT 민감 위치가 잔차 흐름이었던 것과 같은 방향 → G4의 출발 구성.

## 성능 (프로파일러, `profiler.csv`)
컨텍스트 18개. 처리량·지연은 N/A (HailoRT 런타임 데이터가 필요 — 실칩 측정은 본선, D-015).
참고: Hailo 공식 siglip_b_16_image_encoder(224)는 33.0 FPS(배치 1, 공식 `.alls`, Model Zoo 표). 컨텍스트 수가 많을수록 컨텍스트 전환 비용이 커진다.

## 3중 확인
1. 자기 검증: 세 단계 모두 rc=0, HEF·HAR sha256 기록(`files.txt`). 입력 ONNX sha256이 G1 export와 일치
2. 독립 재계산: DFC float 에뮬레이터 출력 vs ORT FP32 ONNX 출력 cos 1.0 (다른 런타임으로 같은 계산), 컨텍스트 수를 프로파일러로 확인(컴파일 로그의 분할 탐색 표기와 별도)
3. 독립 검토: PR에서 서브에이전트 검토 예정

## 남은 것
- **G1 판정 (사람)**: 전 구간 컴파일 성공 → 계획상 SigLIP2 계속(G2~)
- D-011 교차 확인: 같은 입력으로 정민 노트북(DFC 5.4.0)에서 한 번 컴파일해 비교 — 아직 안 함
- 에뮬레이션은 64장(CPU, 양자화 1454초). G4 비교는 표본을 정해서 다시
