# G1 ① 이미지 인코더 ONNX export (2026-10-13)

ROADMAP G1 첫 항목: 이미지 인코더만 ONNX export (opset 17, 입력 1×3×224×224 고정), PyTorch 대비 임베딩 코사인 ≥ 0.9999, 최대 절대 오차 기록, 연산 목록.

## 실행 명령
```bash
cd siglip2
OPENBLAS_NUM_THREADS=1 "$SIGLIP2_PYTHON" s2_export_onnx.py --verify 64 --ops --threads 4 --out ../results/siglip2/G1_onnx_export
```

## 파일 (`$SIGLIP2_ARTIFACTS/onnx`, 커밋하지 않음)
| 파일 | 입력 | 크기 | sha256 |
|---|---|---|---|
| `siglip2_image_fp32.onnx` | batch×3×224×224 (ORT 평가·양자화용) | 354.5 MiB | `333cad08…48840e0` |
| `siglip2_image_fp32_b1.onnx` | 1×3×224×224 고정 (DFC용) | 354.4 MiB | `ce2b66b9…6fd8a70` |

sha256 전체 값은 `export.jsonl`. 같은 코드로 다시 내보내면 sha256이 **완전히 같다** (결정적 export, 아래 3중 확인).

## 결과: PyTorch vs ONNX (CIFAR-100 dev A 앞 64장, ORT 1.25.0 `ORT_ENABLE_ALL`, `export.jsonl` 마지막 줄, 커밋 `ab52093`, dirty=false)
| 파일 | 코사인 최소 | 최대 절대 오차 (정규화 전) | 최대 절대 오차 (정규화 후) | 기준 ≥ 0.9999 |
|---|---|---|---|---|
| 배치 가변 | 0.99999988 | 2.10e-05 | 1.85e-06 | 통과 |
| 배치 1 고정 | 0.99999976 | 2.38e-05 | 1.48e-06 | 통과 |

PyTorch fused attention vs unfused(내보낸 형태): 코사인 최소 0.99999982, 최대 절대 오차(정규화 후) 8.1e-07 → D-019(끄고 내보내기)는 계산을 바꾸지 않는다.

## 연산 목록 (배치 1 고정 파일, 노드 543개, `onnx_ops.json`)
| 구분 | 내용 |
|---|---|
| 패치 임베딩 | `Conv` 1개 (16×16, stride 16) → 196 토큰, 위치 임베딩 더함 (CLS 토큰 없음) |
| LayerNorm | `LayerNormalization` 26개 (opset 17 단일 연산: 블록 12×2 + 마지막 norm + pooling head norm) |
| Attention | 블록마다 QKV `MatMul` → `Split` → QKᵀ `MatMul` → `Softmax` → AV `MatMul` → proj `MatMul`. `Softmax` 13개 = 블록 12 + head 1 |
| MLP GELU | **정확한 GELU (`Erf`)** 13개, tanh 근사 아님 |
| Pooling head | MAP(AttentionPoolLatent): 학습된 latent query 1개 → q/kv `MatMul` → `Softmax` → proj → `LayerNormalization` → MLP(`Erf` GELU) → 768차원. 노드 50개 |
| 남은 동적 연산 | `Shape`·`ConstantOfShape`·`Equal`·`Where`·`Expand`·`Gather` 각 1개 — pooling head의 latent를 배치 크기로 `expand`하는 부분. DFC 파싱에서 문제가 되면 상수 접기(onnx-simplifier 등)로 처리 |
| fused attention | 끄고 내보냄 (D-019). 켜고 내보낸 변형(`fused_variant/onnx_ops.json`)은 배치 1 고정 파일이 노드 686개, `Sqrt` 39·`Cast` 26·`Shape` 14 |

## 3중 확인
1. 자기 검증: 코사인 ≥ 0.9999 assert 통과. 같은 명령을 여러 번 실행(`05524c5`, `615d992`, `ab52093`) → 수치 같음
2. 독립 재계산: 같은 코드로 다른 폴더에 다시 export → 두 파일 모두 sha256 일치 (`reexport_sha256.txt`, `scripts/g1_pipeline.sh` [3/5]). export가 결정적이고, 저장된 ONNX가 커밋된 코드의 산출물임을 확인
3. 독립 검토: 코드 검토 서브에이전트 1회 — "fused vs unfused를 검증하지 않음", "686·sha256 주장에 원본 기록 없음" 지적 → 위 비교와 `fused_variant/`, `reexport_sha256.txt`로 기록 (커밋 `add8dc1`, `84045ff`)

## 결론
SigLIP2 이미지 인코더 float ONNX는 PyTorch와 같다(코사인 0.9999998). DFC parse·optimize·compile은 DFC 5.4.0 설치 후 (휠 대기, D-011).
연산 구성은 Hailo 공식 SigLIP B/16 이미지 인코더(D-018)와 같은 종류(Conv, MatMul, Softmax, LayerNorm, Erf, Add)로, 블록 내부에서 컴파일이 실패할 위험은 낮다고 본다(판정은 사람이 DFC 결과로).
