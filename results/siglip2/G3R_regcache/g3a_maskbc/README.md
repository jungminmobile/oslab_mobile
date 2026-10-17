# 마스크 상수 broadcast 형식 (`--mask-form bcast`, 2026-10-18)

## 결론: **채택 — (a)(b)(c) 모두 통과, FP32 크기 354.51 MiB(R1 354.52와 같음)**. 이후 INT8·G3b는 이 그래프로 한다.

- 바꾼 식: K' = K + Σ_j m_j ⊙ (K_c,j − K), V' 같음. m_j는 (1,1,196,1) one-hot, K_c,j는 헤드별 1행 (1,12,1,64).
  요청은 m × K_c를 broadcast로 곱하는 것이었지만, 둘 다 상수라 `torch.onnx.export`의 상수 접기(그리고 ORT 양자화 전처리·DFC 단순화)가 196토큰 크기 텐서로 미리 계산해 버린다.
  그래서 K가 들어가는 꼴로 바꿨다. sink 마스크가 서로 겹치지 않으므로 full 형식(`../g3a_mask`)과 수학적으로 같다.
- 기본값 `--mask-form full`은 이전과 같다: 커밋 `2a3fa4d`에서 다시 export한 full 파일 sha256이 `b2038137…`·`38b3b521…`로 이전과 같다.

| 확인 | 결과 |
|---|---|
| selftest (무작위 가중치) | bcast vs mean 최대 차 7.5e-7 |
| (a) bcast vs R1 FP32 cos 최소 (dev 64장) | **0.9999998** (최대 차 2.4e-7) → 통과 (`maskcheck.jsonl`) |
| ORT vs PyTorch (64장) | 배치 가변 0.9999998, b1 0.9999998 (`export.jsonl`) |
| (b) 연산 (b1, `ops_compare.json`) | Concat·Slice·Expand·Where 1·1·1·1 = 원래 FP32. 늘어난 것 Sub +24·Mul +38·Add +24·Identity +15. sink 항 12개(블록 5~8 ×1, 9~11 ×2, 풀링 ×2) × K·V = 24, 잔차 마스크 Mul 14 → 설계와 같음. 196 크기 상수는 one-hot 마스크 벡터뿐. 194·195 차원 없음 |
| (c) `hailo parser --hw-arch hailo10h` | **rc=0**, end node `/trunk/Gather`(모델 출력). DFC 단순화 ONNX 352.24 MiB (full은 361.35) → 상수가 다시 커지지 않음 (`parse.log`) |

| 파일 | MiB | sha256 |
|---|---|---|
| `siglip2_image_fp32_rcmaskbc55at5-13at9.onnx` | 354.51 | `export.jsonl` |
| `siglip2_image_fp32_rcmaskbc55at5-13at9_b1.onnx` | 354.48 | `export.jsonl` |
| `siglip2_rcmaskbc55at5_13at9.har` (`~/siglip2_work/hailo_g3r_maskbc/`) | 353.2 | `fbf722b51272b450…` |
