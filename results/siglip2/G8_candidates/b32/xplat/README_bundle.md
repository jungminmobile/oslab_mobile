# SigLIP2 B/32 예선 제출 모델 — 다른 PC 재현 확인 묶음 (2026-10-23, 쭈리)

예선 제출 모델(D-049, B/32 O-AB)이 다른 CPU(다른 x86, ARM)에서도 같은 결과를 내는지 확인하기 위한 묶음이다.
O-AB는 i7(이 PC)과 285K(준용 O1)에서 데이터셋별 top-1이 0.1~0.2%p 달랐다 → 사진별로 얼마나 다른지 잰다.

## 들어 있는 것
| 경로 | 내용 |
|---|---|
| `models/siglip2_image_fp32.onnx` | B/32 FP32 (원본, 기준) |
| `models/siglip2_image_int8_g6b_b32_m2rot_o1b_…_moq0_all.onnx` | **제출 모델 O-AB** (sha256 `7a01044c…`) |
| `models/siglip2_image_int8_rcmaskbcmlp15at6u5-7at10u9_rothad_…_rr_all.onnx` | 비교용 M2rot (출력 재반올림 있음, `9783d7de…`) |
| `dev200/<set>.npz` | dev 6종 각 앞 200장: x(전처리 끝난 float32, 256×256), y, clf(FP32 분류기), ref(FP32 이미지 임베딩) |
| `portable_eval.py` | numpy + onnxruntime만 쓰는 평가 스크립트 |
| `out_i7/` | 쭈리 PC(i7-1165G7, AVX-512 VNNI, Windows, ORT 1.25.0) 결과 — 비교 기준 |
| `SHA256SUMS.txt` | 파일 sha256 |

## 실행 (필요한 것: Python 3.10+, `pip install numpy onnxruntime==1.25.0`)
```bash
sha256sum -c SHA256SUMS.txt          # Windows: certutil -hashfile <파일> SHA256 으로 하나씩
python portable_eval.py --data dev200 --out out_<이름> --threads 4 \
  --models fp32=models/siglip2_image_fp32.onnx \
           oab=models/siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx \
           m2rot=models/siglip2_image_int8_rcmaskbcmlp15at6u5-7at10u9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
```
- 몇 분 걸린다 (사진 1,200장 × 모델 3개). 다른 무거운 프로그램은 꺼도 되고 안 꺼도 된다(정확도만 잼).
- 결과 `out_<이름>/` 폴더(result.json, pred_*.npz)를 통째로 쭈리에게 보내 주면 사진별로 비교한다.
- CPU 이름과 VNNI 지원 여부(`avx512_vnni`, `avx_vnni`)를 함께 알려 주면 좋다.

## 이 PC 기준값 (`out_i7/result.json`)
| 모델 | 고화질 Δ (200장×5) | cos | CIFAR A |
|---|---|---|---|
| O-AB (제출) | +0.4 | 0.9893 | 71.0 (FP32 71.5) |
| M2rot | −0.1 | 0.9772 | 73.0 |
- 이 값은 이 PC의 입력 변형 실험(변형 없음)과 소수점까지 같다 — 다른 경로로 낸 같은 값.
