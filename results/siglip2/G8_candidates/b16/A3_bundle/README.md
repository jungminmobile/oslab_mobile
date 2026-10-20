# G8 A3: 최종 B/16(M2rot_ort)용 소규모 평가 묶음과 기대값 (보고서 6.3) — 2026-10-20

목적: 학술부나 다른 기기가 **저장소·데이터셋 원본 없이** numpy + onnxruntime만으로 같은 평가를 돌려 볼 수 있는 작은 묶음.
- 만든 명령: `bash siglip2/scripts/g8_b16_a3.sh` (커밋 ac8744e). 묶음 위치 `D:\mokerton\artifacts\siglip2\share\g8_b16_final\` (1.2 GB)
- 구성: `dev200/<set>.npz` 6개 (dev 6종 각 앞 200장: 전처리된 입력 x, 정답 y, FP32 텍스트 분류기 clf, FP32 이미지 임베딩 ref), `models/` FP32·M2rot_ort, `portable_eval.py`, `SHA256SUMS.txt`
- **새로 올릴 데이터·모델이 없다:** dev200 6개는 10/15 공개한 `siglip2-xplat-20261015`의 dev200과 sha256이 모두 같고(`SHA256SUMS.txt` 대조), M2rot_ort는 `siglip2-m2rot-ort-20261019`(e84c3727…), FP32는 `siglip2-xplat-20261015`에 있다. 다른 기기에서는 두 Release를 받아 같은 폴더 구조로 두면 된다.
- 단 `portable_eval.py`는 10/15 Release 것(sha256 `e88d…`)과 지금 묶음 것(`5e63…`)이 다르다. 입력 크기 확인 줄 하나만 바뀌었다(224만 → 224·256, G8 B/32용). B/16 결과는 같다. 묶음의 `SHA256SUMS.txt`와 대조하려면 저장소의 `siglip2/scripts/portable_eval.py`를 쓴다.
- 실행 (묶음 폴더에서, 저장소 코드 없이):
```bash
python portable_eval.py --data dev200 --out out --threads 4 \
  --models fp32=models/siglip2_image_fp32.onnx m2rot_ort=models/siglip2_image_int8_rcmaskbcmlp55at5-13at9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
```

## 기대값 (이 PC i7-1165G7 AVX-512 VNNI, ORT 1.25.0, 4스레드, `out_this_pc/result.json`)

| 데이터셋 (각 200장) | FP32 top-1 | M2rot_ort top-1 | cos |
|---|---|---|---|
| CIFAR A | 74.5 | 72.0 | 0.912 |
| Pets | 93.5 | 93.0 | 0.983 |
| Flowers | 85.0 | 86.5 | 0.980 |
| Cars | 93.5 | 94.5 | 0.971 |
| Food | 87.5 | 86.0 | 0.976 |
| Caltech | 95.0 | 96.0 | 0.968 |
| **고화질 5종 평균** | 90.90 | 91.20 (**Δ +0.3**) | **0.9756** |

- 200장 표본이라 Δtop-1은 ±1%p 정도 흔들린다. 같은 파일이면 x86 VNNI PC에서는 위 값과 **같아야** 한다 (i7·258V 비트 단위 일치, 285K 요약값 일치 — G6_m2rot_ort). dev 전체 값은 −0.23 [−0.64, +0.18].
- 저장소 경로(`s2_screen.py`, A2)와 같은 값이다: cos 0.9756, Δ +0.3.

## 3중 확인
1. 자기 검증: 묶음 안 M2rot_ort sha256 = 배포 파일 e84c3727… (스크립트에서 grep 확인), rc=0.
2. 독립 재계산: 같은 파일을 저장소 경로(A2 `s2_screen.py`)로 잰 값과 같음 (cos 0.9756, Δ +0.3).
3. 독립 검토: [G8 끝난 뒤 한 번에]
