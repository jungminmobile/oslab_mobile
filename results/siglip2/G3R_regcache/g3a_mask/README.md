# G3a 재시도: 제자리 마스크 그래프 (D-041, 2026-10-18, 연구실 데스크탑 WSL Ubuntu-22.04)

## 결론: **DFC 파싱 통과** (D-041 확인 기준 (a)(b)(c) 모두 통과). 지시대로 여기서 멈춤 — INT8 재평가·G3b·컴파일은 하지 않았다.

| 기준 (D-041, 실행 전 고정) | 결과 | 판정 |
|---|---|---|
| (a) 마스크 모델 vs R1(토큰 삭제 + 평균 캐시) FP32 임베딩 cos 최소 ≥ 0.99999 (dev 64장, Pets·Caltech 32장씩) | **0.9999998** (최대 절대 차 2.4e-7). 둘 다 원래 FP32 ONNX 대비 cos 최소 0.99940으로 같음 | 통과 |
| (b) Concat·Slice·Expand·Where가 원래 FP32보다 늘지 않음, 토큰 196 유지 | b1: Concat 1·Slice 1·Expand 1·Where 1 = 원래 FP32 b1과 같음 (R1 concat b1은 19·5·17·17). 늘어난 것은 Mul +30·Add +16·Identity +11뿐. Softmax 13개 모두 196 토큰(풀링 헤드 1×196), 194·195 차원 없음 | 통과 |
| (c) b1 `hailo parser onnx --hw-arch hailo10h` | rc=0, HAR 저장. end node `/trunk/Gather` = 모델 출력 노드 (원래 FP32의 `/trunk/attn_pool/Gather`에 해당, 함수로 다시 구현해 이름만 다름) → 호스트로 빠지는 구간 없음 | 통과 |

- Mul +30 = K·V 마스크 곱 2 × (블록 5~11 7개 + 풀링 헤드 1) 16 + 잔차 마스크 곱 2 × 7블록 14. Add +16 = 상수 K_c·V_c 더하기 2 × 8. 개수가 설계와 맞다 (`ops_compare.json`).
- 파서가 "Simplified ONNX model for a parsing retry attempt" 후 성공했다. 원래 FP32 파싱(`../../G1_dfc_gate`, 10/13)도 같은 재시도를 거쳤다 (DFC 기본 동작).
- ORT vs PyTorch (export `--verify 64`): 배치 가변 0.9999998, b1 0.9999998.

## 파일 (대용량은 커밋하지 않음)
| 파일 | MiB | sha256 |
|---|---|---|
| `siglip2_image_fp32_rcmask55at5-13at9.onnx` | 363.62 | `b20381378256c54a…` |
| `siglip2_image_fp32_rcmask55at5-13at9_b1.onnx` | 363.59 | `38b3b521bed11a28…` |
| `siglip2_rcmask55at5_13at9.har` (`~/siglip2_work/hailo_g3r_mask/`) | 363.6 | `89905ac06db41f67…` |

- 크기가 R1(354.52)보다 9 MiB 큰 것은 상수 K_c·V_c를 블록마다 (1, 12, 196, 64) 모양(sink 위치 말고는 0)으로 넣었기 때문이다. 줄일 수 있지만 지금은 그대로 둔다.

## 실행
```bash
# 커밋 12cbe01 (s2_regcache.py MaskRegCacheTrunk, export --graph mask, maskcheck; selftest 마스크 = mean 7.5e-7)
RN50_RESULTS=../results/siglip2/G3R_regcache/g3a_mask "$SIGLIP2_PYTHON" s2_regcache.py maskcheck --n 64
RN50_RESULTS=../results/siglip2/G3R_regcache/g3a_mask "$SIGLIP2_PYTHON" s2_regcache.py export --graph mask --verify 64
# DFC (작업 폴더 ~/siglip2_work/hailo_g3r_mask, 저장소 밖 — hailo_sdk.client.log가 저장소에 생기지 않게)
source ~/hailo10_venv/bin/activate && export CUDA_VISIBLE_DEVICES=-1
hailo parser onnx siglip2_image_fp32_rcmask55at5-13at9_b1.onnx --hw-arch hailo10h --net-name siglip2_rcmask55at5_13at9 < /dev/null
```
- (b) 비교용 원래 FP32 b1은 이 PC에서 `s2_export_onnx.py`로 다시 만들었다 (`../env_labpc/export_fp32`). 배치 가변 파일 sha256이 Release와 같은 `333cad08…`(비트 단위 재현), b1은 `ce2b66b9…`.

## 다음 (용 결정 필요)
- 마스크 그래프는 R1과 FP32 동치이고 Hailo 파서를 통과한다. ORT INT8 정확도 재평가(마스크 그래프는 sink 위치 h=0·상수 K·V라 R1-INT8과 눈금이 다를 수 있음), G3b(16비트 층 줄이기), 컴파일은 하지 않았다.
