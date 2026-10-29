# G3Q 플랫폼 재현 확인: 회전 모델의 ORT 정확도, x86 ↔ ARM (2026-10-18)

**판정용이 아니다.** G3Q 채택 판정(상위 README 8절, 채택 실패)은 바뀌지 않는다. 같은 ONNX·같은 사진이 다른 CPU에서도 같은 경향을 보이는지 남기는 기록이다.

## 조건
| 이름 | CPU | 명령어 | Python / ORT / numpy |
|---|---|---|---|
| `x86_285K` (이 PC, WSL Ubuntu-22.04) | Intel Core Ultra 9 285K | AVX-VNNI, AVX2 (AVX-512 없음) | 3.11.17 / **1.25.0** / 2.4.4 |
| `orin` (`ssh agx`, `~/ssd/junyong/siglip2_xplat`) | Jetson AGX Orin CPU, Cortex-A78AE | asimddp | 3.10.12 / **1.23.2** / 1.26.4 |

- ORT 버전이 다르다 (AGX는 Python 3.10이라 1.24부터 휠 없음). 10/16 기록(`origin/xplat/siglip2-candidates`의 `results/xplat/2026-10-16_siglip2_candidates`)에서 285K의 ORT 1.23.2와 1.25.0은 값이 완전히 같았다.
- 스크립트: `siglip2/scripts/portable_eval.py` (sha256 `e88d1fe6…`, AGX 사본과 같음) `--data dev200 --threads 4`. 정수 커널(ORT_ENABLE_ALL), 배치 50.
- 사진: 10/15 묶음 `dev200/` (dev 6종 × 앞 200장, `SHA256SUMS` 6개 모두 OK, 양쪽 같은 파일). x86 쪽은 AGX에서 받아 `~/siglip2_work/xplat/`에 둠.
- 모델 (양쪽 sha256 확인, `SHA256SUMS_models_rot.txt`):

| 이름 | 파일 | sha256 |
|---|---|---|
| fp32 | Release `siglip2_image_fp32.onnx` | `333cad08…` |
| r0_410d | 이 PC의 R0 `…_u8_ops-MatMul_hc16_rr_idcheck_fc2L2.onnx` | `410d7483…` |
| r0_5eee | Release U8+rr fc2L2 (최종) | `5eee7839…` |
| rothad | `siglip2_image_int8_rothad_…_rr_fc2L2.onnx` | `0d11f89a…` |
| rotrand0 | `siglip2_image_int8_rotrand0_…_rr_fc2L2.onnx` | `c77bb160…` |

## 결과 (dev 6종 × 200장, top-1 %, cos = FP32 임베딩 대비 고화질 5종 평균)
| 플랫폼 | 모델 | CIFAR A | Pets | Flowers | Cars | Food | Caltech | 고화질 Δtop-1 (FP32 대비) | 고화질 cos | R0(410d) 대비 cos | R0 대비 top-1 [95%] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| x86 | fp32 | 74.5 | 93.5 | 85.0 | 93.5 | 87.5 | 95.0 | 기준 | 1 | | |
| x86 | r0_410d | 70.5 | 92.0 | 86.5 | 94.0 | 85.5 | 96.0 | −0.1 | 0.9547 | 기준 | |
| x86 | r0_5eee | 69.0 | 92.5 | 85.5 | 94.0 | 86.0 | 97.0 | +0.1 | 0.9545 | −0.0002 | +0.20 [−0.5, +1.0] |
| x86 | **rothad** | 68.5 | 93.5 | 86.0 | 94.0 | 86.0 | 96.5 | +0.3 | **0.9684** | **+0.0136** | +0.40 [−0.4, +1.3] |
| x86 | rotrand0 | 70.5 | 94.0 | 85.5 | 93.5 | 86.5 | 96.5 | +0.3 | 0.9633 | +0.0086 | +0.40 [−0.5, +1.4] |
| ARM | fp32 | 74.5 | 93.5 | 85.0 | 93.5 | 87.5 | 95.0 | 기준 | 1 | | |
| ARM | r0_410d | 66.5 | 92.5 | 85.5 | 94.5 | 86.0 | 96.5 | +0.1 | 0.9547 | 기준 | |
| ARM | r0_5eee | 70.0 | 92.0 | 86.0 | 94.5 | 86.0 | 96.5 | +0.1 | 0.9547 | +0.0001 | 0.00 [−0.6, +0.6] |
| ARM | **rothad** | 69.0 | 93.0 | 87.0 | 94.5 | 86.5 | 96.0 | +0.5 | **0.9681** | **+0.0134** | +0.40 [−0.3, +1.2] |
| ARM | rotrand0 | 69.5 | 92.5 | 85.5 | 94.5 | 86.0 | 97.0 | +0.2 | 0.9636 | +0.0088 | +0.10 [−0.7, +1.0] |

x86 ↔ ARM 같은 모델끼리 (고화질 5종, `xplat_compare.json`):
| 모델 | 예측 일치 | 사진별 cos 차이 평균 / 최대 | top-1 차 (ARM − x86) [95%] |
|---|---|---|---|
| fp32 | 100% | 4.8e-8 / 3.6e-7 | 0 |
| r0_410d | 97.4% | 0.0041 / 0.041 | +0.20 [−0.5, +0.9] |
| r0_5eee | 98.1% | 0.0040 / 0.022 | 0.00 [−0.6, +0.5] |
| rothad | **98.4%** | **0.0029** / 0.021 | +0.20 [−0.4, +0.8] |
| rotrand0 | 98.3% | 0.0033 / 0.020 | −0.10 [−0.9, +0.7] |

- 표의 cos·top-1은 `result.json`(데이터셋별 cos를 소수 4자리로 반올림한 뒤 평균). `xplat_compare.json`은 반올림 전 평균이라 ARM r0_5eee가 0.9548로 0.0001 다를 수 있다.
- 95% 구간: 데이터셋별 층화 짝 부트스트랩 1000회, 시드 0 (`s2_xplat_compare.py --vs r0_410d`). 200장이라 구간이 넓다.

## 재현 확인
- **10/16 기록과 비트 단위로 같다**: ARM r0_5eee 예측·cos가 10/16 `out_orin_a78ae/pred_u8rr.npz`와 사진마다 같다 (cos 최대 차이 0.0). ARM fp32, x86 r0_5eee(10/16 `out_labpc_ort1250`)도 같다. 고화질 cos 0.9547 / Δ +0.1 (ARM), 0.9545 / +0.1 (x86) 그대로.
- x86 값은 이 PC의 `s2_screen.py` 결과(상위 README 8절 Q2: R0 0.9547 / −0.1, rothad 0.9684 / +0.3, rotrand0 0.9633 / +0.3)와 같다 → 다른 스크립트·다른 데이터 경로(dev 캐시 vs dev200 묶음)로 같은 값.

## 해석
- **회전 효과는 플랫폼과 관계없이 같다**: R0 대비 고화질 cos +0.0136 (x86) / +0.0134 (ARM)로 거의 같고, rothad > rotrand0 > R0 순서도 같다.
- top-1 차이(R0 대비 +0.1~+0.4)는 200장 구간이 모두 0을 포함한다. dev 전체 판정(+0.01 [−0.45, +0.48], x86)과 모순되지 않는다.
- ARM ↔ x86 출력 차이는 회전 모델이 R0보다 약간 작다 (cos 차이 평균 0.0029 vs 0.0041, 예측 일치 98.4 vs 97.4%). 이번 결과로 원인은 판단하지 않는다.
- R0 두 파일(410d = 이 PC에서 만든 것, 5eee = Release)은 두 플랫폼 모두 cos 차이가 0.0002 이하다 (G3R identity의 "활성 눈금 미세 차이"와 맞음).

## 파일
- `out_labpc_ort1250_rot/`, `out_orin_a78ae_rot/`: `result.json`, `pred_<모델>.npz`
- `run_labpc_rot.log`, `run_orin_rot.log`: 실행 로그. `xplat_compare.json`: 짝 비교
- `SHA256SUMS_models_rot.txt`: AGX로 보낸 모델 3개 (AGX에서 `sha256sum -c` 모두 OK)

## 실행 명령
```bash
# AGX (nohup, 약 14분)
cd ~/ssd/junyong/siglip2_xplat && P=percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr
nohup ~/ssd/junyong/xplat/venv/bin/python -u portable_eval.py --data dev200 --out out_orin_a78ae_rot --threads 4 \
  --models fp32=models/siglip2_image_fp32.onnx r0_410d=models_rot/siglip2_image_int8_${P}_idcheck_fc2L2.onnx \
  r0_5eee=models/siglip2_image_int8_${P}_fc2L2.onnx rothad=models_rot/siglip2_image_int8_rothad_${P}_fc2L2.onnx \
  rotrand0=models_rot/siglip2_image_int8_rotrand0_${P}_fc2L2.onnx > run_orin_rot.log 2>&1 &
# 이 PC (약 5분 30초, 같은 모델 순서; 측정 중 G3R 세션의 hailo optimize가 같이 돌았음 — 정확도는 부하와 무관)
cd ~/siglip2_work/xplat && "$SIGLIP2_PYTHON" ~/oslab_rot/siglip2/scripts/portable_eval.py --data dev200 \
  --out <저장소>/results/siglip2/G3Q_rotate/xplat/out_labpc_ort1250_rot --threads 4 --models fp32=… r0_410d=… r0_5eee=… rothad=… rotrand0=…
# 짝 비교 (siglip2/)
RN50_RESULTS=../results/siglip2/G3Q_rotate/xplat "$SIGLIP2_PYTHON" s2_xplat_compare.py --runs x86_285K=… orin=… \
  --models r0_410d r0_5eee rothad rotrand0 --vs r0_410d
```
