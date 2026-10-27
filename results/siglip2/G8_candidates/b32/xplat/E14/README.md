# E14 P2 ARM 확인 — 285K(x86) ↔ AGX(ARM) portable_eval (준용, 2026-10-27)

- 목적: E14 P2(예선 제출 후보, 패치 임베딩 Conv → 자르기 + INT8 MatMul)가 ARM에서도 O-AB와 비슷하게 동작하는지 확인한다.
- 모델은 새로 양자화하지 않았다. test는 쓰지 않았다 (10/23 묶음의 dev 6종 × 앞 200장).
- 비교 방식은 10/23 `../compare_junyong.py`와 같다 (`compare_e14.py` → `compare_e14.json`).

## 판정 기준 (실행 전에 정함)
- 통과: AGX P2의 고화질 Δ가 285K P2와 ±0.5%p 이내이고, 285K↔AGX top-1이 다른 사진 수가 O-AB(같은 실행)의 2배 이하.
- 그렇지 않으면 "ARM 주의"로 보고한다. 제출 결정은 팀이 한다.

## 모델과 sha256
| 이름 | 파일 | sha256 앞 8자리 | 285K | AGX |
|---|---|---|---|---|
| fp32 | `siglip2_image_fp32.onnx` (10/23 묶음) | `be19a8ad` | OK | OK |
| oab | `siglip2_image_int8_g6b_b32_m2rot_o1b_…_moq0_all.onnx` (10/23 묶음) | `7a01044c` | OK | OK |
| f2 | `models_e14/e14_2_F2_fastgelu_bias.onnx` | `03f4de18` | OK | OK |
| p2 | `models_e14/e14_4_P2_patch_int8.onnx` | `e465d10d` | OK | OK |
- F2·P2는 오늘 Windows 재현(`C:\e14data\e14\`)에서 만든 파일을 복사했다. 묶음의 기존 11개 파일은 양쪽 모두 `sha256sum -c SHA256SUMS.txt` 11개 OK (기존 파일은 건드리지 않음). 기록: `SHA256_285k.txt`, `env_agx.txt`.

## 1. 환경
| | 285K (x86) | AGX (ARM) |
|---|---|---|
| CPU | Intel Core Ultra 9 285K (24코어) | Cortex-A78AE ×12 (Jetson AGX Orin, L4T R36.4.7) |
| 정수 명령 | **avx_vnni** 있음 (avx512_vnni 없음) | **asimddp(dotprod)** 있음, i8mm 표시 없음 |
| OS | WSL2 Ubuntu 22.04.5 (커널 6.6.114.1) | Ubuntu 22.04.4 (커널 5.15.148-tegra) |
| Python / ORT / numpy | 3.11.17 / 1.25.0 / 2.4.4 (`~/venvs/siglip2`) | 3.11.17 / 1.25.0 / 2.4.4 (`venv_ort125_np244`, 10/23과 같음) |
| 실행 직전 상태 | 1분 부하 0.06 | 1분 부하 0.14, `/dev/hailo*` 장치 파일 없음 → 연 프로세스 0개, Hailo 미사용 |
| 걸린 시간 fp32 / oab / f2 / p2 (초) | 31 / 15 / 11 / 11 | 117 / 47 / 43 / 40 |

## 2. 데이터셋별 top-1 (200장씩)
| 모델 | 환경 | CIFAR A | Pets | Flowers | Cars | Food | Caltech | 고화질 Δ | 고화질 cos |
|---|---|---|---|---|---|---|---|---|---|
| FP32 | 285K | 71.5 | 91.5 | 84.0 | 91.0 | 84.5 | 93.5 | 0.0 | 1.0000 |
| FP32 | AGX | 71.5 | 91.5 | 84.0 | 91.0 | 84.5 | 93.5 | 0.0 | 1.0000 |
| O-AB | 285K | 71.0 | 91.0 | 84.0 | 92.5 | 84.5 | 94.5 | +0.4 | 0.9893 |
| O-AB | AGX | 72.5 | 91.5 | 84.0 | 92.5 | 85.0 | 93.5 | +0.4 | 0.9894 |
| F2 | 285K | 71.0 | 91.5 | 84.0 | 92.5 | 84.5 | 94.5 | +0.5 | 0.9893 |
| F2 | AGX | 73.0 | 91.5 | 84.0 | 92.0 | 85.0 | 93.5 | +0.3 | 0.9894 |
| **P2** | 285K | 73.0 | 90.5 | 84.0 | 92.5 | 85.0 | 94.0 | **+0.3** | 0.9884 |
| **P2** | AGX | 73.0 | 90.5 | 84.0 | 92.5 | 84.5 | 94.5 | **+0.3** | 0.9884 |
- 고화질 Δ = 고화질 5종(Pets·Flowers·Cars·Food·Caltech) 평균 top-1의 FP32 대비 차이(%p, 1,000장). 200장 묶음이라 0.1%p = 사진 1장 수준이다.

## 3. 285K ↔ AGX 사진별 (1,200장)
| 모델 | top-1이 다른 사진 | 그중 맞음/틀림이 바뀐 사진 | 사진별 \|cos 차이\| 최대 | 데이터셋별 다른 사진 (CIFAR/Pets/Flowers/Cars/Food/Caltech) |
|---|---|---|---|---|
| FP32 | 0 | 0 | 2.4e-7 | 0/0/0/0/0/0 |
| O-AB | 26 (2.2%) | 17 | 0.026 | 12/2/1/6/3/2 |
| F2 | 23 (1.9%) | 14 | 0.026 | 11/0/1/5/4/2 |
| **P2** | **5 (0.4%)** | **4** | **0.015** | 0/0/1/2/1/1 |
(cos는 각 환경 임베딩과 npz 안 FP32 기준 임베딩 사이의 cos다. 두 환경 임베딩끼리의 cos는 portable_eval 출력으로 잴 수 없다.)

FP32 기준 사진별 cos 최소 / 평균
| 모델 | 285K 전체 | AGX 전체 | 285K 고화질 | AGX 고화질 |
|---|---|---|---|---|
| O-AB | 0.8850 / 0.9845 | 0.8802 / 0.9846 | 0.9217 / 0.9893 | 0.9123 / 0.9894 |
| F2 | 0.8833 / 0.9846 | 0.8802 / 0.9847 | 0.9200 / 0.9893 | 0.9136 / 0.9894 |
| P2 | 0.8678 / 0.9829 | 0.8678 / 0.9829 | 0.9172 / 0.9884 | 0.9172 / 0.9884 |

## 4. 10/23 재현 확인
- O-AB와 FP32의 `pred_*.npz`(사진별 예측·cos·정답 모든 배열)가 10/23 결과와 **비트 단위로 같다**: 285K는 `../out_junyong_285k/`, AGX는 같은 venv로 돌린 `../out_junyong_agx_np244/`와 비교. → 같은 환경이 그대로 재현됐다.

## 5. 판정: **통과**
- AGX P2 고화질 Δ +0.3 = 285K P2 +0.3 (차이 0.0%p ≤ 0.5).
- 285K↔AGX top-1이 다른 사진: P2 5장 ≤ O-AB 26장 × 2 = 52장 (O-AB의 0.19배).

## 6. 해석
- P2는 x86과 ARM 사이 차이가 O-AB보다 **오히려 작다** (다른 사진 26 → 5장, 최대 |cos 차이| 0.026 → 0.015). F2는 O-AB와 비슷하다(23장).
- 따라서 O-AB의 x86/ARM 차이의 대부분은 F2와 P2가 다른 곳, 즉 **패치 임베딩 Conv 부분**에서 생기는 것으로 보인다. 원래 Conv 경로의 정수 커널이 x86과 ARM에서 다르게 계산되고, P2의 INT8 MatMul 경로는 양쪽 결과가 더 가깝다는 가설이다. **확인하지는 않았다** (층별 출력 비교 안 함).
- P2의 고화질 Δ(+0.3)가 O-AB(+0.4)보다 1장 낮은 것은 200장 묶음의 잡음 수준이다. 정확도 비교는 정민 E14-4(dev 전체, CI상 차이 없음)를 따른다.

## 실행 명령
```bash
# 285K (WSL) — 10/23 묶음 폴더
cd ~/siglip2_work/b32_xplat_20261023 && mkdir models_e14 && cp /mnt/c/e14data/e14/{e14_2_F2_fastgelu_bias,e14_4_P2_patch_int8}.onnx models_e14/
O=siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
~/venvs/siglip2/bin/python -I portable_eval.py --data . --out out_junyong_285k_e14 --threads 4 \
  --models fp32=siglip2_image_fp32.onnx oab=$O f2=models_e14/e14_2_F2_fastgelu_bias.onnx p2=models_e14/e14_4_P2_patch_int8.onnx
# AGX (~/ssd/junyong/b32_submit_20261023/rel, models_e14/는 scp 후 sha256 확인)
nohup timeout 5400 ../venv_ort125_np244/bin/python -I portable_eval.py --data . --out out_junyong_agx_e14 --threads 4 --models (위와 같음) > run_agx_e14.log 2>&1 &
# 비교
~/venvs/siglip2/bin/python compare_e14.py
```
- portable_eval은 Release 스크립트라 결과에 git 커밋을 남기지 않는다. 비교 스크립트와 결과는 이 브랜치 커밋에 들어 있다.
