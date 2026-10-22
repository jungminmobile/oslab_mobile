# B/32 제출 모델 portable_eval — 준용 x86(285K)·ARM(AGX) 결과와 준혁 i7 비교 (2026-10-23)

- 모델은 새로 만들지 않았다. test는 쓰지 않았다 (dev 6종 × 앞 200장, Release 묶음 그대로).
- Release `siglip2-b32-submit-20261023`을 빈 새 폴더에 받아 `sha256sum -c SHA256SUMS.txt` 11개가 모두 OK인 것을 확인했다(285K 폴더 `~/siglip2_work/b32_xplat_20261023`, AGX는 그 파일을 복사한 `~/ssd/junyong/b32_submit_20261023/rel`에서 다시 확인).
- 실행은 Release 설명의 명령 그대로 했다(`--data . --threads 4`, 모델 3개). AGX는 `nohup timeout 5400`로 실행했고 tmux는 쓰지 않았다. 실행 직전에 `/dev/hailo*`를 연 프로세스가 0개였고 1분 부하는 0.00이었다.
- 결과: `out_junyong_285k/`, `out_junyong_agx/` (result.json, pred_*.npz, run.log). 비교: `compare_junyong.py` → `compare_junyong.json`.

## 1. 환경
| | 준혁 (기준) | 준용 x86 | 준용 ARM |
|---|---|---|---|
| CPU | Intel i7-1165G7 | Intel Core Ultra 9 285K (24코어) | Cortex-A78AE ×12 (Jetson AGX Orin, L4T R36.4.7) |
| 정수 명령 | AVX-512 VNNI | **avx_vnni** 있음, avx512_vnni 없음 | **asimddp(dotprod)** 있음, i8mm 표시 없음 |
| OS | Windows 10.0.26300 | WSL2 Ubuntu 22.04.5 (커널 6.6.114.1) | Ubuntu 22.04.4 (커널 5.15.148-tegra) |
| Python | 3.11.9 | 3.11.17 (venv `~/venvs/siglip2`) | 3.11.17 (uv로 SSD에 새로 설치: `~/ssd/junyong/b32_submit_20261023/venv_ort125`) |
| ORT | 1.25.0 | 1.25.0 | 1.25.0 |
| numpy | 2.4.4 | 2.4.4 | **2.4.6** (`pip install numpy` 최신) |
| 걸린 시간 fp32 / oab / m2rot (초) | 197 / 123 / 219 | 30 / 14 / 17 | 118 / 47 / 50 |

- AGX의 기존 venv는 Python 3.10이라 ORT 1.25.0을 설치할 수 없었다(PyPI의 aarch64 cp310은 1.23.2까지). 그래서 uv로 Python 3.11 venv를 새로 만들었다. 시스템 Python과 기존 venv는 건드리지 않았다.

## 2. 데이터셋별 top-1 (200장씩, FP32 top-1은 세 환경 모두 같음: 71.5 / 91.5 / 84.0 / 91.0 / 84.5 / 93.5)
| 모델 | 환경 | CIFAR A | Pets | Flowers | Cars | Food | Caltech | 고화질 Δ | 고화질 cos |
|---|---|---|---|---|---|---|---|---|---|
| O-AB | 준혁 i7 | 71.0 | 91.0 | 84.0 | 92.5 | 84.5 | 94.5 | +0.4 | 0.9893 |
| O-AB | 285K | 71.0 | 91.0 | 84.0 | 92.5 | 84.5 | 94.5 | +0.4 | 0.9893 |
| O-AB | AGX | **72.5** | **91.5** | 84.0 | 92.5 | **85.0** | **93.5** | +0.4 | 0.9894 |
| M2rot | 준혁 i7 | 73.0 | 91.0 | 84.0 | 90.0 | 83.5 | 95.5 | −0.1 | 0.9772 |
| M2rot | 285K | 73.0 | 91.0 | 84.0 | 90.0 | 83.5 | 95.5 | −0.1 | 0.9772 |
| M2rot | AGX | 73.0 | **91.5** | **83.5** | 90.0 | 83.5 | **94.5** | −0.3 | 0.9770 |

- 285K는 두 INT8 모델 모두 데이터셋별 top-1이 준혁 i7과 **같다**. 다만 비트 단위로 같지는 않다: O-AB의 데이터셋별 cos 평균이 CIFAR A 0.9606 vs 0.9605, Cars 0.9876 vs 0.9875로 소수 넷째 자리가 다르다 (VNNI 종류가 다름: avx_vnni vs AVX-512 VNNI).
- AGX는 O-AB에서 데이터셋별로 최대 1.5%p(CIFAR A 3장) 다르지만, 고화질 평균 Δ는 +0.4로 같다.
- (참고) Release 설명의 "i7과 285K에서 0.1~0.2%p 차이"는 dev 전체 기준이다. 이 200장 묶음에서는 차이가 나지 않았다.

## 3. 사진별 차이
준혁 i7은 `result.json`만 공개되어 있고 `pred_*.npz`는 Release와 `siglip2/g8-candidates` 어디에도 없다. 그래서 **사진별 비교는 285K ↔ AGX만** 했고, i7과는 데이터셋별 정답 수 차이로 정한 하한만 적는다.

### 3-1. 285K ↔ AGX (1,200장)
| 모델 | top-1이 다른 사진 | 그중 맞음/틀림이 바뀐 사진 | 사진별 \|cos 차이\| 최대 | 데이터셋별 다른 사진 (CIFAR/Pets/Flowers/Cars/Food/Caltech) |
|---|---|---|---|---|
| FP32 | **0** | 0 | 2.4e-7 | 0/0/0/0/0/0 |
| O-AB | **26 (2.2%)** | 17 | 0.026 | 12/2/1/6/3/2 |
| M2rot | **47 (3.9%)** | 26 | 0.042 | 14/5/7/9/8/4 |

(cos는 각 환경 임베딩과 npz 안 FP32 기준 임베딩(ref) 사이의 cos다. portable_eval은 임베딩 자체를 저장하지 않으므로, 두 환경 임베딩끼리의 cos는 이 결과로 잴 수 없다.)

### 3-2. FP32 기준 임베딩(ref) 대비 사진별 cos — 최소 / 평균
| 모델 | 285K 전체 1,200장 | AGX 전체 1,200장 | 285K 고화질 1,000장 | AGX 고화질 1,000장 |
|---|---|---|---|---|
| FP32 | 0.9999998 / 1.0000000 | 0.9999998 / 1.0000000 | 0.9999998 / 1.0000000 | 0.9999998 / 1.0000000 |
| O-AB | 0.8850 / 0.9845 | 0.8802 / 0.9846 | 0.9217 / 0.9893 | 0.9123 / 0.9894 |
| M2rot | 0.7550 / 0.9662 | 0.7738 / 0.9660 | 0.9011 / 0.9772 | 0.8837 / 0.9771 |

### 3-3. 준혁 i7과의 차이 (하한: 데이터셋별 정답 수 차이의 합)
| 모델 | 285K vs i7 | AGX vs i7 |
|---|---|---|
| FP32 | 0장 이상 | 0장 이상 |
| O-AB | 0장 이상 | 7장 이상 (CIFAR 3, Pets 1, Food 1, Caltech 2) |
| M2rot | 0장 이상 | 4장 이상 (Pets 1, Flowers 1, Caltech 2) |
- 정확한 사진 수를 알려면 준혁 i7의 `out_i7/pred_*.npz`가 필요하다. 받으면 `compare_junyong.py`에 i7을 넣어 다시 계산할 수 있다.

## 4. 결론
- **FP32는 x86·ARM에서 사진별 top-1이 모두 같다** (cos 차이 2.4e-7, 부동소수점 오차 수준).
- **INT8은 CPU 명령 계열에 따라 사진별 결과가 바뀐다.** 같은 x86 VNNI 계열(285K avx_vnni ↔ i7 AVX-512 VNNI)은 데이터셋별 top-1이 모두 같았지만 비트 단위로는 다르다. ARM(dotprod)은 O-AB 1,200장 중 26장(2.2%)이 x86과 다른 클래스를 냈고, 맞음/틀림이 바뀐 것은 17장이다. 고화질 평균 Δ는 +0.4로 같다.
- 제출 모델 O-AB는 M2rot보다 환경 간 차이가 작다 (26장 vs 47장, 최대 |cos 차이| 0.026 vs 0.042). O-AB는 QKᵀ·Softmax를 float으로 두므로 정수 반올림 차이가 생기는 곳이 적은 것으로 보이지만, 확인하지는 않았다.
- 재현 문서에는 "같은 ONNX라도 x86과 ARM에서 사진 약 2%의 top-1이 달라질 수 있고, 200장 데이터셋에서 데이터셋별 top-1이 최대 1.5%p 다를 수 있다(고화질 평균은 같음)"로 적을 것을 제안한다.

## 5. 같은 날 Release에 추가한 파일 (기존 asset은 그대로)
| 파일 | sha256 앞 8자리 | 내용 |
|---|---|---|
| `siglip2_image_fp32_g6b_b32_m2rot.onnx` | `23236123` | B/32 M2rot FP32 (G6b `s2_g6.py export --cfg m2rot --dynamic`, 커밋 `e7cd97c`) |
| `siglip2_image_fp32_g6b_b32_m2rot_o1b_pre.onnx` | `506e9fe4` | O-AB가 실제로 쓴 `_pre` (`s2_quantize.py --src …_o1b.onnx`의 `quant_pre_process` 출력, 커밋 `3972f43`) |
| `SHA256SUMS_v2.txt` | — | 기존 `SHA256SUMS.txt` 11줄 + 위 2줄 |
- 순서: `23236123` →(`s2_ortfuse.py --skip-ln --fold-shape`, `3972f43`) `_o1b.onnx` `79e797c1`(올리지 않음) →(`quant_pre_process`) `_o1b_pre` `506e9fe4` →(`quantize_static`, `--no-matmul-out-q` 등) O-AB `7a01044c`.
- `_pre`는 O1의 O-B 양자화(02:50) 때 만들어졌고 O-AB(02:51)가 재사용했다 (src보다 새 파일이면 재사용하는 코드, quantize 로그의 `src`와 파일 시각으로 확인). 10/23에 `_o1b.onnx`로 `quant_pre_process`를 다시 돌리니 sha256이 같았다.
- 올린 뒤 빈 새 폴더에 다시 내려받아 sha256 2개가 모두 OK인 것을 확인했고, Release 설명 끝에 만드는 순서를 덧붙였다(원래 설명은 바꾸지 않음).

## 실행 명령
```bash
# 285K (WSL)
mkdir ~/siglip2_work/b32_xplat_20261023 && cd $_
gh release download siglip2-b32-submit-20261023 -R jungminmobile/oslab_mobile && sha256sum -c SHA256SUMS.txt
~/venvs/siglip2/bin/python portable_eval.py --data . --out out_junyong_285k --threads 4 --models fp32=… oab=… m2rot=…   # Release 설명과 같은 파일 3개
# AGX (위 파일을 scp → sha256sum -c 다시 확인)
nohup timeout 5400 venv_ort125/bin/python portable_eval.py --data . --out out_junyong_agx --threads 4 --models … > run_agx.log 2>&1 &
# 비교
python compare_junyong.py
```
- portable_eval은 Release의 스크립트라 결과 줄에 git 커밋을 남기지 않는다. 비교 스크립트는 이 브랜치의 커밋에 들어 있다.
