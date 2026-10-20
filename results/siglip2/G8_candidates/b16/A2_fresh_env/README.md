# G8 A2: 새 가상환경에서 재현 (보고서 9장) — 2026-10-20

목적: 학술부가 README대로 새 환경을 만들어 실행했을 때 같은 결과가 나오는지 확인.
- 실행: `bash siglip2/scripts/g8_b16_a2.sh` (커밋 a286243), 새 venv `D:\mokerton\venvs\siglip2_repro_g8` (Python 3.11.9, `requirements.txt`만 설치, `pip_freeze.txt`)
- 고정한 핵심 패키지(torch 2.14.0, onnxruntime 1.25.0, onnx 1.23.1, open_clip 3.3.0, timm 1.0.30, numpy 2.4.4 등)는 기존 venv와 같다. 고정하지 않은 하위 패키지 5개만 다름 (`pip_freeze_diff_vs_main_venv.txt`: filelock, httpcore2/httpx2, portalocker, truststore, 기존 venv에만 onnx-ir)
- README 3절 순서: FP32 export → 회전 행렬 → M2 캐시·u_c → M2rot export → INT8 (이 PC i7-1165G7)

## sha256 비교 (`sha256_compare.txt`)

| 파일 | 새 venv vs 같은 PC 기존 venv(A1) | 새 venv vs 준용 빌드·배포 |
|---|---|---|
| FP32 ONNX | 같음 | **같음** (`333cad08…`) |
| 회전 행렬 `rotq_rothad.npy` | 같음 | 다름 |
| M2 캐시 npz 2개 | 다름 (아래) | 다름 |
| M2rot FP32 ONNX | **같음** | 다름 |
| M2rot_ort INT8 | **같음** (`009ea9d5…`) | 다름 (`e84c3727…`) |

- 캐시 npz는 안에 기록한 커밋 해시(meta)만 달라 sha256이 다르다. 그 캐시로 만든 ONNX가 비트 단위로 같으므로 숫자 배열은 같다.
- **같은 PC에서는 환경을 새로 만들어도 비트 단위로 같은 파일이 나온다.** 다른 PC(준용 285K, AVX-VNNI)와는 회전 행렬부터 다르다 → 패키지는 원인이 아니다. 남는 차이는 PC(CPU·BLAS)이고, 행렬 곱 반올림 차이로 **추정**한다 (두 회전 행렬 값을 직접 비교하지는 않음 — 가설). 회전 행렬 직교성 오차 1.8e-15로 정상.

## 정확도 (dev 6×200, 새 venv의 ORT, 같은 실행)

| 모델 | 고화질 cos | Δtop-1 (200장) | 크기 MiB |
|---|---|---|---|
| 이 PC 새 빌드 | 0.9755 | −0.1 | 91.73 |
| 배포 파일 M2rot_ort | 0.9756 | +0.3 | 91.72 |

- dev 전체로는 이 PC 빌드 −0.31 vs 배포 −0.23 (A1). 차이 0.08%p.

## 보고서에 쓸 것 (9.4 재현 시 유의사항)
- 채점·재현은 **배포한 ONNX 파일**로 한다 (`gh release download siglip2-m2rot-ort-20261019`). 이 파일은 x86 세 CPU에서 dev 결과가 같다 (G6_m2rot_ort).
- 모델을 처음부터 다시 만드는 명령은 같은 레시피를 보여 준다. 같은 PC에서는 비트 단위로 같은 파일이 나오지만, CPU가 다르면 회전 행렬의 마지막 자리부터 달라져 파일이 달라지고 정확도가 0.1%p 안팎 다를 수 있다.

## 3중 확인
1. 자기 검증: 각 단계 rc=0, QLinearMatMul 79, 보정 장수 assert, export cos 확인 통과.
2. 독립 재계산: sha256을 hashlib로 두 번(A1·A2 스크립트 각각) 계산해 같은 값.
3. 독립 검토 (10/20, 맥락 없는 서브에이전트, 읽기 전용): raw에서 재계산해 일치, 지적 사항 반영 (커밋 97ec79e)
