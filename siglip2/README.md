# SigLIP2 ViT-B/32 이미지 인코더 양자화 — 재현 방법

대회 안내 4.2절 "재현용 코드 및 실행 환경"과 예선 결과보고서 9장에 해당하는 문서다 (v4, 2026-10-23 — 제출 모델 B/32 O-AB, D-048·D-049).
실험 기록·결정은 `ROADMAP.md`, `DECISIONS.md`, `PROGRESS.md`, 결과는 루트 `results/siglip2/`에 있다.

> **제출 모델 = SigLIP2 B/32 O-AB** (D-048·D-049): 레지스터 캐시 M2 + 회전 + O-A(MatMul 출력 재반올림 생략) + O-B(SkipLayerNorm 융합), 이미지 인코더 MatMul 79개 전부 INT8. test는 이 파일로 1회 썼다 (`results/siglip2/G8_candidates/b32/test`).
> **본선 Hailo HEF**: 같은 PyTorch 모델(같은 회전 행렬·sink 규칙·M2 상수)에서 배치 1로 export한 FP32를 Hailo DFC로 컴파일한다. 최종 HEF(GF·ALL8F 등)는 팀이 정한다 (준용 G6d·G6f).
> **채점·재현은 배포한 ONNX 파일로 한다.** 다른 PC에서 다시 만들면 파일이 달라진다 (3절, 7절).
> **onnxruntime으로 실행한다.** 제출 모델에 `com.microsoft` 연산(SkipLayerNormalization)이 있다.

## 0. 무엇을 재현하나

| 모델 | 파일 | 크기 | sha256 |
|---|---|---|---|
| 베이스라인 FP32 | `siglip2_image_fp32.onnx` | 378.36 MB (360.83 MiB) | `be19a8ada01055df…` |
| **최적화 (제출)** | `siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx` | 102.76 MB (98.00 MiB) | `7a01044c6ba32bf4f5a12aba82eb19e06753f13c8bad43b9f6fcfcd9f484dbf6` |
| (비교용) M2rot INT8, 이 PC 빌드 | `siglip2_image_int8_rcmaskbcmlp15at6u5-7at10u9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx` | 102.85 MB | `9783d7defe8c0735…` |

- Release: 지금은 내부 확인용 pre-release `siglip2-b32-submit-20261023` (위 세 파일 + dev 6종 × 200장 묶음 + `portable_eval.py` + `SHA256SUMS.txt` + 이 PC 결과). 최종 제출 Release는 따로 만든다.
- 포맷: ONNX opset 17 + `com.microsoft` 1, 입력 `image` float32 N×3×256×256 (SigLIP2 전처리 후: 256×256, 평균·표준편차 0.5 정규화), 출력 `embedding` 768차원.
- 실행: onnxruntime 1.25.0 CPU, 기본 그래프 최적화(`ORT_ENABLE_ALL`, 정수 커널).
- 같은 파일이 i7-1165G7(Windows)과 Ultra 9 285K(WSL)에서 dev 고화질 평균 89.69 / 89.67%로, Cars에서만 순 약 1장 달랐다 (원인 미확정, 사진별 비교 진행 중).

## 1. 빠른 확인 (저장소 코드·데이터 없이, 내려받기 약 5분 + 실행 약 6분)

numpy와 onnxruntime만 있으면 된다. dev 6종 × 200장 묶음으로 FP32와 제출 모델을 비교한다.

```bash
python -m pip install numpy==2.4.4 onnxruntime==1.25.0
gh release download siglip2-b32-submit-20261023 --repo jungminmobile/oslab_mobile -D release
cd release && sha256sum -c SHA256SUMS.txt && cd ..
python release/portable_eval.py --data release --out out_quick --threads 4 \
  --models fp32=release/siglip2_image_fp32.onnx \
           oab=release/siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
```
- 기대 출력 (`out_quick/result.json`): 제출 모델 고화질 5종 cos **0.9893**, Δtop-1 **+0.4%p** (각 200장), CIFAR A 71.0 (FP32 71.5). 기준값 `release/result_i7-1165G7.json`.
- 확인 기록: 깨끗한 폴더에서 이 순서 그대로 실행해 사진 1,200장의 예측이 비트 단위로 같음 (`results/siglip2/G8_candidates/b32/xplat/repro_check`).
- `gh`가 없으면 GitHub Release 페이지에서 파일을 받아 `release/` 폴더 하나에 둔다.

## 2. 환경 (전체 재현)

- Python 3.11, x86-64 또는 ARM64 CPU. GPU 불필요. 디스크 약 20 GB (데이터·캐시·ONNX).

```bash
git clone https://github.com/jungminmobile/oslab_mobile.git && cd oslab_mobile
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r siglip2/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256        # 꼭 지정 (없으면 B/16 224로 동작)
export SIGLIP2_DATA=/path/to/data SIGLIP2_ARTIFACTS=/path/to/b32_artifacts HF_HOME=/path/to/hf_cache OPENBLAS_NUM_THREADS=1
mkdir -p $SIGLIP2_ARTIFACTS/onnx && cp release/*.onnx $SIGLIP2_ARTIFACTS/onnx/       # 1절에서 받은 모델
```
- Windows PowerShell: `$env:S2_MODEL_ID="hf-hub:timm/ViT-B-32-SigLIP2-256"` 처럼 쓴다.
- `RN50_*` 경로는 `s2common.py`가 위 값으로 자동 설정한다 (RN50 파일을 덮어쓰지 않게 막음, D-013). 산출물 폴더마다 `model_id.txt`로 B/16·B/32 캐시가 섞이지 않게 막는다.
- 처음 실행할 때 Hugging Face `timm/ViT-B-32-SigLIP2-256` 가중치를 받는다 (텍스트 분류기·FP32 기준용). 인터넷 없이 돌리려면 한 번 온라인으로 불러 캐시를 만든 뒤 `HF_HUB_OFFLINE=1`.

## 3. 모델 만들기 (레시피 — 재현은 배포 파일로)

1. FP32 ONNX export: `python s2_export_onnx.py --out <결과 폴더> --verify 1000`
2. 회전 행렬: `RN50_RESULTS=… python s2_rotate.py build` (rothad = 무작위 부호 Hadamard × Householder)
3. 레지스터 캐시 M2 + 회전, 통일 규칙 `15@6:5, 7@10:9` (보정 train 256장, 시드 0):
   `python s2_regcache.py cache --rot rothad --sinks 15@6:5,7@10:9` → `ucache` → `export --graph mask --mask-form bcast --mask-site mlp --rot rothad`
4. INT8 (M2rot): `python s2_quantize.py --src <M2rot FP32> --calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --tag all --dump-ops`
5. O-A·O-B(제출 모델): 준용 브랜치 `siglip2/o1-ort-speed`의 O1 절차 (`results/siglip2/O1/README.md`) — O-B 그래프 융합(SkipLayerNormalization) 뒤 4와 같은 옵션 + MatMul 출력 양자화 제외(`OpTypesToExcludeOutputQuantization=["MatMul"]`)

**같은 PC에서는 다시 만들어도 같은 파일이 나오지만, 다른 PC에서는 다른 파일이 나온다.** float64 행렬 곱 반올림이 CPU·OpenBLAS 커널에 따라 달라 회전 행렬부터 달라진다 (`results/siglip2/G8_candidates/repro_fingerprint`). 같은 레시피 B/32 M2rot이 두 PC에서 dev −0.79 / −0.87이었다. 그래서 **채점·재현은 배포 파일**로 한다.

## 4. 정확도 — dev (구성 비교용)

```bash
cd siglip2
python s2_build_dev.py all --threads 4        # dev 캐시·텍스트 분류기·FP32 기준 (데이터 자동 다운로드, 처음 1~2시간)
RN50_RESULTS=../results/siglip2/repro_b32_dev python s2_paired_ci.py --models siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
```
- 기대값: FP32 고화질 89.96%, 제출 모델 89.69% (Δ **−0.27 [−0.62, +0.09]**), cos 0.989, CIFAR A −0.5. 출력: `paired_ci.jsonl`, 사진별 예측 `pred_<모델>.npz`.
- `paired_ci.jsonl`은 이어 쓰기라 재현할 때는 새 결과 폴더를 지정한다.

## 5. 정확도 — test (최종 1회)

```bash
for s in pets caltech flowers cars food cifar_test imagenetv2; do
  RN50_RESULTS=../results/siglip2/repro_b32_test python s2_test_eval.py $s --threads 4 \
    --models siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
done
RN50_RESULTS=../results/siglip2/repro_b32_test python s2_test_summary.py
```
- test 7종: Oxford Pets·Flowers-102·Stanford Cars·Food-101·Caltech-101 공식 test 전체, CIFAR-100 test(1만), ImageNetV2 matched-frequency(1만, `scripts/test_g6.sh`가 받음).
- 기대값: 고화질 5종 평균 FP32 88.51% → 88.28%, Δ **−0.23 [−0.36, −0.08]%p**. CIFAR-100 +0.93, ImageNetV2 −0.20%p.
- 시간: 4코어 노트북에서 약 3시간 15분 (FP32와 INT8을 함께 잼, ImageNetV2 불러오기 약 30분 포함). 데이터셋마다 `done_<set>.json`이 있으면 다시 돌지 않는다.

## 6. 속도·메모리·크기

```bash
python scripts/cross_latency.py --out out_speed --mem \
  --models b32_fp32=$SIGLIP2_ARTIFACTS/onnx/siglip2_image_fp32.onnx \
           b32_oab=$SIGLIP2_ARTIFACTS/onnx/siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
```
- 배치 1, 워밍업 20, 200장 × 3회 모델 번갈아, 스레드 1·2·4 × 세션 3 → 중앙값. 전원 구성표·충전 상태가 결과에 기록된다.
- 기대값: FP32 대비 4스레드 약 −56~−57%, 1스레드 약 −60~−64% (절대 ms는 PC·실행마다 다름, 같은 PC에서도 2·4스레드 최대 약 40%), 피크 메모리 약 −57%, 크기 378.36 → 102.76 MB (−72.8%).
- **지연은 같은 실행 안의 FP32 대비 비율로 본다.** 노트북은 고성능 모드·충전 상태, 다른 프로그램 없이.
- ORT 세션 설정 `enable_mem_pattern=False`(정민 E12-1, 브랜치 `exp/E12-1-jjury`)를 쓰면 메모리가 약 9% 더 준다(출력 비트 동일). 제출 실행 스크립트에 넣을지는 팀이 정한다.

## 7. 실행 환경 기록

- 결과 jsonl 한 줄마다 커밋 해시, 커밋 안 된 변경 여부(`dirty`), CPU 모델·명령어 플래그(AVX2·AVX-512·VNNI), onnxruntime 버전을 남긴다 (`scripts/env_info.py`, 결과 폴더의 `env.json`).
- PC마다 빌드가 갈리는 지점을 찾는 지문: `scripts/repro_fingerprint.py` (버전·BLAS·회전 행렬 재계산·보정 사진·캐시·ONNX sha256).
- 그림: `scripts/make_report_figs.py`(F1·F2), `scripts/make_report_figs_b32.py`(F3b·F4b·F5) — raw에서 바로 그리고 쓴 숫자를 `docs/figs/figs_data*.json`에 남긴다.
