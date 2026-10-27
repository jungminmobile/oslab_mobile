# SigLIP2 ViT-B/32 이미지 인코더 양자화 — 재현 방법

대회 안내 4.2절 "재현용 코드 및 실행 환경"과 예선 결과보고서 9장에 해당하는 문서다 (v5, 2026-10-27 — 제출 모델 B/32 P2, D-048·D-052).
실험 기록·결정은 `ROADMAP.md`, `DECISIONS.md`, `PROGRESS.md`, 결과는 루트 `results/siglip2/`에 있다.

> **제출 모델 = SigLIP2 B/32 P2** (D-048·D-052): 레지스터 캐시 M2 + 회전 + O-A(MatMul 출력 재반올림 생략) + O-B(SkipLayerNorm 융합) + F2(GELU → FastGelu) + 패치 임베딩 정수화(정민 E14). 패치 임베딩을 포함한 행렬곱 80개 전부 INT8. 이전 제출 파일은 O-AB(`7a01044c…`, D-049)였다. B/32의 test는 ORT 2회(O-AB 10/22 `results/siglip2/G8_candidates/b32/test`, P2 10/27 `b32/test_p2`), 본선 HEF로 Hailo 1회(준용 `B32_hef_test`) 썼다. 교체는 test 전에 dev·지연·메모리로 확정했다(`b32/p2/README.md`). 개발 초기 B/16 단계 이력은 보고서 부록 E.
> **본선 Hailo HEF**: 같은 PyTorch 모델(같은 회전 행렬·sink 규칙·M2 상수)에서 배치 1로 export한 FP32를 Hailo DFC로 컴파일한다. 본선 HEF는 GF로 정했다(D-050). ALL8F는 절제 실험이다 (준용 G6d·G6f·B32_hef_test).
> **채점·재현은 배포한 ONNX 파일로 한다.** 다른 PC에서 다시 만들면 파일이 달라진다 (3절, 7절).
> **onnxruntime으로 실행한다.** 제출 모델에 `com.microsoft` 연산(SkipLayerNormalization 24, FastGelu 13)이 있다. 학술부 답(10/27): 채점은 팀이 낸 실행 스크립트로 한다.
> **제출 실행 설정 = `--no-mem-pattern`**(ORT `enable_mem_pattern=False`, E12-1 C3). 메모리만 줄고 출력은 기본 세션과 비트 동일하다(`results/siglip2/G8_candidates/b32/p2/c3_equal`, `xplat_i7_c3`).
> **본선 이식용 바탕**: 같은 모델의 표준 연산 FP32 `siglip2_image_fp32_g6b_b32_m2rot.onnx`(`23236123…`, Release에 있음). 제출 P2는 이 그래프를 CPU용으로 양자화·변환한 파일이고, 본선 HEF 바탕(`a0eb502b…`)과 initializer가 비트 단위로 같다.

## 0. 무엇을 재현하나

| 모델 | 파일 | 크기 | sha256 |
|---|---|---|---|
| 베이스라인 FP32 | `siglip2_image_fp32.onnx` | 378.36 MB (360.83 MiB) | `be19a8ada01055df…` |
| **최적화 (제출, P2)** | `e14_4_P2_patch_int8.onnx` | 95.65 MB (91.22 MiB) | `e465d10da92cf0f830b95d29d72cf3251ea34da90f54b6b5c1f3e16510efd6b2` |
| (이전 제출 파일) O-AB | `siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx` | 102.76 MB (98.00 MiB) | `7a01044c6ba32bf4f5a12aba82eb19e06753f13c8bad43b9f6fcfcd9f484dbf6` |
| 비상 대체: M2rot INT8, 표준 연산만 (채점 환경이 com.microsoft 연산을 못 돌릴 때) | `siglip2_image_int8_g6b_b32_m2rot_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx` | 102.85 MB | `3a46ae8af54f3119…` (제출 모델과 같은 바탕 `23236123…`, 본선 HEF와 이어짐) |
| (비교용) M2rot INT8, 이 PC 빌드 | `siglip2_image_int8_rcmaskbcmlp15at6u5-7at10u9_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx` | 102.85 MB | `9783d7defe8c0735…` (회전 행렬이 달라 본선 HEF와 이어지지 않음 — 대체 파일 아님) |

- Release: 지금은 내부 확인용 pre-release 두 개다. `siglip2-b32-submit-20261023`(FP32·O-AB·M2rot 두 개 + dev 6종 × 200장 묶음 + `portable_eval.py` + `SHA256SUMS*.txt` + 이 PC 결과 + 중간 파일 `23236123…` → `79e797c1…` → `506e9fe4…`), `siglip2-p2-20261027`(P2, F2 `03f4de18…`, SHA256SUMS). 최종 제출 Release는 하나로 묶어 따로 만든다.
- 제출 파일을 만드는 순서: O-AB `7a01044c…` → `e14/make_models_e14.py`(정민 브랜치 `jungmin/e14-graph`) → F2 `03f4de18…` → P2 `e465d10d…`. 세 PC에서 만든 파일의 sha가 같다.
- 포맷: ONNX opset 17 + `com.microsoft` 1 (SkipLayerNormalization·FastGelu), 입력 `image` float32 N×3×256×256 (SigLIP2 전처리 후: 256×256, 평균·표준편차 0.5 정규화), 출력 `embedding` 768차원.
- 실행: onnxruntime 1.25.0 CPU, 기본 그래프 최적화(`ORT_ENABLE_ALL`, 정수 커널).
- 다른 CPU: 같은 묶음(1,200장)에서 x86 두 대(i7-1165G7 AVX-512 VNNI, Ultra 9 285K AVX-VNNI)는 제출 P2 사진 1장만 달랐다. dev 전체 고화질 평균은 i7 89.91%, 정민 258V·준용 285K 89.87%(CIFAR A·Food·Caltech 1~2장). ARM(Jetson AGX Orin)은 1,200장 중 4~5장(0.4%) 달랐고 고화질 평균 Δ는 285K와 같았다(이전 O-AB는 25~26장). FP32는 세 대 모두 같았다 (`results/siglip2/G8_candidates/b32/p2/xplat_i7/compare_p2_3way.json`).
- 입력: 0~255 픽셀을 /127.5 − 1 한 값(open_clip 기본 전처리)일 때 P2의 첫 층 입력 양자화가 무손실이다.

## 1. 빠른 확인 (저장소 코드·데이터 없이, 내려받기 약 5분 + 실행 약 6분)

numpy와 onnxruntime만 있으면 된다. dev 6종 × 200장 묶음으로 FP32와 제출 모델을 비교한다. (최종 제출 Release로 하나로 묶기 전까지는 두 Release에서 받는다)

```bash
python -m pip install numpy==2.4.4 onnxruntime==1.25.0
gh release download siglip2-b32-submit-20261023 --repo jungminmobile/oslab_mobile -D release
gh release download siglip2-p2-20261027 --repo jungminmobile/oslab_mobile -p e14_4_P2_patch_int8.onnx -D release
cd release && sha256sum -c SHA256SUMS.txt && sha256sum e14_4_P2_patch_int8.onnx && cd ..   # P2: e465d10d…
python release/portable_eval.py --data release --out out_quick --threads 4 --no-mem-pattern \
  --models fp32=release/siglip2_image_fp32.onnx \
           p2=release/e14_4_P2_patch_int8.onnx
```
- 기대 출력 (`out_quick/result.json`): 제출 P2 고화질 5종 cos **0.9884**, Δtop-1 **+0.4%p** (각 200장, 285K·AGX는 +0.3), CIFAR A 73.0 (FP32 71.5). 이 PC 결과 `results/siglip2/G8_candidates/b32/p2/xplat_i7/result.json`.
- 확인 기록: 깨끗한 폴더에서 이 순서 그대로 실행해 사진 1,200장의 예측이 비트 단위로 같음 (이전 제출 파일 O-AB 기준, `results/siglip2/G8_candidates/b32/xplat/repro_check`).
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
5. O-A·O-B(이전 제출 파일 O-AB): 준용 브랜치 `siglip2/o1-ort-speed`의 O1 절차 (`results/siglip2/O1/README.md`) — O-B 그래프 융합(SkipLayerNormalization) 뒤 4와 같은 옵션 + MatMul 출력 양자화 제외(`OpTypesToExcludeOutputQuantization=["MatMul"]`)
6. F2·P2(제출 모델): 정민 브랜치 `jungmin/e14-graph`의 `e14/make_models_e14.py` (`E14_O_ONNX`=O-AB 경로, `E14_WORK`=출력 폴더) — O-AB에서 GELU 융합(F2) → 패치 임베딩 Conv를 자르기 + INT8 MatMul로(P2). 양자화를 다시 하지 않는 그래프 변환이라 같은 O-AB에서는 어느 PC에서나 같은 파일이 나온다(세 PC sha 일치)

**같은 PC에서는 다시 만들어도 같은 파일이 나오지만, 다른 PC에서는 다른 파일이 나온다.** float64 행렬 곱 반올림이 CPU·OpenBLAS 커널에 따라 달라 회전 행렬부터 달라진다 (`results/siglip2/G8_candidates/repro_fingerprint`). 같은 레시피 B/32 M2rot이 두 PC에서 dev −0.79 / −0.87이었다. 그래서 **채점·재현은 배포 파일**로 한다.

## 4. 정확도 — dev (구성 비교용)

```bash
cd siglip2
python s2_build_dev.py all --threads 4        # dev 캐시·텍스트 분류기·FP32 기준 (데이터 자동 다운로드, 처음 1~2시간)
RN50_RESULTS=../results/siglip2/repro_b32_dev python s2_paired_ci.py --models e14_4_P2_patch_int8.onnx
```
- 기대값: FP32 고화질 89.96%, 제출 P2 89.91% (Δ **−0.05 [−0.41, +0.33]**), cos 0.988, CIFAR A −0.2 (이 PC i7. 다른 x86에서는 89.87%). 출력: `paired_ci.jsonl`, 사진별 예측 `pred_<모델>.npz`.
- `paired_ci.jsonl`은 이어 쓰기라 재현할 때는 새 결과 폴더를 지정한다.

## 5. 정확도 — test (최종 1회)

```bash
for s in pets caltech flowers cars food cifar_test imagenetv2; do
  RN50_RESULTS=../results/siglip2/repro_b32_test python s2_test_eval.py $s --threads 4 \
    --models e14_4_P2_patch_int8.onnx
done
RN50_RESULTS=../results/siglip2/repro_b32_test python s2_test_summary.py
```
- test 7종: Oxford Pets·Flowers-102·Stanford Cars·Food-101·Caltech-101 공식 test 전체, CIFAR-100 test(1만), ImageNetV2 matched-frequency(1만, `scripts/test_g6.sh`가 받음).
- 기대값: 고화질 5종 평균 FP32 88.51% → 88.30%, Δ **−0.20 [−0.33, −0.07]%p**. CIFAR-100 +0.57, ImageNetV2 −0.43%p (`results/siglip2/G8_candidates/b32/test_p2`). (이전 제출 파일 O-AB: 88.28%, Δ −0.23 [−0.36, −0.08]%p, P2 − O-AB +0.02 [−0.10, +0.15])
- 시간: 4코어 노트북에서 약 3시간 15분 (FP32와 INT8을 함께 잼, ImageNetV2 불러오기 약 30분 포함). 데이터셋마다 `done_<set>.json`이 있으면 다시 돌지 않는다.

## 6. 속도·메모리·크기

```bash
python scripts/cross_latency.py --no-mem-pattern --out out_speed --mem \
  --models b32_fp32=$SIGLIP2_ARTIFACTS/onnx/siglip2_image_fp32.onnx \
           b32_p2=$SIGLIP2_ARTIFACTS/onnx/e14_4_P2_patch_int8.onnx
```
- 배치 1, 워밍업 20, 200장 × 3회 모델 번갈아, 스레드 1·2·4 × 세션 3 → 중앙값. 전원 구성표·충전 상태가 결과에 기록된다.
- 기대값: FP32 대비 4스레드 약 −60~−64%, 1스레드 약 −68% (절대 ms는 PC·실행마다 다름, 같은 PC에서도 2·4스레드 최대 약 40%), 피크 메모리 약 −63%, 크기 378.36 → 95.65 MB (−74.7%). 보고서 대표값은 10/27 저녁 C3 실행(4스레드 −64.4%, 1스레드 −68.3%, 메모리 −63.4%, `results/siglip2/G8_candidates/latency_this_pc/b32_p2_c3_20261027`), 같은 날 기본 세션 실행은 −59.8% / −67.6% / −62.6%(`b32_p2_20261027`).
- **지연은 같은 실행 안의 FP32 대비 비율로 본다.** 노트북은 고성능 모드·충전 상태, 다른 프로그램 없이.
- `--no-mem-pattern`(C3)의 메모리 효과는 측정 정의에 따라 다르다: 이 스크립트에서 P2 169.8 → 166.1 MiB(−2.2%), 정민 E12-1 스크립트에서 O-AB −4.0%(`results/siglip2/G8_candidates/b32/submit_prep/mem_pattern`). 지연은 바꾸지 않는다(E12-1).

## 7. 실행 환경 기록

- 결과 jsonl 한 줄마다 커밋 해시, 커밋 안 된 변경 여부(`dirty`), CPU 모델·명령어 플래그(AVX2·AVX-512·VNNI), onnxruntime 버전을 남긴다 (`scripts/env_info.py`, 결과 폴더의 `env.json`).
- PC마다 빌드가 갈리는 지점을 찾는 지문: `scripts/repro_fingerprint.py` (버전·BLAS·회전 행렬 재계산·보정 사진·캐시·ONNX sha256).
- 그림: `scripts/make_report_figs.py`(F1·F2), `scripts/make_report_figs_b32.py`(F3b·F4b·F5) — raw에서 바로 그리고 쓴 숫자를 `docs/figs/figs_data*.json`에 남긴다.
