# textenc — SigLIP2-B/16 텍스트 인코더 INT8 스모크 (담당: 준용)

팀 규칙(PRD R6: 텍스트 인코더는 FP32로 클래스 임베딩 1회 계산)은 **바꾸지 않는다.**
이 폴더는 "텍스트 인코더 INT8이 가능한가, 할 가치가 있는가"만 스모크로 확인한다. HEF 컴파일은 범위 밖.
`rn50/`, `siglip2/`, `PRD.md`, 루트 문서는 수정하지 않고 import만 한다. 결과는 `results/textenc/`.

## 구조 (open_clip 3.3.0 실제 모델에서 셈, `te_export.py`가 assert로 다시 확인)

| 구성 | 파라미터 | 비율 | FP32 | INT8 가중치 |
|---|---|---|---|---|
| 토큰 임베딩 표 (256,000 × 768) | 196.6M | 69.6% | 750.0 MiB | 187.5 MiB |
| 트랜스포머 블록 12개 | 85.1M | 30.1% | 324.5 MiB | 81.1 MiB |
| 헤드 (ln_final + Linear 768→768) | 0.59M | 0.2% | 2.3 MiB | 0.6 MiB |
| 합계 | 282.3M | | 1076.9 MiB | |

- context 64, 양방향 attention(causal mask 없음), **pad mask 없음**, 풀링은 **항상 위치 63** (`pool_type: last`)
- 크기는 임베딩 표에서, 연산 시간은 블록에서 나온다 → 변형 D(임베딩 표만)와 A·B·C(블록 MatMul·Gemm)를 나눠서 잰다
- 그래프에서 attention 출력 투영(out_proj)과 text_projection은 `Gemm`으로 나온다 → "MatMul만 INT8"은 MatMul·Gemm으로 둔다

## 변형

| 이름 | 내용 | 파일 |
|---|---|---|
| A | 동적 INT8, MatMul·Gemm 가중치 (ORT 기본 per-tensor) | `siglip2_text_int8_dyn.onnx` |
| B-u8 / B-s8 | 정적 QDQ, MatMul·Gemm만, 가중치 S8 채널별, 활성값 U8/S8 비대칭, Percentile 99.999(비대칭), 보정 문장 256 | `..._qdq_p99999_{u8,s8}.onnx` |
| C-u8 / C-s8 | B + 거대 활성값 층 FP32 (`protect.json`, 규칙: 활성값 입력 max/99.9% ≥ 20, 결과 보기 전 고정) | `..._protect.onnx` |
| D | 토큰 임베딩 표만 INT8 (행별 대칭, Gather(int8) → Cast → ×눈금, 고른 행만 펼침) | `..._embed8.onnx` |
| X+D | 위 블록 변형에 D를 덧붙인 조합 (따로 만들어 직접 잼) | `..._<X>_embed8.onnx` |

보정 문장: CLIP 공식 ImageNet 템플릿 80개 × ImageNet 클래스(Pets·Food 클래스 이름과 겹치는 것 제외), 시드 0에서 256개.
거대 활성값 점검은 그 앞 64개, 임베딩 확인용은 보정과 겹치지 않는 시드 1 문장 64개.

## 실행 (x86, 저장소 루트, SigLIP2 venv)

필요: `SIGLIP2_DATA`·`SIGLIP2_ARTIFACTS`, Pets·Food dev 캐시(`$SIGLIP2_ARTIFACTS/cache/{pets,food}_dev_{y,clf,ref_fp32}.npy`), Hugging Face 접속.
캐시가 없으면 먼저 (siglip2 쪽 스크립트, 결과는 artifacts 아래에만 쌓임):

```
"$SIGLIP2_PYTHON" siglip2/s2_export_onnx.py --out "$SIGLIP2_ARTIFACTS/results_tmp/export"
"$SIGLIP2_PYTHON" siglip2/s2_build_dev.py pets --threads 4
"$SIGLIP2_PYTHON" siglip2/s2_build_dev.py food --threads 4
```

스모크 전체 (약 1시간, 대부분 보정과 지연 측정):

```
nohup bash textenc/run_smoke.sh > textenc_smoke.log 2>&1 &
```

## ARM 지연 (Jetson AGX Orin, ORT 1.23.2 / Python 3.10)

onnxruntime·numpy·psutil만 있으면 된다(torch·transformers 불필요). x86에서 만든 ONNX와 `tokens_pets.npy`를 `~/ssd/junyong/textenc/`로 복사한 뒤:

```
cd ~/ssd/junyong/oslab_mobile && nohup python3 textenc/te_bench.py --dir ~/ssd/junyong/textenc --threads 1 2 4 --out results/textenc/smoke/speed_agx --models fp32=siglip2_text_fp32.onnx D=siglip2_text_int8_embed8.onnx C-u8+D=siglip2_text_int8_qdq_p99999_u8_protect_embed8.onnx C-s8+D=siglip2_text_int8_qdq_p99999_s8_protect_embed8.onnx > te_bench_agx.log 2>&1 &
```

1차 스모크 결과(x86)는 `results/textenc/smoke/README.md`. 위 4개가 그 결과의 후보다.

## 검증 방법 (코드)

- `TE_DRYRUN=1`이면 랜덤 가중치·가짜 토크나이저·가짜 dev로 전체 경로만 확인한다. 이 값은 결과로 쓰지 않는다.
- export: PyTorch vs ORT cos ≥ 0.9999 assert, 팀 캐시 분류기(PyTorch)와 ORT FP32 분류기 비교
- 보정: 회차마다 실제로 쓰인 문장 수 = 256 assert (실패하면 파일 삭제)
- 정확도는 ORT 기본 설정(정수 커널)만. 최적화된 그래프의 연산 개수(`QLinearMatMul` 등)를 quantize_log에 남긴다

## NPU(HEF) 시험 — 블록 12개만 (`te_export_blocks.py`, `hailo_parse_text.sh`)

Hailo Model Zoo처럼 토큰 임베딩(Gather)과 풀링·projection은 호스트에서, 트랜스포머 블록만 NPU에 둔다.
attention은 고정 shape 연산으로 다시 썼다. 결과 그래프에는 LayerNormalization·MatMul·Add·Slice·Reshape·Transpose·Mul·Softmax·Tanh(·Gemm)만 남는다
(Shape·Gather·Cast·ArgMax 없음). PyTorch 대비 cos ≥ 0.99999982.

1. ONNX 만들기 (SigLIP2 venv가 있는 x86):
   ```
   python textenc/te_export.py --out results/textenc/smoke && python textenc/te_export_blocks.py --out results/textenc/hailo_parse
   ```
   WSL Ubuntu 22.04에는 Python 3.10만 있고 팀 고정 버전(numpy 2.4 등)은 3.11 이상이 필요하다. 그래서 uv로 3.11 venv를 따로 만든다:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env && uv venv ~/textenc_venv -p 3.11 && uv pip install -p ~/textenc_venv -r siglip2/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
   ```
   ```
   export SIGLIP2_DATA=~/textenc_work/data SIGLIP2_ARTIFACTS=~/textenc_work/art HF_HOME=~/textenc_work/hf && ~/textenc_venv/bin/python textenc/te_export.py --out results/textenc/smoke && ~/textenc_venv/bin/python textenc/te_export_blocks.py --out results/textenc/hailo_parse
   ```
2. DFC (WSL, `~/hailo10_venv`):
   ```
   nohup bash textenc/hailo_parse_text.sh ~/textenc_work/art/textenc > hailo_text.log 2>&1 &
   ```
   결과는 `results/textenc/hailo_parse/<날짜>/steps.txt`(단계별 rc)와 로그.

### 2차: 양자화 구성별 정확도 (DFC 에뮬레이션, `textenc/hailo/`)

1차 결과(2026-10-16, 데스크탑 DFC 5.4.0, CPU·레벨 0): 블록 12개 parse·compile 성공, HEF 67 MB, 컨텍스트 11개 이상.
보정 입력은 NHWC (64,1,64,768)이어야 한다. HAR 층 이름 대응: 블록 0 c_fc = `conv3`, 블록 0 c_proj = `conv4`.

```
python textenc/te_dfc_evalset.py      # torch 환경: dfc_eval_{pets,food,holdout}.npz
nohup bash textenc/hailo/run_text_q.sh <HAR·보정·npz가 있는 폴더> > text_q.log 2>&1 &   # WSL hailo10_venv
```
구성: `t0_base`(전부 8비트) / `t1_cproj0`(블록 0 c_proj 16비트) / `t2_cproj0_ewadd`(+ 잔차 합 16비트).
