# G3R 판별 실험: sink 토큰을 레지스터로 빼내 SigLIP2를 전부 8비트로 (설계, 2026-10-18)

> 상태: **G0~G2·G3a 완료 (10/18, 연구실 데스크탑). G2 채택 실패 확정(① 정확도 비열등 미달), G3a 1차 실패(DFC가 토큰 축 Concat 미지원) → 제자리 마스크 그래프(D-041)로 재시도해 **DFC 파싱 통과**, 상수 broadcast(`g3a_maskbc/`)로 크기 R1과 같음, ORT INT8은 R1-INT8과 동률. **G3b 실패** (전부 8비트 붕괴, 16비트 줄인 구성 cos 0.73~0.78 < 0.952; 원인 가설: sink 잔차 0 → Hailo LN 분해의 1/√ε 이상치, `g3b/`). 컴파일 안 함.** 결과는 8·9절. G3b는 시작하지 않음. 판정 기준(5절)은 결과를 보기 전에 고정했고 바꾸지 않았다.
> 브랜치: `siglip2/g3r-regcache` (`siglip2/g3-sensitivity`에서 분기). 담당: 용.

## 1. 한 줄 질문

블록 5·9에서 거대 값을 갖는 고정 위치 토큰(패치 55, 패치 13)을 **지우고**, 그 토큰의 K·V를 **상수로 캐시해 attention에 붙이면**,
FP32 정확도는 그대로이면서 fc2 예외 없이 **MatMul 전부를 INT8**(ORT 정수 커널)로 만들 수 있는가?
그리고 그 그래프가 Hailo DFC를 통과해 16비트 층을 줄일 수 있는가?

## 2. 왜 이 실험인가 (근거)

| 근거 | 내용 | 출처 |
|---|---|---|
| 원인 | 블록 5 MLP가 패치 55에, 블록 9 MLP가 패치 13에 거대 값(채널 67·301·145·483, 최대 수백)을 써 넣는다. 위치는 dev 6종에서 100% 고정 | `G3_why` 5·6-2절 |
| 기전 | 망가지는 것은 거대 값 토큰이 아니라 **같은 텐서를 쓰는 보통 토큰**이다 (high-norm 토큰만 8비트 0.9999 / 보통 토큰만 0.814) | `G3_why` 6-2절 |
| 처방의 상한 (가짜 양자화) | high-norm 토큰만 FP32로 빼고 텐서 눈금 → fc2 5·9: cos 0.995, **fc2 12개 전부: 0.988** (지금 텐서 하나: 0.707) | `G3_why` 6-3절 (`why_fix`) |
| 기능 | 거대 값을 그냥 지우면 cos 0.50, sink로 가는 attention을 막으면 0.81 → **지우기만 하면 안 되고 대체물이 필요** | `G3_why` 6-2절 B |
| 선행 연구 | RegCache: 민감 층 입력에서 high-norm 토큰을 지우고, 그 토큰들의 KV 평균을 레지스터로 앞에 붙인다. 학습 불필요, SigLIP2-B/16 포함 실험 | arXiv 2510.04547 |
| 선행 연구 | 학습 없이 register 역할 뉴런의 활성값을 추가 토큰 하나로 옮기면 패치의 high-norm 이상치가 사라진다 | arXiv 2506.08010 (NeurIPS 2025) |

**RegCache와 다른 점 (우리 기여)**
1. **정적 인덱스**: RegCache는 실행 중 ℓ∞ 상위 토큰을 골라(TopK) 지운다. 우리는 위치가 고정임을 측정했으므로 상수 인덱스로 지운다 → TopK·Gather 없이 Slice·Concat만 쓴다 (NPU 친화).
2. **위치별 캐시**: 여러 토큰의 KV를 하나로 평균하지 않고, sink 토큰마다 **자기 K·V**를 층마다 캐시한다. 그래서 "캐시 = 그 사진의 실제 K·V"로 두면 원래 모델과 **수학적으로 같다**(오라클, 3-2절). 오차는 "상수로 바꾼 근사" 하나로만 생기고, 그 크기를 따로 잴 수 있다.
3. **정수 커널·NPU 검증**: ORT 정적 QDQ(U8+rr, 텐서별)와 Hailo DFC(`a8_w8`)에서 직접 확인한다.

## 3. 방법

### 3-1. 바꾸는 것 (구성 R1)

```
블록 0~4 : 그대로 (196 토큰)
블록 5 입력: 패치 55 삭제 → 195 토큰
블록 5~11 attention: K·V 뒤에 패치 55의 캐시 K₅..₁₁·V₅..₁₁ (헤드별 상수) 1개를 붙임
블록 9 입력: 패치 13 삭제 → 194 토큰
블록 9~11 attention: 패치 13의 캐시 K·V도 붙임
MAP 풀링 헤드: kv 뒤에 두 토큰의 캐시 K·V를 붙임
```

- sink 토큰이 다른 토큰에 영향을 주는 길은 **attention의 K·V뿐**이다 (Block = 토큰별 LN·MLP + attention). 그래서 K·V만 대체하면 나머지 토큰의 계산은 그대로다.
- 캐시 값: 원래 FP32 모델에서 **보정 구간(train) 256장, 시드 0**으로 sink 토큰의 층별·헤드별 K·V를 평균한다. dev는 쓰지 않는다 (CLAUDE.md 데이터 규칙).
- 지운 토큰 자리의 내용(패치 13은 블록 8까지 보통 내용 토큰)은 잃는다. 그 영향은 G1b에서 정확도로 잰다.

### 3-2. 비교 구성

| ID | 내용 | 용도 |
|---|---|---|
| R0 | 현재 최종: fc2L2 U8 + reduce_range (블록 5·9 fc2 FP32) | 기준 (D-030) |
| R1-oracle | R1인데 캐시 대신 **그 사진의 실제** sink K·V를 붙임 (FP32) | 구현 검증: 원래 FP32와 같아야 함 |
| **R1** | 패치 55·13 삭제 + 위치별 평균 캐시 | **주 구성** |
| R2 | 삭제만, 캐시 없음 | 소거 실험 (캐시가 필요한지) |
| R3 | R1 + 패치 1 (블록 5~7의 약한 sink, 64장 중 49~60장) | 조건부: R1 뒤에도 블록 5~7 잔차 범위가 크면 (5절 G1 기준) |

τ(레지스터 복사 수) 탐색은 하지 않는다. 위치별 캐시는 원래 attention 몫을 그대로 재현하므로 τ=1이 자연스럽다. G1b가 실패할 때만 원인 분석으로 시도한다.

### 3-3. 측정 (팀 프로토콜 그대로, PRD 5절)

- 정확도: dev 6종, 고화질 5종 평균 Δtop-1(짝 부트스트랩 1000회 95% 구간)과 cos. **Δ는 모두 원래 FP32 ONNX(`siglip2_image_fp32.onnx`, Baseline) 대비**로 적는다.
- 크기 MiB, peak 메모리(모델마다 새 프로세스), 지연 배치 1·워밍업 20·200장×3·스레드 1/2/4, `OPENBLAS_NUM_THREADS=1`.
- INT8 조건은 R0과 **변수 하나만** 다르게: MatMul만, Percentile 99.999, highres 256 s0, U8 + reduce_range, hist-chunk 16. 다른 점은 `--exclude` 없음 하나.
- 모든 결과 jsonl에 커밋 해시·dirty·옵션·CPU·ORT 버전 기록.

## 4. 단계와 예상 시간

| 단계 | 하는 일 | 어디서 | 예상 |
|---|---|---|---|
| 준비 | `s2_regcache.py` 작성 (통계·캐시 생성·모델 래퍼·오라클 검사·ONNX export). `s2_quantize.py`에 **기본 꺼진** `--src` 옵션 추가 (옵션 끄면 sha256 `5eee7839…` 그대로인지 확인) | Claude | 2~3시간 |
| 준비 | 무작위 가중치 SigLIP 구조로 오라클 동치·export 단위 검사 | Claude (클라우드) | 위에 포함 |
| G0 | sink 토큰 K·V가 사진마다 얼마나 같은지 (보정 64장 + dev 64장) | 데이터 있는 PC | 5분 |
| G1a | R1-oracle vs 원래 FP32 | 〃 | 5분 |
| G1b | R1·R2 FP32 정확도: 스크리닝 6×200 → 통과 시 dev 전체 + 블록별 잔차 범위 | 〃 | 10분 + 30분 |
| G2 | R1 ONNX export → INT8(예외 없음) → dev 전체 짝 비교(R0·R1) → 지연 3세션·peak 메모리 | 〃 | 약 1시간 30분 |
| G3a | `hailo parser --hw-arch hailo10h`로 R1 FP32 ONNX 파싱 (상수 Concat·Slice 지원 여부) | 연구실 데스크탑 WSL Ubuntu-22.04 | 10분 |
| G3b | (G2·G3a 통과 시) 16비트 층을 줄인 `.alls` → 에뮬레이터 Pets 64장 → 스크리닝 200×5 | 〃 | 약 2시간 30분 |
| G3c | (선택) 컴파일 → HEF 크기·컨텍스트 수 | 〃, 밤샘 | 약 2시간 |

데이터 있는 PC는 dev 캐시(`$SIGLIP2_ARTIFACTS/cache/*`)와 보정 후보(`calib_highres_f32.npy`)가 필요하다. 연구실 데스크탑에 없으면 `s2_build_dev.py all`로 만든다 (다운로드 포함, 시간은 확인 필요).

## 5. 판정 기준 (결과 전에 고정)

| 관문 | 통과 조건 | 실패하면 |
|---|---|---|
| G0 | (기록용) 층·헤드별 cos(사진의 sink K, 평균 K)와 V의 중앙값. **0.95 미만인 층이 있으면** 상수 근사 위험으로 표시 | 그래도 G1b까지는 간다 (최종 판단은 정확도) |
| G1a | 최종 임베딩 cos 최소 ≥ 0.99999 (dev 64장) | 구현 버그. 고치기 전에는 다음 단계 금지 |
| G1b | 스크리닝 cos ≥ 0.99 **그리고** dev 전체 Δtop-1 95% 구간 하한 ≥ −0.5%p (원래 FP32 대비) | 상수 근사가 안 맞음 → τ·R3 원인 분석만 하고 결과는 실패로 기록 |
| G1b 부가 | 블록 5~11 잔차 흐름 최대/99.9% 분위수 ≤ 10 (지금 39~68) | 10 초과 블록이 5~7이면 R3 시도 |
| **G2 (채택)** | ① R1-INT8 vs R0 짝 차이 95% 구간 하한 ≥ −0.3%p (정확도 비열등) ② 4스레드 지연 ≤ R0 (3세션 중앙값) ③ 크기 < R0 | 하나라도 실패 → 최종에 넣지 않음. 보고서에는 "시도·결과·원인"으로 기록 |
| G3a | 호스트로 떨어지는 구간 없이 파싱 성공 | 상수 K·V를 외부 입력으로 넣는 대안 그래프 시도 1회, 그래도 실패하면 ORT 전용으로 기록 |
| G3b | 16비트 층을 줄인 구성이 D256L1 이상의 스크리닝 cos (0.952) | 현재 D256L1 유지 |

최종 모델 교체 여부는 이 표를 근거로 **팀이 정한다** (CLAUDE.md). best 설정 변경은 협업 규칙 3절의 비교표(현재 best / +R1)로만 한다.

## 6. 미리 적어 두는 위험

- **대회 범위**: 토큰 삭제가 "양자화가 아닌 경량화"로 읽힐 수 있다. 목적은 연산 절감이 아니라 **텐서별 8비트를 가능하게 하는 그래프 변환**이다 (SmoothQuant·회전과 같은 부류). 토큰 수는 196 → 194로 계산량 차이는 약 1%다. 보고서에 이 점을 명시하고, 필요하면 학술부에 한 줄 문의한다.
- **CPU 속도 이득은 작다**: FP32로 남은 fc2 2개는 전체 연산의 약 5%라 지연 개선은 크지 않을 것으로 본다 (추정). 이 실험의 주 이득은 정확도·크기·Hailo 16비트 감소다.
- **패치 13 내용 손실**: 블록 8까지 오른쪽 위 모서리 내용을 실어 나르던 토큰을 지운다. CIFAR(작은 사진)에 더 민감할 수 있어 CIFAR A를 따로 본다.
- **준혁 트랙과의 관계**: 코드는 `siglip2/`에 새 파일로만 추가하고 기존 기본 동작은 바꾸지 않는다. 시작 전에 준혁에게 공유한다.

## 7. 실행 순서 (데이터 있는 PC, `siglip2/` 폴더)

코드: [`siglip2/s2_regcache.py`](../../../siglip2/s2_regcache.py), `s2_quantize.py --src`(기본 꺼짐).

```bash
export OPENBLAS_NUM_THREADS=1
R=../results/siglip2/G3R_regcache

# 0. 구현 검증 (데이터 불필요, 약 15초) — 무작위 가중치로 오라클 동치·ONNX·연산 종류
"$SIGLIP2_PYTHON" s2_regcache.py selftest
# 0-b. --src 옵션을 끈 기본 경로가 기존 최종과 비트 단위로 같은지 (sha256 5eee7839…, G3_w4 identity_check와 같은 방법)
RN50_RESULTS=$R/identity "$SIGLIP2_PYTHON" s2_quantize.py --calib-source highres --calib-n 256 --op-types MatMul \
  --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range \
  --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/" --tag idcheck_fc2L2
# 기존 최종 파일을 덮어쓰지 않도록 다른 이름(idcheck)으로 만든다. sha256은 Python hashlib로 (sha256sum은 역슬래시 경로에서 오판정, G3_w4)
"$SIGLIP2_PYTHON" -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" \
  "$SIGLIP2_ARTIFACTS/onnx/siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_idcheck_fc2L2.onnx"
# 기대값 5eee783955591d1de818b532ea73b2bb2ea8a008e327cc92d323992c0f9083ac
# 이 PC에서 5eee가 안 맞을 때의 대체 확인 (용 승인, 10/18): 같은 PC에서 --src 추가 전 커밋 c887a42의 s2_quantize.py(git worktree)로
#   같은 명령을 태그 idcheck_base로 돌려 지금 코드 결과(idcheck_fc2L2)와 sha256 비교. 같으면 코드 동일성 통과 + 5eee 불일치는 PC(CPU·패키지) 차이로 기록, 다르면 코드 문제로 멈춤

# 1. 평균 캐시 (보정 train 256장 s0)
RN50_RESULTS=$R "$SIGLIP2_PYTHON" s2_regcache.py cache
# G0 + G1b 부가 (sink K·V 일관성, 잔차·fc2 범위 비율)
RN50_RESULTS=$R "$SIGLIP2_PYTHON" s2_regcache.py stats --n 64
# G1a + G1b 스크리닝 (dev 6종 × 200장, PyTorch FP32)
RN50_RESULTS=$R "$SIGLIP2_PYTHON" s2_regcache.py fp32 --k 200 --modes oracle mean none

# 2. (G1b 스크리닝 통과 시) ONNX export → dev 전체 FP32 짝 비교
RN50_RESULTS=$R "$SIGLIP2_PYTHON" s2_regcache.py export --verify 64
RN50_RESULTS=$R/fp32_full "$SIGLIP2_PYTHON" s2_paired_ci.py --models siglip2_image_fp32.onnx siglip2_image_fp32_rc55at5-13at9.onnx

# 3. G2: 예외 없이 INT8 (R0과 --exclude만 다름) → R0과 짝 비교 → 지연·메모리
RN50_RESULTS=$R/int8 "$SIGLIP2_PYTHON" s2_quantize.py --src siglip2_image_fp32_rc55at5-13at9.onnx \
  --calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 \
  --act-type u8 --reduce-range --tag all --dump-ops
RN50_RESULTS=$R/int8 "$SIGLIP2_PYTHON" s2_paired_ci.py --models siglip2_image_fp32.onnx \
  siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx \
  siglip2_image_int8_rc55at5-13at9_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_all.onnx
# 지연 3세션 (s2_bench_speed.py, 스레드 1·2·4), peak 메모리 (s2_model_stats.py) — G3_cumulative/speed와 같은 명령에 모델만 추가

# 4. G3a: DFC 파싱 (WSL Ubuntu-22.04, ~/hailo10_venv, CUDA_VISIBLE_DEVICES=-1)
hailo parser onnx siglip2_image_fp32_rc55at5-13at9_b1.onnx --hw-arch hailo10h
```

### 구현 검증 기록 (selftest, 2026-10-18, 클라우드 CPU, timm 1.0.30 · torch 2.14.1)
- 무작위 가중치 `vit_base_patch16_siglip_224`, 사진 4장: sink K·V 추출 경로 = timm trunk (차이 0.0), **오라클 = 원래 (최대 차이 < 1e-4)**, 사진 1장의 자기 캐시 mean = 원래 (5.5e-7), ORT = PyTorch (1.3e-6).
- ONNX 연산: **MatMul 79개로 원래 모델과 같다.** 늘어난 것은 Concat(+34)·Expand(+16) 등 모양 연산뿐이고 TopK·NonZero·ScatterND는 없다. 배치 가변 export라 Expand·Where가 생겼고, Hailo용 b1 파일에서는 상수로 접힐 것으로 예상한다 (G3a에서 확인).
- 무작위 가중치에는 sink가 없으므로 정확도 의미는 없다. 실제 가중치 검증은 G1a.

## 8. 1단계 결과 (2026-10-18, 연구실 데스크탑 WSL Ubuntu-22.04, 커밋 `92d9617`, dirty 아님)

환경: Intel Core Ultra 9 285K (avx2 O, avx512f X, avx512_vnni X, avx_vnni O), Python 3.11.17, torch 2.14.0+cpu, timm 1.0.30, ORT 1.25.0.
고정 버전 13개 모두 requirements와 같음 (`env_labpc/`). dev 7종(CIFAR A·B + 고화질 5종)의 장수·보정 후보 수·FP32 top-1이 준혁 `G2_fp32_dev`와 같음.
기준 FP32는 Release `siglip2_image_fp32.onnx` (sha256 `333cad08…`).

| 관문 | 결과 | 판정 | raw |
|---|---|---|---|
| 0-a selftest (무작위 가중치) | 오라클·자기 캐시·ORT 검사 모두 통과, ORT vs PyTorch 9.6e-7 | 통과 | 실행 로그 |
| 0-b identity | Release `5eee…`와 불일치. 같은 PC에서 지금 코드 = `--src` 추가 전 코드(c887a42) = `410d7483…` | 코드 동일성 통과 (차이는 PC: 활성 눈금 127쌍 상대 3e-7~5e-6) | `identity/README.md` |
| cache | 보정 train 고화질 256장, 시드 0 → `regcache_55at5-13at9_highres256_s0.npz` sha256 `e6e8fbde…` (meta에 커밋 해시가 들어가 커밋마다 sha가 바뀜) | — | `cache.jsonl` |
| G0 (기록용) | sink K·V vs 평균 캐시, 최소 헤드 중앙 cos (K, V): b5 (0.998, 0.978), b6~b8·b10·b11·pool ≥ 0.993, **b9 (0.862, 0.394)** | **b9 0.95 미만 → 상수 근사 위험 표시** | `stats.jsonl` |
| G1b 부가 (dev 64장 미리보기) | 잔차 max/p99.9 원래 → mean: 블록 5~11 38~68 → 7.8~20 (b5 17.1, b6 20.0, b7 18.2, b8 13.2, b9 9.6, b10 7.8, b11 9.5) | 미리보기에서 b5~b8·b11이 10 초과. 최종 판정은 dev 전체로 | `stats.jsonl` |
| **G1a** | oracle 임베딩 cos 최소 **1.000000** (dev 6종 × 200장), top-1 원래와 모두 같음 | **통과** (≥ 0.99999) | `fp32_screen.jsonl` |
| **G1b 스크리닝** | mean(R1): 고화질 cos 평균 **0.99990** (최소 0.99854), Δtop-1 +0.00%p, CIFAR A +0.00%p | **통과** (≥ 0.99) | `fp32_screen.jsonl` |
| (소거) none(R2) | 고화질 cos 0.92128 (최소 0.75095), Δtop-1 −3.00%p, CIFAR A −4.50%p | 실패 → **캐시가 필요하다** | `fp32_screen.jsonl` |

- 스크리닝은 PyTorch FP32, 데이터셋마다 앞 200장. Δ는 Release FP32 ONNX(ref_fp32) 대비. `original`(변환 없음 PyTorch)도 cos 1.000000.
- b9의 V가 사진마다 꽤 다른데도(cos 0.39) mean의 최종 cos가 0.9999인 것은, 블록 9 이후 13번 sink로 가는 attention 몫이 작거나 V 차이가 출력에 크게 번지지 않기 때문으로 보인다 (가설, 확인 안 함).
- 실행 중 찾은 코드 버그 2개는 고친 뒤 위 결과를 다시 냈다: stats fc2 hook이 출력을 바꿔치기(`078342a`), fp32에서 oracle 다음 mean이 RegCacheTrunk를 이중으로 감쌈(`f55c3e7`). 결과 행 CPU·ORT 기록 추가(`92d9617`).
- **G2의 R0는 이 PC에서 만든 fc2L2(`410d…`, identity 파일과 같은 조건)를 쓴다** (용 결정, 10/18). R1-INT8과 같은 PC·같은 조건에서 만든 INT8끼리 비교하기 위해서다.

## 9. 2단계·G2·G3a 결과 (2026-10-18, 같은 PC)

### 요약표 (dev 전체, 정수 커널 ORT_ENABLE_ALL, 지연은 **비공식**)
| 구성 | 크기 MiB | 지연 t1 / t2 / t4 ms (3세션 중앙값) | peak 메모리 MiB | 고화질 Δtop-1 [95% 구간] | cos | CIFAR A Δ |
|---|---|---|---|---|---|---|
| FP32 (Release `333cad08…`) | 354.45 | 234.53 / 129.29 / 69.38 | 656.1 | 기준 (91.66%) | 1 | 기준 (76.90%) |
| R1 FP32 (`fbd4c5bf…`) | 354.52 | (재지 않음) | (재지 않음) | **+0.10 [0.00, +0.22]** | 0.9999 | 0.00 |
| R0 fc2L2 U8+rr, 이 PC (`410d7483…`) | 105.30 (−70.3%) | 95.17 / 55.38 / 31.10 (−59.4 / −57.2 / −55.2%) | 236.7 (−63.9%) | −0.62 [−1.09, −0.08] | 0.9548 | −5.10 |
| **R1-INT8** 예외 없음 (`825a84bb…`) | **91.84 (−74.1%)** | **89.03 / 51.37 / 29.81 (−62.0 / −60.3 / −57.0%)** | 256.1 (−61.0%) | −0.68 [−1.17, −0.19] | **0.9628** | −5.30 |
| **R1-INT8 − R0** | **−13.46 (−12.8%)** | **−6.14 / −4.01 / −1.29 (−6.5 / −7.2 / −4.1%)** | +19.4 (+8.2%) | **−0.06 [−0.44, +0.33]** | +0.008 | −0.20 |

- 고화질 FP32 평균 91.66% = Pets 93.33·Flowers 86.73·Cars 93.80·Food 89.44·Caltech 95.02의 평균. 데이터셋별 top-1: `int8/paired_ci.jsonl`, `fp32_full/paired_ci.jsonl`.
- R1-INT8 데이터셋별 Δ (R0): Pets −0.20 (−0.40), Flowers −1.46 (−0.84), Cars −0.73 (−0.63), Food −1.32 (−1.62), Caltech +0.29 (+0.39), CIFAR A −5.30 (−5.10).
- 정수 커널: R1-INT8은 MatMul 79개 **전부** QLinearMatMul (R0는 77개, fc2 2개 float). `int8/quantize_log.jsonl`. 보정 실제 장수 256.
- R0(이 PC)는 Release 파일(AVX-512 PC, −0.78 [−1.26, −0.31])과 비트 단위로 다르고 정확도도 조금 다르다 (identity 참고). 같은 PC에서 만든 INT8끼리 비교했다 (용 결정).

### G2 채택 판정 (5절 기준 그대로)
| 조건 | 결과 | 판정 |
|---|---|---|
| ① R1-INT8 vs R0 짝 차이 95% 구간 하한 ≥ −0.3%p | −0.06 [**−0.44**, +0.33] | **실패** |
| ② 4스레드 지연 ≤ R0 (3세션 중앙값) | 29.81 ≤ 31.10 ms | 통과 |
| ③ 크기 < R0 | 91.84 < 105.30 MiB | 통과 |
| **G2 (채택)** | ①이 미달 | **실패 (확정, 용 10/18).** 최종에 넣지 않음 |

**G2 기록 (보고서용: 시도·결과·원인)**
| 항목 | 내용 |
|---|---|
| 시도 | sink 토큰(패치 55@블록 5, 13@블록 9)을 상수 인덱스로 지우고 층별 평균 K·V를 attention·풀링 헤드에 붙여, fc2 FP32 예외 없이 MatMul 79개를 전부 INT8로 (R0와 `--exclude`만 다름) |
| 결과 | FP32 동치성은 지켰다 (G1b +0.10 [0.00, +0.22]%p). INT8은 크기 −12.8%, 4스레드 지연 −4.1%, cos 0.9548 → 0.9628로 좋아졌지만, R0 대비 정확도 −0.06 [−0.44, +0.33]%p로 비열등 기준(하한 ≥ −0.3%p)을 못 넘었다 |
| 원인 (**가설**, 확인 안 함) | R0와 R1 둘 다 FP32 대비 약 −0.6~−0.7%p를 잃는다. 둘이 같이 가진 손실이므로 sink 토큰(fc2)이 아니라 다른 곳, 특히 D-031에서 짚은 **QKV 투영의 U8 정수 경로·LayerNorm 출력 범위**가 원인일 가능성이 크다. 이 부분은 별도 회전 실험(`siglip2/g3q-rotate`)에서 다룬다. R3(1번 패치 추가 제거)·dev 전체 잔차 재측정은 하지 않는다 (용 결정, 10/18) |
| 남은 쓸모 | sink 제거 그래프는 "fc2 예외 없이 전부 정수"를 FP32 동치로 만든다는 것을 보였다. 회전 실험이 공통 손실을 줄이면, 그 위에서 R1을 다시 비교할 수 있다 |

- 참고: 점추정 차이는 −0.06%p로 거의 같다. dev 약 4900장으로는 "0.3%p보다 나쁘지 않다"를 보일 만큼 구간이 좁지 않다.

### 측정 조건·기록
- 지연: `s2_bench_speed.py --warmup 20 --n 200 --repeats 3`, 배치 1, 세 모델 번갈아, 스레드 1·2·4 × 3세션 (`scripts/g3r_speed.sh`, raw `speed/speed_*.json`, `speed/runs.jsonl` 9줄 모두 dirty 아님).
  세션 1·2(t1·t2)는 커밋 `56097dd`, 나머지는 `be28bd8` (측정 스크립트 게이트만 바뀜, 측정 코드 같음).
- 부하 기록 (`speed/gate.log`): 다른 WSL 배포판은 꺼져 있음(Ubuntu 26.04·docker-desktop Stopped). 측정 직전 다른 프로세스 CPU 합 3~4% (대부분 원격 데스크톱 `remoting_host`·`remoting_desktop` 2~4%).
  처음 게이트로 쓴 WMI `LoadPercentage`가 프로세스 합 3.5%일 때도 28~36%로 나와 s2 t4에서 10분 넘게 대기 → 게이트를 "다른 프로세스 CPU 합 < 10%"로 바꾸고 s2 t4부터 이어서 쟀다.
- 1차 측정(05:50~06:27)은 `siglip2/hailo_sdk.client.log`(DFC 로그) 때문에 dirty로 판정되어 `speed_dirty/`로 분리했다. 보고서에 쓰지 않는다.
- peak 메모리: `s2_model_stats.py`, 모델마다 새 프로세스, 추론 50회 중 최대 RSS, 1회 측정 (`model_stats/model_stats.jsonl`, `be28bd8`, dirty 아님).
  R1-INT8 peak가 R0보다 19 MiB 크다. 1차(dirty) 측정에서는 load 뒤 RSS가 R1 198.7 < R0 236.6이었다 → 메모리는 1회 측정이라 변동이 크다 (G2 판정 항목 아님).

### G3a (DFC 파싱): 실패 — `g3a/README.md`
- `/trunk/Concat_2` (블록 5 K에 캐시 상수 K를 토큰 축으로 붙이는 Concat)에서 `UnsupportedModelError`. DFC concat은 feature 축만 지원. 그 전에 MatMul shape 계산 IndexError.
- b1 파일에도 Expand 17·Where 17이 남았다 (7절 "b1에서 상수로 접힐 것" 예상과 다름).
- **재시도 (D-041, 사후 변경): 제자리 마스크 그래프로 파싱 통과.** 토큰 196개를 유지하고 sink 위치의 K·V를 상수로 바꿔 끼우는 방식으로, R1과 FP32 cos 최소 0.9999998, Concat·Slice·Expand·Where는 원래 FP32와 같은 수. 자세한 내용은 `g3a_mask/README.md`. G3b는 시작하지 않았다.

### 기록용: 다른 토큰이 sink 패치에 주는 attention 비중 (`attn_share.jsonl`, dev 64장, 원래 FP32, 헤드 평균 / 최대 헤드, 균등 분배 = 1/196 = 0.0051)
| 블록 | 패치 13 | 패치 55 |
|---|---|---|
| 5 | 0.0032 / 0.0059 | 0.060 / 0.104 |
| 8 | 0.0014 / 0.0027 | 0.434 / 0.564 |
| **9** | **0.0029 / 0.0078** | 0.447 / 0.571 |
| 10 | 0.048 / 0.119 | 0.427 / 0.646 |
| 11 | 0.051 / 0.085 | 0.222 / 0.400 |
| 풀링 헤드 | 0.025 / 0.081 | 0.140 / 0.475 |

- 확인: 블록 9에서 13번 패치는 아직 sink가 아니다 (받는 attention이 균등 분배보다 작음). 그래서 블록 9의 V 일관성이 0.39여도 최종 cos가 0.9999로 유지된다. 13번은 블록 10·11에서 attention을 받기 시작하고, 그때 V 일관성은 0.995·0.999다 (8절 G0).
- 55번 패치는 블록 2부터 이미 attention을 크게 받는다 (0.13). 지우는 시점(블록 5)보다 앞이다.

## 10. 참고

- Jiang et al., Vision Transformers Don't Need Trained Registers, NeurIPS 2025 (arXiv 2506.08010)
- RegCache: Activation Quantization of Vision Encoders Needs Prefixing Registers (arXiv 2510.04547)
- Chen et al., PrefixQuant (arXiv 2410.05265) — 입력과 무관한 이상치 토큰을 앞에 고정해 정적 양자화
- Darcet et al., Vision Transformers Need Registers (2023) · Sun et al., Massive Activations (2024)
