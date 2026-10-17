# G3R 판별 실험: sink 토큰을 레지스터로 빼내 SigLIP2를 전부 8비트로 (설계, 2026-10-18)

> 상태: **설계만 완료. 아직 아무것도 실행하지 않았다.** 판정 기준(5절)은 결과를 보기 전에 고정한다.
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

## 8. 참고

- Jiang et al., Vision Transformers Don't Need Trained Registers, NeurIPS 2025 (arXiv 2506.08010)
- RegCache: Activation Quantization of Vision Encoders Needs Prefixing Registers (arXiv 2510.04547)
- Chen et al., PrefixQuant (arXiv 2410.05265) — 입력과 무관한 이상치 토큰을 앞에 고정해 정적 양자화
- Darcet et al., Vision Transformers Need Registers (2023) · Sun et al., Massive Activations (2024)
