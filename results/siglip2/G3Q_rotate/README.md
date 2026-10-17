# G3Q: LayerNorm을 그대로 둔 오프라인 회전으로 SigLIP2 INT8 손실 줄이기 (설계, 2026-10-18)

브랜치 `siglip2/g3q-rotate` (기준 `origin/siglip2/g3-sensitivity` + `707c031`의 `s2_quantize.py --src`만). 담당: 용(junyong0111-cmd).
**이 README의 1~6절(설계·판정 기준)은 결과를 보기 전에 고정한다.** 결과는 7절 이후에 덧붙인다.

## 1. 한 줄 질문

잔차 흐름을 직교 행렬 Q로 돌린 "수학적으로 같은" 모델을 만들면, R0(fc2L2 U8 + reduce_range)와 같은 INT8 조건에서 정확도가 유의하게 좋아지는가?

## 2. 왜 이 실험인가 (근거)

- **D-031**: U8 손실(U8 − S8 = −0.41 [−0.78, −0.02]%p)은 전부 **QKV 투영의 정수 경로**에서 나온다 (QKV만 U8로 둔 모델 = U8 모델과 예측이 완전히 같음). QKV 투영의 입력은 **LayerNorm1 출력**이다.
- **G3R 9절**: sink 토큰을 빼서 fc2를 전부 정수로 해도 R0와 R1-INT8 모두 FP32 대비 약 −0.6~−0.7%p를 잃는다. 둘이 같이 가진 손실이라 LayerNorm 출력 → QKV(그리고 fc1·풀링 kv) 쪽이 원인일 가능성이 크다 (가설).
- **G3_why**: 블록 5~11 잔차 흐름에 특정 채널(67·301·145·483)의 거대 값이 있다. LayerNorm은 토큰마다 나누기만 하고 채널별 γ를 곱하므로, 채널 쏠림이 LN 출력과 그 뒤 MatMul 입력의 텐서별 8비트 눈금에 남는다.
- 회전(QuaRot·SpinQuant 계열)은 채널 쏠림을 768채널 전체에 고르게 퍼뜨린다. 여기서는 그래프를 바꾸지 않고(LayerNorm 연산을 그대로 두고) 가중치만 바꾼다 → ORT·Hailo 모두 원래 FP32 ONNX와 같은 연산 구성.

## 3. 방법

### 3-1. 변환 (가중치만 바꿈, 그래프 연산 종류·개수는 원래와 같음)

표기: 잔차 흐름 x (토큰마다 768차원 열벡터), n = 768, 1 = 모두 1인 벡터, C = I − 11ᵀ/n (중심화).

1. **LayerNorm 접기**: norm1 → `attn.qkv`, norm2 → `mlp.fc1`, 최종 `trunk.norm` → `attn_pool.kv`.
   `W ← W·diag(γ)`, `b ← b + W·β` (원래 W 기준), LN은 γ = 1, β = 0.
2. **중심화**: 잔차에 더해지는 것 = 패치 임베딩 conv 가중치·bias, `pos_embed`, 블록마다 `attn.proj`·`mlp.fc2`의 가중치·bias.
   출력 채널 방향으로 `W ← C·W`, `b ← C·b`, `pos ← pos·C`. LayerNorm은 토큰별 평균 이동에 불변이고, 잔차 흐름을 읽는 곳은 norm1·norm2·trunk.norm 셋뿐이다 (timm `Block`: `x + attn(norm1(x))`, `x + mlp(norm2(x))`; ls1·ls2·norm_pre·fc_norm·head는 Identity, CLS·register 토큰 없음, attn_pool.pos_embed 없음 — 코드에서 assert).
3. **회전**: Q (768×768 직교, Q·1 = 1). 쓰는 쪽 `W ← Q·W`, `b ← Q·b`, `pos ← pos·Qᵀ`. 읽는 쪽(접힌 qkv·fc1·attn_pool.kv) `W ← W·Qᵀ`.
   - 근거: Q가 직교이고 Q1 = 1이면 Qᵀ1 = 1 → mean(Qx) = mean(x), Qx − mean·1 = Q(x − mean·1), ‖Qx − mean·1‖ = ‖x − mean·1‖ → **LN₀(Qx) = Q·LN₀(x)** (LN₀ = γ 1, β 0, eps 포함 그대로).
   - 참고: Q1 = 1이면 이 등식은 중심화 없이도 성립한다. 중심화(2)는 사양대로 넣는다 (잔차 텐서의 토큰별 평균을 0으로 만들어 Add 쪽 눈금에도 영향을 줄 수 있음, 동치성은 그대로).
   - MLP 내부(3072)와 attention 헤드 내부(q·k·v, proj 입력)는 회전하지 않는다. 풀링 헤드의 q(latent)·proj·norm·mlp는 잔차 흐름이 아니므로 그대로.
4. 계산은 float64로 하고 마지막에 float32로 바꾼다.

### 3-2. Q 후보 (미리 두 개만, 결과를 본 뒤 다른 Q를 추가하지 않는다)

공통: P = e₁과 1/√n을 맞바꾸는 Householder 반사 (`P = I − 2vvᵀ/‖v‖²`, `v = e₁ − 1/√n`, P는 대칭·직교, P·e₁ = 1/√n).

| 이름 | 구성 | Q·1 = 1 확인 |
|---|---|---|
| `rotrand0` (a) | `Q = P·diag(1, R)·P`, R = 767×767 무작위 직교 (torch 생성기 시드 0, float64 정규분포 → QR, R 대각 부호로 부호 고정) | P·1 = √n·e₁ → diag(1,R)는 e₁ 고정 → P·e₁·√n = 1 |
| `rothad` (b) | `Q = (H/√n)·P`, H = H₁₂ ⊗ H₆₄ (H₁₂ = Paley 구성, H₆₄ = Sylvester), 둘 다 첫 행·첫 열이 모두 1이 되게 정규화 → H의 첫 열 = 1 | P·1 = √n·e₁ → (H/√n)·√n·e₁ = H의 첫 열 = 1 |

### 3-3. 비교 구성 (모두 이 PC, 같은 INT8 조건)

| 구성 | 설명 |
|---|---|
| R0 | 이 PC에서 만든 fc2L2 U8 + rr (`siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_idcheck_fc2L2.onnx`, sha256 `410d7483…`, G3R 8절, 용 결정) |
| Rot-FP32 | 회전 FP32 ONNX `siglip2_image_fp32_<rotrand0|rothad>.onnx` (+ `_b1`) |
| Rot-INT8 | 회전 FP32를 R0와 **같은 명령 + `--src`만** 다르게 양자화: `--calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/" --tag fc2L2` (노드 이름이 원래와 같은지 Q0에서 확인) |
| (덤) Rot-INT8-all | 같은 조건에서 `--exclude` 없이 전부 INT8 (`--tag all`). 기록용, 판정에 안 씀 |

### 3-4. 측정 (팀 프로토콜 그대로, PRD 5절)
- 정확도: 정수 커널, ORT 1.25.0 `ORT_ENABLE_ALL`, 4스레드. 스크리닝 `s2_screen.py --k 200` (dev 6종 × 200장), dev 전체 `s2_paired_ci.py` (층화 짝 부트스트랩 1000회, 고화질 5종 평균).
- 지연: `s2_bench_speed.py --warmup 20 --n 200 --repeats 3`, 배치 1, 모델 번갈아, 스레드 1·2·4 × 3세션 (G3R과 같은 방법). **시작 전에 용에게 묻고, `ps`로 다른 python·hailo 작업이 없는지 확인한 뒤에만** 잰다. 이 PC 속도는 "비공식".
- 환경: Intel Core Ultra 9 285K (avx2 O, avx512f X, avx512_vnni X, avx_vnni O), WSL Ubuntu-22.04, venv `~/venvs/siglip2` (G3R과 공유, 새 패키지 설치 안 함). DFC 5.4.0 `~/hailo10_venv`, `CUDA_VISIBLE_DEVICES=-1`.
- 산출물 이름에 `rot` 태그를 붙여 G3R 파일과 겹치지 않게 한다.

## 4. 단계와 판정 기준 (결과 전에 고정)

| 단계 | 내용 | 통과 기준 | 실패 시 |
|---|---|---|---|
| **Q0** selftest | 무작위 가중치 `vit_base_patch16_siglip_224` (LN γ·β와 bias도 무작위로), 두 Q 모두 | ① 회전 모델 = 원래 (PyTorch 출력 최대 차이 < 1e-4) ② ORT = PyTorch (회전 ONNX, < 1e-4) ③ ONNX 연산 종류별 개수가 원래 export와 같음 (노드 이름 목록도 같음) ④ QᵀQ = I, Q1 = 1 (float64 오차 < 1e-10) | 구현 버그 → 고친다 |
| **Q1** FP32 동치 (실제 가중치) | 회전 FP32 ONNX(배치 가변·b1) ORT 임베딩 vs 원래 FP32 ONNX 임베딩(dev 캐시 `ref_fp32`), dev 64장 (Pets·Caltech dev 앞 32장씩, G3_why·G3R과 같은 구성) | cos 최소 ≥ **0.99999** (두 Q, 두 파일 모두) | 구현 버그 → **멈추고 보고** |
| **Q2** 스크리닝 | Rot-INT8(fc2L2)과 R0를 `s2_screen.py --k 200`으로 같은 실행에서 잼 | Q별로: 고화질 cos ≥ R0 고화질 cos. 미달 Q는 **실패로 기록**. 둘 다 미달이면 **멈춤** | 기록 후 멈춤 |
| Q2 → Q3 | Q2를 통과한 Q 가운데 스크리닝 고화질 cos가 더 높은 **하나만** Q3로 보낸다 (같으면 `rothad`) | — | — |
| **Q3** 채택 | ① dev 전체 짝 비교에서 Rot-INT8 − R0 고화질 Δtop-1 95% 구간 **하한 > 0** ② 4스레드 지연 ≤ R0 (3세션 중앙값, 같은 측정 안에서) ③ G3a: `hailo parser onnx <회전 b1> --hw-arch hailo10h`가 호스트로 떨어지는 구간 없이 성공 (`-y` 없이) | 셋 다 만족하면 **채택 후보** (최종 교체는 팀 결정) | 하나라도 미달이면 **채택 실패**로 기록 |

- 크기는 같은 조건의 양자화라 사실상 같으므로 비교하지 않는다 (기록만).
- Q3 ①은 "유의하게 좋음"이다. 점추정이 좋아도 구간이 0을 포함하면 실패로 기록한다.
- 덤(Rot-INT8-all)은 Q2와 같은 스크리닝으로 기록만 한다.

## 5. 미리 적어 두는 위험

- 회전은 채널 쏠림은 퍼뜨리지만 **토큰 쏠림**(sink 토큰 2개의 거대 노름)은 그대로다. 텐서별 눈금이 거대 노름 토큰에 맞춰지는 손실(G3_why)은 회전으로 줄지 않을 수 있다. LN 출력은 토큰별로 정규화되므로 QKV 입력 쪽은 영향이 작을 것으로 본다.
- 회전된 가중치는 출력 채널별 분포가 바뀐다. per-channel 가중치 양자화라 큰 문제는 아니라고 보지만, 패치 임베딩 conv 등 쓰는 쪽 가중치의 범위가 달라진다 (MatMul만 양자화하므로 conv는 FP32).
- 잔차 흐름 Add 텐서와 fc2 출력은 MatMul만 양자화하는 조건에서 QDQ가 걸리는 범위가 원래와 같다 (MatMul 입력·출력). fc2 출력(잔차에 씀)도 회전되므로 fc2 출력 눈금이 바뀐다.
- 이 PC의 R0는 Release 파일과 비트 단위로 다르다 (G3R identity). 비교는 이 PC에서 만든 INT8끼리만 한다.

## 6. 실행 순서 (`siglip2/` 폴더)

```bash
source ~/siglip2_env.sh
export OPENBLAS_NUM_THREADS=1
R=../results/siglip2/G3Q_rotate

# Q0 (데이터 불필요)
"$SIGLIP2_PYTHON" s2_rotate.py selftest
# Q 행렬 생성·확인
RN50_RESULTS=$R "$SIGLIP2_PYTHON" s2_rotate.py build
# Q1: PyTorch 동치(dev 64장) → ONNX export(배치 가변 + b1) + ORT 동치
RN50_RESULTS=$R "$SIGLIP2_PYTHON" s2_rotate.py fp32 --q rotrand0 rothad
RN50_RESULTS=$R "$SIGLIP2_PYTHON" s2_rotate.py export --q rotrand0
RN50_RESULTS=$R "$SIGLIP2_PYTHON" s2_rotate.py export --q rothad
# Q2: 양자화 (R0와 --src만 다름) → 스크리닝
for q in rotrand0 rothad; do
  RN50_RESULTS=$R/int8 "$SIGLIP2_PYTHON" s2_quantize.py --src siglip2_image_fp32_$q.onnx \
    --calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 \
    --act-type u8 --reduce-range --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/" --tag fc2L2 --dump-ops
done
RN50_RESULTS=$R/screen "$SIGLIP2_PYTHON" s2_screen.py --k 200 --tag rot --models \
  siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_idcheck_fc2L2.onnx \
  siglip2_image_int8_rotrand0_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx \
  siglip2_image_int8_rothad_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
# Q3 ①: dev 전체 짝 비교 (Q2에서 고른 하나)
RN50_RESULTS=$R/full "$SIGLIP2_PYTHON" s2_paired_ci.py --models siglip2_image_fp32.onnx \
  siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_idcheck_fc2L2.onnx \
  siglip2_image_int8_<q>_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx
# Q3 ③: DFC 파싱 (WSL, ~/hailo10_venv)
CUDA_VISIBLE_DEVICES=-1 hailo parser onnx siglip2_image_fp32_<q>_b1.onnx --hw-arch hailo10h < /dev/null
# Q3 ②: 지연 3세션 (용 확인 + ps 확인 후)
```
