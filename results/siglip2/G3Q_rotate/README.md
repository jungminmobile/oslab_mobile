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
5. ONNX 후처리: 접은 LN 25개의 γ(모두 1)·β(모두 0)가 같은 값이라 torch exporter가 initializer를 하나로 합치고 `Identity` 노드 48개로 잇는다 (원래 export에는 Identity 없음, Q0에서 발견). export 뒤 그 Identity마다 출력 이름으로 initializer 사본을 만들고 노드를 지워 원래와 같은 그래프로 되돌린다 (`undedup_initializers`, 계산은 같음).

### 3-2. Q 후보 (미리 두 개만, 결과를 본 뒤 다른 Q를 추가하지 않는다)

공통: P = e₁과 1/√n을 맞바꾸는 Householder 반사 (`P = I − 2vvᵀ/‖v‖²`, `v = e₁ − 1/√n`, P는 대칭·직교, P·e₁ = 1/√n).

| 이름 | 구성 | Q·1 = 1 확인 |
|---|---|---|
| `rotrand0` (a) | `Q = P·diag(1, R)·P`, R = 767×767 무작위 직교 (torch 생성기 시드 0, float64 정규분포 → QR, R 대각 부호로 부호 고정) | P·1 = √n·e₁ → diag(1,R)는 e₁ 고정 → P·e₁·√n = 1 |
| `rothad` (b) | `Q = (H/√n)·diag(1, S)·P`, H = H₁₂ ⊗ H₆₄ (H₁₂ = Paley 구성, H₆₄ = Sylvester), 둘 다 첫 행·첫 열이 모두 1이 되게 정규화 → H의 첫 열 = 1. S = ±1 부호 767개 (torch 생성기 시드 0, QuaRot의 무작위 부호 Hadamard와 같은 꼴) | P·1 = √n·e₁ → diag(1,S)는 e₁ 고정 → (H/√n)·√n·e₁ = H의 첫 열 = 1 |

- **설계 수정 기록 (10/18, Q0 selftest에서 발견, 실제 가중치 결과 보기 전)**: 처음 사양 `Q = (H/√n)·P`는 H/√n과 P가 같은 두 벡터(e₁ ↔ 1/√n)를 맞바꾸기 때문에 Q·e₁ = e₁이 되어 **채널 0이 전혀 섞이지 않았다** (Q의 원소 하나가 1). 부호 S를 끼워 이것을 없앴다. 두 Q 모두 max|Q| < 0.5를 selftest에서 확인한다 (rotrand0 0.164, rothad 0.141).

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
| **Q0** selftest | 무작위 가중치 `vit_base_patch16_siglip_224` (LN γ·β와 bias도 무작위로), 두 Q 모두 | ① 회전 모델 = 원래 (PyTorch 출력 최대 차이 < 1e-4) ② ORT = PyTorch (회전 ONNX, < 1e-4) ③ ONNX 연산 종류별 개수가 원래 export와 같음 (연산 순서, 범위가 붙은 노드 이름, initializer 이름 목록도 같음. 자동 번호 이름 `Constant_<번호>`는 후처리로 번호만 밀리므로 제외) ④ QᵀQ = I, Q1 = 1 (float64 오차 < 1e-10) | 구현 버그 → 고친다 |
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

## 7. Q0 selftest 결과 (2026-10-18, 연구실 데스크탑 WSL Ubuntu-22.04, timm 1.0.30 · torch 2.14.0+cpu · ORT 1.25.0)

무작위 가중치 `vit_base_patch16_siglip_224` (LN γ·β, 모든 bias, pos_embed 무작위 + 블록 5 fc2 bias 채널 67에 +20), 사진 4장. `"$SIGLIP2_PYTHON" s2_rotate.py selftest`, 약 50초.

| 검사 | rotrand0 | rothad | 기준 |
|---|---|---|---|
| QᵀQ − I 최대, Q1 − 1 최대 | 2.0e-15, 3.0e-14 | 1.4e-15, 1.8e-14 | < 1e-10 |
| max\|Q\| (고정 기저 없음) | 0.164 | 0.141 | < 0.5 |
| ① 회전 vs 원래 PyTorch 출력 최대 차이 | 8.3e-7 | 9.5e-7 | < 1e-4 |
| LN₀(Qx) = Q·LN₀(x) (블록 0 norm1) | 4.4e-6 | 4.1e-6 | < 1e-4 |
| 잔차 토큰 평균 (중심화) | 4.7e-8 | 3.0e-8 | < 1e-4 |
| ② ORT vs PyTorch (배치 가변 / b1) | 1.1e-6 / 1.1e-6 | 1.1e-6 / 1.0e-6 | < 1e-4 |
| ③ 연산 종류별 개수, 연산 순서·범위 이름·initializer 이름 (배치 가변 / b1) | 같음 / 같음 | 같음 / 같음 | 같음 |

- **Q0 통과.** 도중에 찾은 두 가지(LN initializer 합치기로 생긴 Identity 48개, rothad의 채널 0 고정)는 3-1절 5번, 3-2절 설계 수정 기록에 적고 고친 뒤 위 결과를 냈다.

## 8. Q1·Q2·Q3 결과 (2026-10-18, 같은 PC, 정수 커널 ORT 1.25.0 `ORT_ENABLE_ALL`, 4스레드)

모든 판정 행은 커밋 `f086aea`, dirty 아님 (`build.jsonl`·`fp32.jsonl` 마지막 줄, `int8/quantize_log.jsonl`, `screen/screen.jsonl`, `full/paired_ci.jsonl`).
- `export.jsonl` 두 줄은 `dirty: true`다. 실행 당시 커밋 안 된 변경이 `s2_rotate.py`의 Q 로드 검사(`f086aea`로 커밋한 내용 그대로)뿐이었다. 그 뒤 코드를 바꾸지 않았으므로 실행 코드 = `f086aea`. 같은 ONNX를 다시 쓰면 그 파일로 만든 INT8과 섞일 수 있어 다시 내보내지 않았다.
- 첫 양자화(rotrand0 fc2L2)는 `f086aea` 커밋 직전에 시작했다 (차이는 위와 같은 `s2_rotate.py` 한 곳, `s2_quantize`와 무관). 로그에는 `f086aea`, dirty 아님으로 남았다.

### Q1 FP32 동치: **통과** (두 Q)
| | rotrand0 | rothad |
|---|---|---|
| Q 파일 sha256 | `45b7b8c6…` | `3a6391a2…` |
| PyTorch 회전 vs 원래 PyTorch 최대 차이 (dev 64장) | 2.1e-5 | 1.0e-5 |
| ORT 회전 ONNX vs 원래 FP32 임베딩 cos 최소 (배치 가변 / b1) | 0.99999988 / 0.99999988 | 0.99999988 / 0.99999988 |
| ONNX sha256 (배치 가변 / b1) | `16764bf9…` / `847b75e4…` | `ae4ccab1…` / `7a07bdee…` |
| 연산·순서·범위 이름·initializer 이름 vs Release FP32(`333cad08…`)·원래 b1 | 같음 | 같음 |

기록용 진단 (`fp32.jsonl`, dev 64장, PyTorch): **LN 출력의 채널 쏠림이 거의 사라진다.**
| LN 출력 (= MatMul 입력) | 원래 | rotrand0 | rothad |
|---|---|---|---|
| qkv 입력 채널별 max의 최대/중앙값 (블록 0~11) | 4.8 ~ 11.9 | 1.35 ~ 1.75 | 1.33 ~ 1.65 |
| qkv 입력 max / p99.9 | 2.37 ~ 4.27 | 1.52 ~ 1.70 | 1.49 ~ 1.90 |
| fc1 입력 max / p99.9 | 3.30 ~ 8.23 | 1.52 ~ 1.74 | 1.46 ~ 1.74 |
| attn_pool.kv 입력 max (max/p99.9) | 41.0 (2.25) | 5.1 (1.56) | 5.4 (1.68) |

### Q2 스크리닝 (dev 6종 × 200장, `screen/screen.jsonl`): **두 Q 모두 통과**
| 구성 | sha256 | MiB | 고화질 cos | 고화질 Δtop-1 | CIFAR A top-1 (FP32 74.5) |
|---|---|---|---|---|---|
| R0 fc2L2 U8+rr (`410d7483…`) | | 105.3 | 0.9547 | −0.1 | 70.5 |
| rotrand0 fc2L2 | `c77bb160…` | 105.3 | 0.9633 | +0.3 | 70.5 |
| **rothad fc2L2** | `0d11f89a…` | 105.3 | **0.9684** | +0.3 | 68.5 |
| (덤) rotrand0 전부 INT8 | `06e41dd3…` | 91.8 | 0.7209 | −52.0 | 19.0 |
| (덤) rothad 전부 INT8 | `f9ba7539…` | 91.8 | 0.7277 | −49.7 | 18.5 |

- 정수 커널: fc2L2 두 모델은 R0와 같이 QLinearMatMul 77개 + float MatMul 2개 (`int8/quantize_log.jsonl`). 보정 실제 장수 256.
- 4절 규칙대로 cos가 높은 **rothad**만 Q3로 보냈다.
- 덤: 회전만으로는 fc2 예외 없이 전부 INT8을 버티지 못한다 (cos 0.72). 거대 값은 토큰 쏠림(sink 토큰)이라 채널 회전으로 풀리지 않는다는 5절 위험과 맞는다. fc2 입력(3072, GELU 뒤)은 회전하지 않는다.

### Q3 채택 판정 (rothad fc2L2, 4절 기준 그대로)
dev 전체 짝 비교 (`full/paired_ci.jsonl`, 층화 짝 부트스트랩 1000회):
| 구성 | Pets | Flowers | Cars | Food | Caltech | 고화질 Δtop-1 [95%] (FP32 91.66%) | cos | CIFAR A (FP32 76.90) |
|---|---|---|---|---|---|---|---|---|
| R0 (`410d…`) | 92.93 | 85.89 | 93.17 | 87.82 | 95.41 | −0.62 [−1.09, −0.08] | 0.9548 | 71.80 |
| rothad fc2L2 | 92.83 | 86.00 | 93.80 | 87.82 | 94.82 | −0.61 [−1.07, −0.14] | **0.9685** | 71.40 |
| **rothad − R0** | −0.10 | +0.11 | +0.63 | 0.00 | −0.59 | **+0.01 [−0.45, +0.48]** | +0.014 | −0.40 |

- R0 값은 G3R 9절(−0.62 [−1.09, −0.08], cos 0.9548)과 같다 (같은 파일·같은 PC 재현).

| 조건 | 결과 | 판정 |
|---|---|---|
| ① rothad − R0 95% 구간 하한 > 0 | +0.01 [**−0.45**, +0.48] | **실패** |
| ② 4스레드 지연 ≤ R0 | 재지 않음 (①에서 이미 실패. 재려면 용 확인 후 `scripts/g3q_speed.sh`) | — |
| ③ G3a `hailo parser --hw-arch hailo10h` (b1, `-y` 없음) | 두 Q 모두 rc=0, 22초. 시작·끝 노드가 원래 FP32 G1 파싱과 같음 (`image` → `/trunk/attn_pool/Gather`). `g3a/parse_*.log` | 통과 |
| **Q3 (채택)** | ①이 미달 | **채택 실패** — 최종(R0)을 바꾸지 않는다 |

### 해석 (가설 포함)
- 회전은 의도한 일을 했다: LN 출력의 채널 쏠림이 거의 사라졌고(채널 max 최대/중앙 5~12 → 1.3~1.8), 정수 모델의 FP32 대비 cos가 0.955 → 0.969로 올랐다. 그런데 top-1은 그대로다 (점추정 +0.01%p).
- 가설 1: D-031의 U8 손실 원인 중 **채널 쏠림 몫은 작다.** 남은 손실은 텐서별 눈금이 거대 노름 토큰(sink)에 맞춰지는 토큰 쏠림이나, LN 출력 아닌 다른 MatMul 입력(attention 확률·V, fc2 입력)에서 나올 수 있다 (확인 안 함).
- 가설 2: cos 개선(+0.014)이 top-1을 바꿀 만큼 결정 경계 근처 사진에 몰리지 않았다. dev 약 4900장에서 구간 폭이 ±0.47%p라, 0.3%p 정도의 작은 개선은 이 표본으로는 보이지 않는다.
- 남은 쓸모: 그래프가 원래와 같아 ORT·Hailo 어디에나 그대로 들어간다. Hailo `a8_w8`에서 LN 출력 쪽 손실이 크다면(DFC는 다른 보정·눈금을 씀) 같은 회전 ONNX로 시험해 볼 수 있다. G3R(sink 제거 마스크 그래프)과 함께 쓰면 토큰 쏠림·채널 쏠림을 둘 다 다루는 조합이 된다 — 둘 다 이번 판정 밖이고, 하려면 새 설계·기준이 필요하다.

## 9. 플랫폼 재현 확인 (기록용, 판정 아님): `xplat/README.md`
- 같은 dev 6×200 묶음·같은 `portable_eval.py`로 x86 285K(ORT 1.25.0)와 ARM AGX Orin(ORT 1.23.2)을 쟀다. R0 대비 rothad 고화질 cos는 **+0.0136 (x86) / +0.0134 (ARM)**로 같고, 순서 rothad > rotrand0 > R0도 같다. 200장 top-1 차이는 모두 구간이 0을 포함한다.
- 10/16 기록의 Orin·285K U8+rr(5eee) 예측과 사진 단위로 비트까지 같다 (재현 확인).

## 10. 진단 (판정용 아님): ORT에 남은 손실(R0·rothad 모두 FP32 대비 약 −0.6%p)이 attention 행렬곱에서 나오는가 (설계, 10/18 — 실행 전에 고정)
### 10-0. 이미 돌렸는지
- 아니다. `G3_sensitivity`의 "attention 행렬곱"은 그 연산만 INT8로 한 **민감도** 실험이고(거의 무손실), 최종 구성(R0)에서 attention 행렬곱만 FP32로 되돌린 실험은 없다. `G3R_regcache/diag5`는 DFC 층별 SNR이라 다르다.

### 10-1. 전제 확인: R0의 attention 행렬곱은 정수 커널인가 — **그렇다 (가설 기각 안 됨)**
- R0(`410d…`) QDQ 모델: 블록 attention MatMul 24개 모두 입력 앞에 DequantizeLinear. ORT_ENABLE_ALL 최적화 그래프(`optimized_model_filepath`로 저장): 24개 모두 **QLinearMatMul** (그래프 전체 QLinearMatMul 77 = 79 − fc2 2개). softmax는 QLinearSoftmax(U8 출력).
- 노드 (`diag/attn_matmul_nodes.csv`, R0·rothad·두 FP32 ONNX 모두 같은 이름):
  - **Q × Kᵀ** = `/trunk/blocks/blocks.N/attn/MatMul` (입력: scale Mul, Kᵀ Transpose)
  - **확률 × V** = `/trunk/blocks/blocks.N/attn/MatMul_1` (입력: Softmax, V Squeeze)
  - 풀링 헤드의 attention 2개(`/trunk/attn_pool/…`)는 대상 아님.

### 10-2. 가설
- 남은 손실은 softmax 출력(attention 확률)을 U8 텐서별 눈금으로 재는 데서 나온다. 균등 확률 1/196 ≈ 0.005가 눈금 한 칸(≈ 1/255)과 비슷해서 대부분 0~1칸으로 뭉개지고, sink 토큰에 확률이 몰리면(블록 6~11에서 27~52%, G3_why) 눈금이 더 거칠어진다.

### 10-3. 진단 모델 (R0와 같은 명령, `--exclude`에만 추가)
공통: `--calib-source highres --calib-n 256 --op-types MatMul --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range`, 기본 exclude `/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/`.
| 이름 | 추가 exclude | 추가 keep | FP32로 남는 attention 행렬곱 |
|---|---|---|---|
| D1 | `/attn/MatMul_1` | — | 확률 × V 12개 |
| D2 | `/attn/MatMul` | 블록 0~11 `/attn/MatMul_1` 12개 (부분 문자열이라 MatMul_1도 걸리므로) | Q × Kᵀ 12개 |
| D3 | `/attn/MatMul` | — | 24개 모두 |
- 같은 3개를 회전 FP32(`siglip2_image_fp32_rothad.onnx`) 위에도 만든다 (rothad_D1~D3). 실제로 빠지는 노드 수를 양자화 전에 `select_excluded`로 확인하고, 양자화 로그·최적화 그래프의 QLinearMatMul 수로 다시 확인한다.

### 10-4. 측정·해석 규칙 (실행 전에 고정)
- dev 6×200 스크리닝 (`s2_screen.py --k 200`, 정수 커널, 4스레드), R0·rothad를 같은 실행에 넣는다. 표: 구성 / 정수 MatMul 수 / 고화질 cos / 고화질 Δtop-1 (FP32 대비) / CIFAR A Δ.
- **D1 또는 D3가 R0 대비 고화질 Δtop-1을 +0.3%p 이상 회복하면 "attention 확률이 주원인 후보"** → dev 전체 짝 비교 1회 (R0 vs 가장 좋은 D, 고화질 Δtop-1 기준, 같으면 cos가 높은 쪽). 아니면 **"attention 행렬곱은 주원인 아님"**으로 기록하고 끝.
- rothad_D1~D3는 같은 규칙을 rothad 대비로 적용해 기록만 한다 (dev 전체 비교는 R0 쪽 규칙으로만 결정).
- 지연은 재지 않는다. 양자화·스크리닝 전 `free -g` 여유 ≥ 10 GB 확인 (G3R 세션 DFC와 공유).

### 10-5. 결과 (10/18 18:28~18:44, 커밋 `c13e575`, dirty 아님, `diag/quantize_log.jsonl`·`diag/screen/screen.jsonl`, 실행 `scripts/g3q_diag_attn.sh`)
- 보정 실제 장수 256 (6개 모두). 다른 세션과 메모리 공유: 시작 시 여유 20 GB, 대기 없음.

| 구성 | 정수 MatMul (QLinearMatMul) | float attention 행렬곱 | 블록 softmax | 고화질 cos | 고화질 Δtop-1 (FP32 대비) | CIFAR A Δ |
|---|---|---|---|---|---|---|
| R0 | 77 | 0 | QLinearSoftmax 12 | 0.9547 | −0.1 | −4.0 |
| D1 (확률×V float) | 65 | MatMul 12 | float Softmax 12 | 0.9555 | 0.0 | −4.0 |
| D2 (Q×Kᵀ float) | 65 | FusedMatMul 12 | float Softmax 12 | 0.9546 | −0.3 | −5.0 |
| D3 (둘 다 float) | 53 | 24 | float Softmax 12 | 0.9553 | +0.1 | −5.5 |
| rothad | 77 | 0 | QLinearSoftmax 12 | 0.9684 | +0.3 | −6.0 |
| rothad_D1 | 65 | 12 | float 12 | 0.9696 | +0.1 | −8.0 |
| rothad_D2 | 65 | 12 | float 12 | 0.9684 | +0.3 | −6.5 |
| rothad_D3 | 53 | 24 | float 12 | 0.9697 | −0.5 | −6.5 |

- dev 6종 × 앞 200장, 정수 커널 ORT 1.25.0 4스레드. R0·rothad 값은 Q2(8절)와 같다 (재현).
- D2의 Q×Kᵀ 12개는 ORT가 scale·transpose와 합쳐 `com.microsoft.FusedMatMul`(float)로 돌렸다. D1~D3 모두 블록 softmax 12개가 QLinearSoftmax에서 float Softmax로 바뀌었다 (softmax 입력이나 출력 쪽 행렬곱이 float가 되면 ORT가 softmax를 정수로 합치지 않음). 풀링 헤드 softmax 1개는 정수 그대로.

### 10-6. 해석 (10-4 규칙 그대로)
- R0 대비 고화질 Δtop-1 회복: **D1 +0.1%p, D3 +0.2%p** → 둘 다 +0.3%p 미만 → **"attention 행렬곱은 주원인 아님".** dev 전체 비교는 하지 않는다.
- cos도 거의 그대로다 (D1 +0.0008, D3 +0.0006). rothad 위에서도 같다 (cos +0.0012~+0.0013, Δtop-1은 −0.8~0.0으로 오히려 흔들림 범위 안).
- 즉 attention 확률·Q×Kᵀ를 FP32로 되돌려도 FP32 대비 cos 손실(약 0.045)의 2% 정도만 회복된다. 10-2 가설(attention 확률의 U8 눈금)은 이 표본에서 지지되지 않는다.
- 남은 후보 (확인 안 함, 기록만): ORT에서 정수로 도는 나머지 MatMul 53개 — QKV·proj·fc1·fc2(블록 5·9 제외)와 풀링 헤드. D-031은 QKV 정수 경로가 S8 대비 손실의 원인이라고 했지만, 회전으로 QKV 입력 채널 쏠림을 없앤 rothad도 FP32 대비 손실이 비슷했다(8절) → 채널 쏠림 아닌 다른 원인(토큰 쏠림에 따른 텐서별 눈금, fc1·fc2 활성값 등)이 남아 있을 수 있다.
- 지연은 재지 않았다.

### 10-7. 진단 2 (판정용 아님): 연산 묶음 하나씩만 FP32로 되돌리기 (설계, 10/18 — 실행 전에 고정)
- R0 조건 그대로(fc2L2 U8+rr, percentile 99.999, highres 256 s0, hc16), 기본 exclude `/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/`에 한 묶음씩만 추가:

| 이름 | 추가 exclude 패턴 | FP32로 남는 MatMul (기본 fc2 2개 포함) |
|---|---|---|
| E1 | `/attn_pool/` | 9 = 풀링 헤드 7 (q, kv, attention 2, proj, mlp fc1, mlp fc2) + 2 |
| E2 | `/attn/qkv/` | 14 = 블록 qkv 12 + 2 |
| E3 | `/attn/proj/` | 14 = 블록 attn proj 12 + 2 (풀링 헤드 `/attn_pool/proj/`는 안 걸림) |
| E4 | `/blocks.N/mlp/fc1/` × 12 (N = 0~11) | 14 = 블록 fc1 12 + 2 (`/mlp/fc1/`로 쓰면 풀링 헤드 fc1도 걸리므로 블록 번호를 넣음) |
| E5 | `/blocks.N/mlp/fc2/` × 10 (N ≠ 5, 9) | 12 = 블록 fc2 전부 |
- 노드 목록 `diag/group_exclude_nodes.csv`, 패턴 `diag/group_exclude_patterns.json` (양자화 전 `select_excluded` 시험 실행으로 확인, `/blocks.1/`이 `/blocks.10/`을 잡지 않음 확인).
- 측정: dev 6×200 스크리닝 (정수 커널, 4스레드), R0를 같은 실행에 넣는다. 표: 구성 / 정수 MatMul 수 / 고화질 cos / Δtop-1 / CIFAR A Δ.
- **해석 규칙:** R0 대비 **cos +0.0135 이상**(남은 손실 약 0.045의 30%) 이거나 **고화질 Δtop-1 +0.3%p 이상**인 묶음이 있으면 **"주원인 묶음"** → 그 묶음만 FP32로 둔 모델로 (a) dev 전체 짝 비교 1회 (R0 대비, `s2_paired_ci.py`) (b) 4스레드 지연 3세션 (`s2_bench_speed.py`, R0와 번갈아, 시작 전 `ps`로 다른 python·hailo 작업 없음 확인, 있으면 대기). 둘 이상이면 cos 회복이 가장 큰 하나. 없으면 **"특정 묶음에 몰리지 않음(손실이 고르게 퍼짐)"**으로 기록하고 끝.
