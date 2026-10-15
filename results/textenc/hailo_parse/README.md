# 텍스트 인코더 HEF 시험 — 블록 12개만 NPU (2026-10-16, 연구실 데스크탑)

환경: DESKTOP-3 WSL Ubuntu-22.04, Hailo DFC 5.4.0, `--hw-arch hailo10h`. GPU를 숨기고 CPU로 돌려서 **최적화 레벨 0**이다.
작업 폴더는 데스크탑 `~/textenc_dfc/20261016_033709`. 아래 숫자는 사용자가 붙여 준 로그(`steps.txt`, `text_q.log`)에서 옮겼다.

구성: 토큰 임베딩 조회와 위치 임베딩 더하기, 위치 63 선택, projection은 호스트가 하고, 블록 12개 + ln_final은 NPU가 한다.
입력 ONNX는 `te_export_blocks.py`가 만든 것이다(attention을 고정 shape 연산으로 다시 씀, PyTorch 대비 cos 0.9999998, sha256 a76fa59c…).

## 1. 올라가는가 — 예

| 단계 | 결과 |
|---|---|
| parse | 성공 (16초) |
| optimize (보정 문장 64개, 3차원 입력) | 실패: Slice 층 `IndexError`. 보정 입력은 NHWC **(64,1,64,768)** 이어야 한다 |
| optimize (무작위 보정) | 성공 (195초) |
| compile | **성공** (906초), HEF **67 MB**, 컨텍스트 11개 이상 |

HAR 층 이름: 184층. 블록 0 c_fc는 `conv3`(GELU까지 합쳐짐), 블록 0 c_proj는 `conv4`다.

## 2. 정확도 — DFC 에뮬레이션 (`textenc/hailo/run_text_q.sh`, 보정 64문장, 레벨 0)

Pets dev 990장 / Food dev 985장. 이미지는 FP32 임베딩 그대로 두고 텍스트만 바꿨다. 기준은 FP32 ONNX 텍스트 분류기(93.33 / 89.44)다.

| 구성 | 16비트 층 | 클래스 cos 최소 / 평균 | 확인용 문장 cos | Pets Δ | Food Δ | 평균 Δtop-1 |
|---|---|---|---|---|---|---|
| fp_optimized (변환만) | — | 0.99998 / 0.99999 | 0.99999 | 0.0 | +0.1 | +0.05 |
| t1_cproj0 | 블록 0 c_proj | 0.598 / 0.751 | 0.709 | −90.5 | −88.6 | **−89.6 (붕괴)** |
| t2_cproj0_ewadd | + 잔차 합 전부 | 0.556 / 0.927 | 0.890 | −9.9 | −6.1 | **−8.0** |

- 변환 자체는 정확하다(fp_optimized cos 0.99998).
- **CPU(ORT)와 달리 블록 0 c_proj 하나만 보호해서는 안 된다.** Hailo는 잔차 합(Add)도 8비트로 바꾸는데, 잔차 흐름 채널 556에 거대 값(약 434)이 실려 있어서 잔차 합을 16비트로 두지 않으면 무너진다.
- 잔차 합까지 16비트로 둬도 −8%p다. ORT의 C+D(−0.25%p)와는 차이가 크다.
- 처음 정한 판단 기준(cos ≥ 0.99, Δ ≥ −1.0%p)을 넘는 구성은 아직 없다.
- t0_base(전부 8비트) 줄은 이 요약에 옮기지 않았다(로그 앞부분에 있음).

## 3. 추가 시험 (같은 날, `t3_cproj0_ewadd_ln`, `t4_all16`)

| 구성 | 결과 |
|---|---|
| t3: t2 + `{layer_normalization*}` 16비트 | optimize 성공. 그러나 **t2와 모든 숫자가 끝자리까지 같다** (Δ −8.0) → 설정이 적용되지 않았다 |
| t4: conv·ew_add·LayerNorm·normalization·matmul·softmax 전부 16비트 | optimize **실패**: 자동으로 끼워진 `precision_change12` 층(`a16_w16_a8`)이 hailo10h에서 지원되지 않음 |

t3가 효과 없던 이유는 로그와 HAR 비교로 확인했다.
- optimize 순서는 `Mixed Precision`(16비트 지정)이 먼저이고 그다음이 `LayerNorm Decomposition`이다.
- optimize 뒤 HAR은 184층에서 321층으로 늘었다. `layer_normalization` 25개와 `softmax` 12개는 사라지고, reduce_mean 25·ew_sub 37·reduce_sum 37·ew_mult 37·feature_multiplier 25·activation 26·reduce_max 12 등으로 쪼개졌다.
- 원래 LayerNorm 층에 붙인 16비트 지정은 쪼개진 층으로 옮겨지지 않는다.

## 4. 결론 (2026-10-16)

- **텍스트 인코더 블록 12개는 Hailo-10H HEF로 컴파일된다**(67 MB). 토큰 임베딩·위치 63 선택·projection은 호스트가 맡는다.
- **최적화 레벨 0(GPU 없음)에서 가장 좋은 구성은 Δtop-1 −8.0%p**(블록 0 c_proj + 잔차 합 16비트)다. 판단 기준(−1.0%p)에 못 미친다.
- 원인은 CPU 스모크와 같다. 블록 0이 잔차 흐름 채널 556에 써 넣는 거대 값(약 434)이다. ORT에서는 잔차 Add·LayerNorm을 FP32로 둘 수 있었지만, Hailo에서는 이 경로가 모두 양자화된다. LayerNorm은 쪼개진 뒤에 정밀도를 지정할 방법을 아직 찾지 못했다.
- 배포 때 텍스트 인코더는 쓰지 않으므로(클래스 임베딩 미리 계산), **텍스트는 FP32 유지**로 둔다.

다시 열 때 해 볼 것 (선택):
1. 쪼개진 LayerNorm 층(ew_sub 등)의 이름으로 16비트를 직접 지정한다. optimize 뒤 HAR의 층 이름이 필요하다.
2. 데스크탑에 CUDA를 설치해 최적화 레벨 2 이상(QFT)으로 t2를 다시 돌린다.
