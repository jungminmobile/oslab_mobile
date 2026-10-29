# SigLIP2 트랙 현황 (한 장 요약)

- 담당: 손준혁(쭈리) · 마지막 갱신: 2026-10-15 01시
- 모델: `timm/ViT-B-16-SigLIP2`, 224×224, **이미지 인코더만** 양자화 (텍스트는 FP32로 분류기 1회 계산)
- 목표: SigLIP2 하나를 끝까지 최적화한다. 예선은 모델 하나만 제출하므로 RN50과 **같은 조건**으로 재서 더 좋은 쪽을 고른다 (D-034, 10/15 — 이전 목표 "CNN vs Transformer 비교"는 대체됨)
- 이 문서는 요약이다. 숫자의 근거는 각 결과 폴더의 `README.md`와 원본 jsonl에 있다.

## 1. 어디에 있나 (RN50과 겹치지 않게 분리)

| 구분 | 위치 |
|---|---|
| 코드·문서 | `siglip2/` 폴더만 (RN50 `rn50/`, `PRD.md`, `hailo/`, 루트 문서는 **수정하지 않음**) |
| 결과 | `results/siglip2/<단계>_<주제>/README.md` |
| 브랜치 | `siglip2/g0-handoff` → `g0-env` → `g1-onnx-export` → `g2-fp32-baseline` → `g3-sensitivity` (단계별로 쌓음) |
| PR | [#3](https://github.com/jungminmobile/oslab_mobile/pull/3) G0 문서 → [#4](https://github.com/jungminmobile/oslab_mobile/pull/4) 환경 → [#5](https://github.com/jungminmobile/oslab_mobile/pull/5) ONNX·INT8 → [#6](https://github.com/jungminmobile/oslab_mobile/pull/6) FP32 기준치 → [#7](https://github.com/jungminmobile/oslab_mobile/pull/7) 민감도·Hailo. **main 병합은 아직 안 함** (#3만 main을 향함) |
| 결정 기록 | [`siglip2/DECISIONS.md`](DECISIONS.md) (D-001~D-027) · 진행 로그 [`siglip2/PROGRESS.md`](PROGRESS.md) · 계획 [`siglip2/ROADMAP.md`](ROADMAP.md) |
| 대용량 파일 | 저장소에 없음. `D:\mokerton\artifacts\siglip2\`(ONNX·HEF·캐시), 경로와 sha256만 결과 폴더에 기록 |

RN50 공통 코드(데이터 선택 해시, CIFAR 구간, 프롬프트, 평가, 속도 측정)는 **복사하지 않고 import**해서 같은 조건을 맞췄다.
dev 사진 장수가 RN50 기록과 데이터셋마다 같다(Pets 990, Flowers 957, Cars 967, Food 985, Caltech 1024, CIFAR A·B 1000) → 같은 사진.

## 2. 단계 현황 (ROADMAP G0~G7)

| 단계 | 상태 | 결과 폴더 |
|---|---|---|
| G0 준비 | ✅ 완료 | `results/siglip2/G0_smoke` |
| G1 컴파일 게이트 | ✅ **SigLIP2 계속** (판정 10/13, D-026) | `G1_onnx_export`, `G1_int8_minmax_all`, `G1_dfc_gate` |
| G2 FP32 기준치 | ✅ 완료 (지연은 이 PC 비공식) | `G2_fp32_baseline`, `G2_fp32_dev`, `G2_prompt_compare` |
| G3 데스크탑 양자화·민감도 | 🔶 거의 완료 (지연 재측정 중, 원인 분석 완료) | `G3_sensitivity`, `G3_calib_method`, `G3_cumulative`, `G3_why` |
| G4 Hailo DFC 혼합 정밀도 | 🔶 진행 중 (C·D·E 스크리닝, D 컴파일 완료) | `G4_dfc_first`, `G4_dfc_screen`, `G4_compile`, `G4_noise` |
| G5 실칩 | ⏸ 본선(Pi 5 + Hailo-10H)으로 이동 (D-015) | |
| G6 동결·test 1회 / G7 제출 | ⏳ 예정 | |

## 3. 핵심 결과

### 3-1. FP32 기준치 (결과표 S-A행, `G2_fp32_baseline`)
| 파라미터 | GMACs | ONNX 크기 | peak 메모리 | 지연 1/2/4스레드 (비공식) | 고화질 5종 dev 평균 top-1 | CIFAR A |
|---|---|---|---|---|---|---|
| 92.9M | 17.7 | 354.45 MiB | 433 MiB | 403 / 348 / 265 ms | **91.66** | 76.90 |

프롬프트는 RN50과 같은 CLIP 공식 템플릿 (SigLIP 권장 "This is a photo of {}."보다 유의하게 좋음, D-025).

### 3-2. 전부 INT8로 하면 무너진다 (`G1_int8_minmax_all`)
ORT 정수 커널, MinMax, 전부 INT8: 고화질 top-1 **91.66 → 1.37**. 정수 커널 문제가 아니라(fake-quant도 같음) **활성값 이상치** 때문:
블록 5부터 잔차 흐름 최대값이 99.9% 분위수의 39~68배, MLP fc2 입력은 블록 5·9에서 124배·286배.

### 3-3. 민감도 지도 — 하나만 INT8 (`G3_sensitivity`, dev 각 200장, 고화질 cos)
| 위치 | cos | 해석 |
|---|---|---|
| 블록 1~4, QKV, attention 행렬곱 | 0.989 이상 | 거의 무손실 |
| patch, 블록 0, head, proj, fc1 | 0.91~0.98 | 작은 손실 |
| **블록 5~11 (하나만 INT8이어도)** | **0.41~0.51** | 붕괴 |
| **MLP fc2 (전 블록)** | **0.65** | 붕괴 |

**CNN vs Transformer:** RN50은 앞쪽(stem·layer1)이 민감, SigLIP2는 **뒤쪽 블록의 잔차 흐름과 fc2**가 민감 — 위치가 반대다.

### 3-4. 데스크탑(ORT) 후보 (`G3_calib_method`, `G3_cumulative`)
공통: **MatMul만 INT8**(잔차 Add·LayerNorm·GELU는 FP32), Percentile 99.999 보정(비대칭), highres 256장, 정수 커널.
dev 전체, 짝 부트스트랩 95% 구간(PRD 5절):

| 후보 | FP32로 둔 fc2 | 크기 (FP32 대비) | 고화질 cos | 고화질 Δtop-1 [95% 구간] |
|---|---|---|---|---|
| fc2L7 | 블록 5~11 | 139.0 MiB (−60.8%) | 0.958 | −0.46 [−0.91, +0.03] |
| fc2L4 | 블록 5·9·10·11 | 118.8 MiB (−66.5%) | 0.958 | −0.34 [−0.82, +0.16] |
| **fc2L2** | **블록 5·9** | **105.3 MiB (−70.3%)** | 0.956 | −0.50 [−1.01, +0.01] |
| fc2L2 (U8 활성값) | 블록 5·9 | 105.3 MiB | 0.956 | −0.91 [−1.40, −0.41] (유의한 손실) |

- S8 세 후보는 서로 **동률**(유의한 차이 없음). PRD 규칙(동률이면 cos)으로는 fc2L7, 크기로는 fc2L2가 앞섬 → **최종 선택은 팀이 후보표로**
- 보정 방식: fc2를 보호하면 Percentile 99.999 > Entropy > MinMax ≫ Percentile 99.99 (조금만 잘라야 함)
- 지연(PRD 속도 하한 확인)은 PC가 한가할 때 자동 측정 예정

### 3-5. Hailo-10H (DFC 5.4.0, `G1_dfc_gate`, `G4_dfc_first`)
- **전 구간 컴파일 성공**: HEF 69.2 MB, 컨텍스트 18개, 호스트로 떨어진 구간 없음. DFC float 그래프 = ONNX (cos 1.0)
- 8비트 에뮬레이션 (Pets 64장, FP32 95.31):

| 구성 | 16비트로 둔 층 | cos | top-1 |
|---|---|---|---|
| 기본 | 없음 | 0.12 | 3.1 |
| A | 잔차 합 + 패치 | 0.72 | 40.6 |
| C (Hailo 공식 SigLIP 패턴) | A + 블록 0·5 MLP | **0.963** | 92.2 |
| B | A + 블록 5~11 fc2 | 0.936 | 93.8 |

Hailo 공식 설정이 16비트로 두는 블록 5가 ORT 민감도 분석의 블록 5와 겹친다.

**확대 평가 (dev 6종 × 200장, ORT와 같은 사진으로 짝 비교, `G4_dfc_screen`)**:

| 구성 | 고화질 cos | Δtop-1 [95% 구간] |
|---|---|---|
| ORT fc2L2 (참고) | 0.956 | +0.3 [−0.6, +1.2] |
| D = C + 블록 9 fc2 | 0.952 | −1.1 [−2.3, 0.0] |
| C 공식 패턴 | 0.949 | −1.4 [−2.5, −0.1] |
| E 블록 0·5·9 fc2만 | 0.919 | −2.3 [−3.7, −0.9] |

C와 D는 동률이고, 둘 다 ORT보다 1.4~1.7%p 유의하게 낮다. **D 컴파일 성공: HEF 99.8 MB, 컨텍스트 17개** (`G4_compile`). DFC 층별 잡음 분석은 DFC 내부 오류로 2회 실패해 보류했다 (`G4_noise`).

### 3-6. 왜 SigLIP2는 블록 5부터 무너지나 (`G3_why`, 서브에이전트 독립 검토 2회)
- 블록 5·9의 **MLP가 고정 위치 토큰(패치 55, 블록 9부터 패치 13)의 몇 개 채널(67·301·145, 483)에 거대 값(최대 수백)을 써 넣는다**. 다른 토큰들은 attention의 27~52%를 그 토큰에 몰아준다 (attention sink / register). 데이터셋 6종 모두에서 같다.
- 지우면 cos 0.50 → 모델이 쓰는 기능이다.
- 텐서 하나에 눈금 하나인 8비트에서 이 값이 눈금을 22~78배 넓혀 **보통 토큰의 작은 값이 0으로 뭉개진다** (블록 5는 GELU 음수 꼬리, 블록 9는 fc2 출력). 그래서 "블록 5·9 fc2만 FP32"가 통한다. 보정 방식(MinMax·Percentile)은 이 두 층에서 거의 차이가 없다.
- RN50(CNN)은 앞쪽이 민감하다. SigLIP2는 **거대 값을 처음 써 넣는 중간 MLP와 그 뒤 잔차 흐름**이 민감하다.

### 3-7. 10/14 오후~밤 추가 결과
- **지연** (비공식, 고성능 모드): U8 fc2L2는 두 측정 모두 모든 스레드에서 FP32의 약 절반이다 (4스레드 77~107 ms 대 148~211 ms). S8 대 FP32의 4스레드 순위는 측정마다 바뀌었다 (+2.5% / −10.6%) → 공식 판정은 정민 노트북으로.
- **S8이 4스레드에서 느려지는 원인** (ORT 연산별 프로파일): S8에서는 QKV 투영 MatMul 12개 + 헤드 kv가 정수로 합쳐지지 않는다. 그 양자화·역양자화가 스레드를 늘려도 줄지 않는다 (4스레드 시간의 36%).
- **S8과 U8의 정확도 차이(−0.41%p)는 전부 QKV 정수 경로에서 나온다.** QKV만 U8로 둔 모델은 U8과 예측이 5,923장 모두 같다. 대책(reduce_range·U8U8) 실험 중.
- **DFC 보정 256장 구성 D**: cos 0.955, −0.2 [−1.4, +1.0]으로 ORT와 동률 (64장 D는 1.4%p 낮았음).
- **원인 분석 3단계**: 토큰별·채널별 눈금이면 fc2를 전부 8비트로 해도 cos 0.999.
- **새 venv 재현**: export·양자화 sha256과 dev 전체 예측이 비트 단위로 같다 (`results/siglip2/G3_repro`).
- **후보 peak 메모리**: fc2L2 228 MiB (FP32 433 대비 −47%).

## 4. 지키고 있는 규칙
- 정확도는 **정수 커널**(ORT 기본 설정)로만, fake-quant는 디버깅만 (PRD R5, D-008)
- test 분할은 아직 한 번도 안 씀 (G6에서 1회)
- 보정에 실제로 쓰인 장수를 assert (RN50 "256장 중 16장" 버그 재발 방지)
- 모든 결과에 커밋 해시·dirty·CPU 플래그 기록, 결과마다 3중 확인(자기 검증·독립 재계산·서브에이전트 검토)
- 이 PC(i7-1165G7, AVX512-VNNI)에서 잰 속도는 **비공식**. 공식 비교는 정민 노트북 스레드 4 (D-010)

## 5. @정민 공유 (RN50에도 해당하는 것)
1. **ORT 1.25 기본 Entropy 보정은 KL 탐색을 안 한다**: `num_bins=128`, `num_quantized_bins=128` 기본값이면 후보 임계값이 하나뿐이라 사실상 MinMax 범위. E2에서 Entropy를 비교하려면 `CalibNumBins`·`CalibNumQuantizedBins` 지정 필요해 보임
2. ORT Percentile·Entropy 보정기는 모든 배치의 중간 텐서를 메모리에 쌓은 뒤 계산한다 (큰 모델·많은 장수에서 메모리 부족)
3. S8 활성값이면 MatMul 일부(SigLIP2에서는 QKV 투영 12개 + 헤드 kv 1개, 10/14 정정)가 정수로 융합되지 않는다(U8은 융합됨). PRD 3절 "x86에서 S8S8 = U8S8"은 conv 모델 확인이라 MatMul은 다를 수 있음
4. 이 PC도 VNNI가 있어서 비-VNNI 포화 재현은 불가
5. `rn50/README.md`의 Pets 저장 경로가 `datasets.py`가 찾는 경로와 다름, PRD 3절 "DFC 미설치" 문구가 오래됨 (PR #3)
6. (Windows) Python의 `subprocess`로 그냥 `bash`를 실행하면 `C:/Windows/System32/bash.exe`(WSL)가 잡혀 Windows 환경 변수가 전달되지 않는다 → Git Bash 전체 경로를 쓴다
7. WSL에서 `git status`로 dirty를 판정하면 Windows가 CRLF로 체크아웃한 파일을 수정으로 본다 → `git -c core.autocrlf=true`

- (10/17) **ImageNetV2 클래스 이름 동점**: `open_clip.zero_shot_metadata.IMAGENET_CLASSNAMES`에 같은 이름이 두 쌍 있다 ("missile" 657·744, "sunglasses" 836·837). 텍스트 벡터가 같아 점수가 정확히 동점이 되고, `topk_metrics`(argsort)와 argmax의 top-1이 0.05%p 안팎 다를 수 있다 (SigLIP2 test에서 5장 차이). RN50 test도 같은 목록을 쓴다 → 보고서에 각주 권장 (`results/siglip2/G6_test/README.md`)

## 6. 다음 할 일
- G3: 후보 지연 재측정 (스레드 1·2·4, U8 포함, 첫 측정은 다른 작업과 겹쳐 오염) → PRD 속도 하한 확인
- **후보표 초안**: `results/siglip2/candidates/README.md` → 지연 채운 뒤 팀이 선택 (Claude는 선택하지 않음)
- G4: 최종 `.alls`(C 또는 D) 결정 후 정민 노트북 교차 확인 1회 (D-027), 필요하면 E 컴파일
- G6: 최종 구성 동결, test 1회, 재현 테스트
