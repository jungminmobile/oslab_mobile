# SigLIP2 트랙 현황 (한 장 요약)

- 담당: 손준혁(쭈리) · 마지막 갱신: 2026-10-14
- 모델: `timm/ViT-B-16-SigLIP2`, 224×224, **이미지 인코더만** 양자화 (텍스트는 FP32로 분류기 1회 계산)
- 목표: RN50 트랙(정민)과 **같은 조건**으로 재서 "CNN과 Transformer는 양자화에 민감한 위치가 다르다"를 보인다
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
| G3 데스크탑 양자화·민감도 | 🔶 거의 완료 (지연 측정 대기) | `G3_sensitivity`, `G3_calib_method`, `G3_cumulative` |
| G4 Hailo DFC 혼합 정밀도 | 🔶 진행 중 | `G4_dfc_first` |
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

Hailo 공식 설정이 16비트로 두는 블록 5가 ORT 민감도 분석의 블록 5와 겹친다. 64장이라 C·B 순위는 참고 수준. 구성 D·E 진행 중.

## 4. 지키고 있는 규칙
- 정확도는 **정수 커널**(ORT 기본 설정)로만, fake-quant는 디버깅만 (PRD R5, D-008)
- test 분할은 아직 한 번도 안 씀 (G6에서 1회)
- 보정에 실제로 쓰인 장수를 assert (RN50 "256장 중 16장" 버그 재발 방지)
- 모든 결과에 커밋 해시·dirty·CPU 플래그 기록, 결과마다 3중 확인(자기 검증·독립 재계산·서브에이전트 검토)
- 이 PC(i7-1165G7, AVX512-VNNI)에서 잰 속도는 **비공식**. 공식 비교는 정민 노트북 스레드 4 (D-010)

## 5. @정민 공유 (RN50에도 해당하는 것)
1. **ORT 1.25 기본 Entropy 보정은 KL 탐색을 안 한다**: `num_bins=128`, `num_quantized_bins=128` 기본값이면 후보 임계값이 하나뿐이라 사실상 MinMax 범위. E2에서 Entropy를 비교하려면 `CalibNumBins`·`CalibNumQuantizedBins` 지정 필요해 보임
2. ORT Percentile·Entropy 보정기는 모든 배치의 중간 텐서를 메모리에 쌓은 뒤 계산한다 (큰 모델·많은 장수에서 메모리 부족)
3. S8 활성값이면 활성값끼리 곱하는 MatMul 일부가 정수로 융합되지 않는다(U8은 융합됨). PRD 3절 "x86에서 S8S8 = U8S8"은 conv 모델 확인이라 MatMul은 다를 수 있음
4. 이 PC도 VNNI가 있어서 비-VNNI 포화 재현은 불가
5. `rn50/README.md`의 Pets 저장 경로가 `datasets.py`가 찾는 경로와 다름, PRD 3절 "DFC 미설치" 문구가 오래됨 (PR #3)

## 6. 다음 할 일
- G3: 후보 지연 측정 (스레드 1·2·4, U8 포함) → PRD 속도 하한 확인
- G4: 구성 D·E 결과, 후보 조합 컴파일(HEF 크기·컨텍스트 수), 최종 후보로 정민 노트북 교차 확인 1회 (D-027)
- ORT 후보 ↔ DFC 구성 짝을 맞춘 **후보표** → 팀이 선택 (Claude는 선택하지 않음)
- G6: 최종 구성 동결, test 1회, 재현 테스트
