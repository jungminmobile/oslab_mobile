# G3 보정 방식 비교: MinMax / Percentile / Entropy (2026-10-13)

ROADMAP G3 "보정 방식 비교" (RN50 PRD E2와 같은 3종). MatMul만 INT8(잔차 Add·LayerNorm·GELU는 FP32) 구성 2개에서 비교.
스크리닝: dev 6종 각 앞 200장, **정수 커널**, 고화질 cos(주)·Δtop-1(참고) (D-024). 보정 highres 256장 시드 0. 이 PC.
**결론: 블록 5~11 fc2를 FP32로 두면 Percentile 99.999(비대칭)가 가장 좋다 — 고화질 cos 0.959, Δtop-1 +0.5%p, 139.0 MiB(−60.8%).
최선 방식은 구성에 따라 뒤집힌다: fc2까지 INT8이면 Percentile 99.99가 낫고(이상치를 잘라 fc2를 살림), fc2를 보호하면 99.99는 크게 나빠진다(의미 있는 큰 값까지 잘라냄).**

## 실행 명령
```bash
cd siglip2
RN50_RESULTS=../results/siglip2/G3_calib_method/hist_chunk_check "$SIGLIP2_PYTHON" s2_hist_chunk_check.py   # 나눠 누적 동등성
bash scripts/g3_calib.sh                                                                                   # 9개 구성 (커밋 e57f356)
```

## 메모리 문제와 "나눠서 누적" (`hist_chunk_check/`)
ORT 1.25 Percentile·Entropy 보정기는 모든 배치의 중간 텐서를 쌓은 뒤 히스토그램을 만들어 이 PC(16 GB)에서 실패했다(`bad allocation`).
`s2_quantize.py --hist-chunk 16`이 ORT의 히스토그램 합치기(`merge_histogram`)로 16장씩 누적한다(쭈리 승인 10/13). 32장에서 ORT 표준과 비교(판단 규칙 사전 고정: cos 차이 ≤ 0.002, scale 중앙값 상대 차이 ≤ 1%).
**주의 (독립 검토 지적): 이 확인은 `--chunk 8`(배치마다 누적, 32장 = 4번)로 했다. 실제 보정은 16장씩·256장(16번)이라, 쓰인 조건 그대로의 동등성은 아직 확인하지 않았다** → 16장 단위 재확인은 아래 "16장 단위 재확인" 절.

| 방법 | scale 상대 차이 (중앙값 / 최대 / 1% 초과 텐서) | zero point 다른 텐서 | 고화질 cos 표준 → 나눠 누적 | 판정 |
|---|---|---|---|---|
| Percentile 99.999 | 0.012% / 2.6% / 1개 (184개 중) | 7 | 0.6672 → 0.6669 (−0.0003) | **동등** |
| Entropy | 0% / 21.2% / 59개 | 63 | 0.6568 → 0.6598 (+0.0030) | 기준 초과 → "근사"로 표시(쭈리 결정) — **아래 정정 참고** |

**정정 (독립 검토, ORT 1.25 `calibrate.py` 확인): ORT 기본 Entropy는 `num_bins=128`, `num_quantized_bins=128`이라 KL 탐색 후보가 하나뿐(`range(64, 65)`)이고 임계값 = 전체 범위다. 즉 ORT 기본 Entropy는 사실상 MinMax와 같은 범위를 쓴다.**
나눠 누적하면 `merge_histogram`이 첫 묶음의 구간 폭을 유지하고 범위가 커진 텐서에만 구간을 덧붙여 구간 수가 128을 넘는다 → 그 텐서들에서만 실제 KL 탐색이 일어난다. 그래서 아래 "Entropy" 행은 ORT 기본 Entropy의 근사가 아니라 **다른 동작(부분적인 KL 탐색)**이다. Percentile(2048구간)은 구간 해상도만 달라진다.
이 점은 RN50 트랙의 Entropy 보정(PRD E2)에도 해당한다 — ORT 기본값 그대로면 Entropy ≈ MinMax 범위. 정민에게 공유한다.

### 16장 단위 재확인 (`hist_chunk_check16/`, 10/14 밤샘, 32장 = 16장 × 2번)
같은 판단 규칙(cos 차이 ≤ 0.002, scale 중앙값 상대 차이 ≤ 1%)으로 실제 보정과 같은 16장 단위를 확인했다.

| 방법 | scale 상대 차이 (중앙값 / 최대 / 1% 초과 텐서) | zero point 다른 텐서 | 고화질 cos 표준 → 나눠 누적 | 판정 |
|---|---|---|---|---|
| Percentile 99.999 | 0.001% / 2.6% / 1개 (184개 중) | 6 | 0.6672 → 0.6671 (−0.0001) | **동등** |
| Entropy | 0% / 19.4% / 37개 | 42 | 0.6568 → 0.6571 (+0.0003) | cos는 기준 안이다. 다만 scale이 1% 넘게 다른 텐서가 37개라, 위 정정대로 "다른 동작"이라는 표시를 유지한다 |

- 첫 실행(커밋 `706e473`)은 `dirty: true`로 기록됐다. 원인은 코드 변경이 아니라 `siglip2/`에 남아 있던 hailo 로그 파일이다. `scripts/overnight_win2.sh`로 깨끗한 상태에서 다시 돌려 값이 같은지 확인한 뒤 확정한다.
- 256장(16번 누적)은 표준 쪽이 메모리 때문에 만들어지지 않아 직접 비교할 수 없다. 32장에서 2번 누적한 결과가 동등했다. 그래서 Percentile은 누적 횟수가 늘어도 구간 해상도 차이만 생긴다고 보지만, 이것은 추론이다.

아래 표의 Percentile·Entropy는 모두 16장씩 나눠 누적한 값(`_hc16`)이다.

## 결과 (`screen.jsonl`, 커밋 `e57f356`, dirty=false)
| 구성 | 방법 | MiB | 고화질 cos | 고화질 Δtop-1 | CIFAR A cos | CIFAR A Δtop-1 |
|---|---|---|---|---|---|---|
| MatMul만 | MinMax | 91.82 | 0.6424 | −65.3 | 0.7623 | −61.5 |
| | Percentile 99.99 | 91.82 | **0.6834** | **−27.0** | 0.7149 | −27.0 |
| | Percentile 99.999 | 91.82 | 0.6609 | −62.5 | 0.7587 | −58.5 |
| | Entropy (나눠 누적, ORT 기본과 다른 동작) | 91.82 | 0.6457 | −64.8 | 0.7610 | −61.0 |
| MatMul만 − 블록 5~11 fc2 | MinMax | 139.03 | 0.9360 | −3.1 | 0.8700 | −17.0 |
| | Percentile 99.99 | 139.03 | 0.6987 | −24.1 | 0.7264 | −26.0 |
| | **Percentile 99.999** | 139.03 | **0.9586** | **+0.5** | 0.8801 | −6.0 |
| | Entropy (나눠 누적, ORT 기본과 다른 동작) | 139.03 | 0.9397 | −2.1 | 0.8761 | −13.0 |
| | MinMax, **U8** 활성값 (D-023) | 139.03 | 0.9365 | −3.1 | 0.8698 | −14.5 |

MinMax·MatMul만은 12:24(`882fd7b`)와 16:47(`e57f356`) 두 번 재서 같은 값(다시 만든 모델도 같음).

## 정수로 실행된 노드 (`quantize_log.jsonl`의 `ort_optimized_graph`, VERIFICATION D-G3)
| 구성 | 정수 커널 | float으로 남은 계산 |
|---|---|---|
| MatMul만, S8 | QLinearMatMul 66, QLinearSoftmax 13 | MatMul 13, LayerNormalization 25 |
| − 블록 5~11 fc2, S8 | QLinearMatMul 59, QLinearSoftmax 13 | MatMul 20, LayerNormalization 25 |
| − 블록 5~11 fc2, **U8** | **QLinearMatMul 72**, QLinearSoftmax 13 | **MatMul 7**, LayerNormalization 25 |

- `--op-types MatMul`인데도 Softmax가 정수(QLinearSoftmax)로 실행된다: 앞뒤 MatMul의 Q/DQ가 Softmax를 감싸 ORT가 융합한 것.
- **S8 활성값에서는 활성값끼리 곱하는 attention MatMul 26개(QKᵀ·AV, 블록 12 + head 1 × 2) 중 13개(attention 모듈마다 1개)가 정수로 융합되지 않는다. U8에서는 정수로 실행된다.** 정확도는 같은 수준(cos 0.9360 vs 0.9365)이라, U8이 더 빠를 수 있다 → 지연 비교 후 D-023(S8 기본) 재검토를 제안할 예정. PRD 3절의 "x86에서 S8S8 = U8S8"은 conv 모델 확인이라 이 경우와 다르다.

## 해석
- fc2(블록 5~11)가 INT8이면 어떤 방법도 cos 0.68 이하. **방법 선택보다 혼합 정밀도(어디를 FP32로 둘지)가 훨씬 크다.**
- fc2를 보호하면 99.999처럼 아주 조금만 자르는 것이 최선. 99.99는 남은 텐서(LayerNorm 뒤 등)의 의미 있는 큰 값을 잘라 무너진다 → "massive activation은 잘라내면 안 된다"는 해석과 맞다.
- "Entropy"(나눠 누적)는 MinMax보다 조금 낫지만 99.999보다 낮다. ORT 기본 Entropy와는 다른 동작이라 RN50 Entropy 결과와 직접 비교하지 않는다.

## 3중 확인
1. 자기 검증: 9개 모두 보정 256장(`calib_seen`), dirty=false. 나눠 누적의 동등성을 사전 규칙으로 판정
2. 독립 재계산: 같은 구성을 두 번 재서 같은 값. 정수 노드 목록을 ORT 최적화 그래프로 따로 확인
3. 독립 검토: 서브에이전트 1회 — 수치 전부 재계산 일치. 지적(동등성 확인이 chunk 8이었음, ORT 기본 Entropy = 사실상 전체 범위, attention MatMul 개수 표현, `hist_chunk_check.jsonl` 커밋 필드는 실행 시작 커밋이고 모델 생성 커밋과 다름(코드 동일), "bad allocation" 로그 미보관) 반영

## 다음
- 누적 구성(`../G3_cumulative`): 보호하는 fc2를 7 → 4 → 2개로 줄여 크기 곡선, 지연(스레드 1·2·4, U8 변형 포함)
- 상위 후보는 dev 전체로 재측정
