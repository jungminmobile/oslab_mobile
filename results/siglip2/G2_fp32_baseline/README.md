# G2 FP32 기준치 — 결과표 S-A행 (2026-10-13)

SigLIP2 이미지 인코더 FP32의 정확도·크기·메모리·지연. 모든 감소율·Δ의 기준점.
**속도는 이 PC(i7-1165G7) 값이라 비공식**이다 (D-010). 공식 비교는 정민 노트북 스레드 4로 다시 잰다.
test 분할은 아직 쓰지 않았다 (ROADMAP G2의 "test 기준치 1회"는 최종 후보와 함께 G6에서 한 번에 하기로 미룸, D-025).

## 실행 명령
```bash
cd siglip2
bash scripts/g2_pipeline.sh     # 커밋 27ccbb4, 모든 단계 dirty=false
```

## S-A행 (FP32)
| 항목 | 값 | 원본 |
|---|---|---|
| 모델 | `timm/ViT-B-16-SigLIP2` 이미지 인코더, 224×224, 임베딩 768 | |
| 파라미터 | **92,884,224** (PyTorch 순회 = ONNX initializer 합, 두 방법 일치) | `model_stats.jsonl` |
| GMACs | **17.71** (구조식) / 17.73 (fvcore, LayerNorm 0.019 포함 — 차이 전부 설명됨) | `model_stats.jsonl` |
| ONNX 크기 | **354.45 MiB** (`siglip2_image_fp32.onnx`) | |
| peak 메모리 (새 프로세스, 4스레드) | 세션 생성 416.8 MiB / 추론 중 **432.7 MiB** | `model_stats.jsonl` |
| 지연 1 / 2 / 4스레드 (비공식) | 403.3 ± 6.7 / 348.1 ± 10.5 / **264.7 ± 10.0** ms | `speed/speed_*.json` |
| 고화질 5종 dev 평균 top-1 | **91.66** | `../G2_fp32_dev/evaluate.jsonl` |

### dev 정확도 (FP32 ONNX, ORT 기본 설정, CLIP 공식 템플릿 = RN50과 같음)
| dev | n | top-1 | top-5 |
|---|---|---|---|
| CIFAR A | 1000 | 76.90 | 94.40 |
| CIFAR B (최종 후보 확인용) | 1000 | 77.40 | 94.60 |
| Pets | 990 | 93.33 | 99.60 |
| Flowers | 957 | 86.73 | 95.19 |
| Cars | 967 | 93.80 | 99.90 |
| Food | 985 | 89.44 | 98.17 |
| Caltech | 1024 | 95.02 | 99.32 |

dev 장수가 RN50 트랙 기록(`results/E0_measure/2026-10-12_repro_check`)과 데이터셋마다 같다 → 같은 해시 규칙으로 같은 사진.

### 참고: INT8 전부(MinMax) — S-B0 후보 (정확도 붕괴, `../G1_int8_minmax_all`)
| 항목 | FP32 | INT8 전부 | 변화 |
|---|---|---|---|
| 크기 | 354.45 MiB | 89.59 MiB | **−74.7%** |
| peak 메모리 (추론) | 432.7 MiB | 221.9 MiB | −48.7% |
| 지연 1 / 2 / 4스레드 (비공식) | 403.3 / 348.1 / 264.7 ms | 317.5 / 298.7 / 261.6 ms | −21% / −14% / −1.2% (4스레드는 표준편차보다 작은 차이) |
| 고화질 dev top-1 | 91.66 | 1.37 | −90.29%p |

4스레드에서 속도 이득이 거의 없다: 정수로 계산되는 것은 MatMul 66개·Softmax 13개뿐이고, LayerNorm·GELU·잔차 Add 앞뒤의 Q/DQ 변환(585개)이 비용을 더한다는 가설. G3 혼합 정밀도에서 지연 곡선을 따로 잰다.

## 프롬프트 템플릿 결정 (`../G2_prompt_compare`)
판단 규칙(실행 전에 정함, `s2_prompt_compare.py` 머리말): 고화질 평균 차이의 짝 부트스트랩 95% 구간이 0을 포함하면 RN50과 같은 CLIP 템플릿 유지, siglip이 유의하게 좋으면 사람에게 제안.

| dev | CLIP 공식 (RN50과 같음) | SigLIP "This is a photo of {}." |
|---|---|---|
| CIFAR A | 76.90 | 73.40 |
| Pets | 93.33 | 90.20 |
| Flowers | 86.73 | 84.54 |
| Cars | 93.80 | 92.97 |
| Food | 89.44 | 88.32 |
| Caltech | 95.02 | 94.63 |
| **고화질 평균** | **91.66** | 90.13 (차이 −1.53, 95% 구간 [−2.06, −0.98]) |

→ **CLIP 공식 템플릿이 유의하게 좋다. RN50과 같은 템플릿을 유지한다** (평가 프로토콜 변경 없음, D-025).

## 공개 보고치와 비교 (VERIFICATION B 2단계, 부분 확인)
open_clip 공개 결과표(`docs/openclip_classification_results.csv`, 받아 둔 파일 `$SIGLIP2_ARTIFACTS/logs/`)에 **SigLIP2 행은 없다**. 가장 가까운 SigLIP(1세대) B/16 224, test 분할:
| | Caltech | CIFAR-100 | Food | Flowers | Pets | Cars |
|---|---|---|---|---|---|---|
| SigLIP B/16 224 (공개, test) | 95.2 | 72.2 | 91.6 | 85.5 | 94.2 | 90.8 |
| SigLIP2 B/16 224 (우리, dev) | 95.0 | 76.9 | 89.4 | 86.7 | 93.3 | 93.8 |
모델 세대·분할이 달라 "1%p 이내" 기준을 그대로 적용할 수 없다. 6종 모두 수 %p 안이고, 프롬프트·클래스 이름·전처리가 같은 경로인 RN50 트랙은 CLIP 논문과 일치를 확인했다(`docs/CODE_REVIEW.md`). Hailo 표의 SigLIP B/16 float CIFAR-100(72.2)은 open_clip 값과 같다.

## 3중 확인
1. 자기 검증: dev 장수·클래스 수 assert, 전처리 범위 [−1, 1] assert. 프롬프트 비교·평가·모델 지표·지연 모두 커밋 `27ccbb4`, dirty=false
2. 독립 재계산:
   - 파라미터 수 PyTorch vs ONNX 일치, GMACs fvcore vs 구조식 차이 0.1%(LayerNorm 몫으로 정확히 설명)
   - **캐시 결정성**: pets dev 캐시를 다른 폴더에 다시 만들어 6개 파일 sha256 모두 일치 (`../G2_fp32_dev/cache_check/sha256.txt`) → 커밋 전에 만든 캐시도 커밋된 코드의 산출물과 같음
   - FP32 dev top-1: 캐시 생성 때 값(`../G2_fp32_dev/baselines.jsonl`)과 `rn50/evaluate.py` 경로 값(`evaluate.jsonl`)이 7종 모두 같음
3. 독립 검토: PR에서 서브에이전트 검토 예정

## 이상치·주의
- 2스레드가 1스레드보다 13%만 빠르다(FP32 403 → 348 ms). 4코어 노트북의 터보 클럭·전력 제한 영향 추정. 전원 연결 100%(`speed/runs.jsonl`). 공식 수치는 정민 노트북에서.
- 지연 측정 입력은 `rn50/bench_speed.py` 그대로(CIFAR dev A 앞 200장, CLIP 전처리) — 크기가 같고 값은 속도에 영향이 없어 그대로 사용.
