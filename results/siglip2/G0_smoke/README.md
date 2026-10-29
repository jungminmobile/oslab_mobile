# G0 스모크 테스트 (2026-10-13)

ROADMAP G0 완료 기준: "SigLIP2를 불러와 이미지 한 장의 임베딩이 나오고, PyTorch 기준 FP32 zero-shot이 CIFAR-100 dev 100장에서 돈다."
**정확도 비교용 수치가 아니다** (100장, 표본이 작음). 파이프라인 점검용이다.

## 실행 명령
```bash
cd siglip2
OPENBLAS_NUM_THREADS=1 "$SIGLIP2_PYTHON" scripts/g0_smoke.py --n 100 --threads 4 --out ../results/siglip2/G0_smoke
```

## 조건
| 항목 | 값 |
|---|---|
| 모델 | `hf-hub:timm/ViT-B-16-SigLIP2` (open_clip 3.3.0, timm 1.0.30, transformers 5.19.0) |
| 실행 | PyTorch FP32, CPU 4스레드 (이 PC, 비공식) |
| 사진 | CIFAR-100 train[45000:45100] = RN50과 같은 dev A 구간 앞 100장 |
| 프롬프트 | CIFAR-100 18개 템플릿 (`rn50/prompts`), 프롬프트 앙상블 |
| 전처리 | 모델 설정 그대로: 224×224 squash 리사이즈(자르기 없음), bicubic, 평균·표준편차 0.5 |
| 환경 | `env.json` (i7-1165G7, avx512_vnni ✅, avx_vnni ❌, ORT 1.25.0) |

## 결과 (`g0_smoke.jsonl` 두 번째 줄, 커밋 `1aed107`, dirty=false)
| 항목 | 값 |
|---|---|
| 이미지 임베딩 | (1, 768), 길이 1, 유한값 |
| 파라미터 | 이미지 인코더 92.88M / 전체 375.19M |
| CIFAR dev A 앞 100장 top-1 / top-5 | 76.0 / 95.0 |
| logit_scale / logit_bias | 112.67 / −16.77 |

## 3중 확인
1. 자기 검증: shape·길이·유한값·사진 수·클래스 수·템플릿 수 assert 통과. 같은 사진을 두 번 넣은 임베딩이 비트 단위로 같음. 같은 명령을 두 번 실행(커밋 전 `0be48db` dirty / 커밋 후 `1aed107`) → top-1·top-5 같음
2. 독립 재계산: top-1을 `topk_metrics`와 다른 방법(argmax 직접 비교)으로 다시 계산 → 같음. SigLIP 점수 sigmoid(scale·cos+bias)의 argmax = 코사인 argmax (siglip2/CLAUDE.md 4절 가정 확인)
3. 독립 검토: PR에서 서브에이전트 검토 예정

## 결론
G0 완료 기준 충족. 첫 실행은 `transformers` 미설치로 실패했고(D-022), 설치 후 통과.
