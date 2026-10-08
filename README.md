# CLIP 엣지 양자화 — Raspberry Pi 5 + Hailo-10H

2026 모바일공학 추계 경진대회 **트랙 2 (임베디드 환경 온디바이스 AI 최적화), 자유주제**.
CLIP 이미지 인코더를 양자화해 Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H)에서 빠르고 작게 돌리면서
제로샷 분류 정확도 손실을 최소화한다.

| 담당 | 모델 | 폴더 |
|---|---|---|
| 정민 | CLIP-RN50 (+ 상위 모델 RN50x4) | `rn50/` |
| 준혁 | SigLIP2 ViT-B/16 또는 ViT-B/32 | `siglip2/` |

## 폴더 구조

```
oslab_mobile/
├── README.md                 ← 지금 이 파일 (프로젝트 개요·구조)
├── .gitignore                ← 데이터·가중치·onnx 등 큰 파일 제외
│
├── docs/                     ← 실험 설계·검토 문서
│   ├── 실험계획표PRD_2026-10-07.md   야간 실험 계획·결과·결론 (R0~R8, X0~X8)
│   └── CODE_REVIEW.md                코드 검토 1차(10/6)·2차(10/7), 보정 버그 기록
│
├── rn50/                     ← CLIP-RN50 / RN50x4 실험 코드 (예선: 데스크톱 CPU)
│   ├── README.md             실행 순서·명령어 모음
│   ├── requirements.txt      버전 고정 의존성 (Python 3.11, CPU)
│   ├── common.py             공통 설정: 경로·전처리·데이터 구간·ONNX 세션·지표
│   ├── prompts/              프롬프트 템플릿 (CIFAR-100 18개)
│   │
│   │   [1. 준비]
│   ├── baseline_rn50.py      PyTorch 원본 FP32 베이스라인
│   ├── export_onnx.py        PyTorch → ONNX (+ AttnPool q1 최적화, 동일성 검증)
│   ├── datasets.py           dev(train 분할, 해시 선택)·보정 후보 생성 / test 최종 평가
│   │
│   │   [2. 양자화]
│   ├── quantize_onnx.py      ONNX Runtime 정적 INT8/W4A8 양자화 (층 선택·보정 출처)
│   ├── make_stem_combos.py   stem 내부 조합 모델 생성
│   ├── clip_weights.py       4비트용 채널별 MSE 최적 가중치 잘라내기
│   │
│   │   [3. 측정·정리]
│   ├── evaluate.py           모델 × 데이터셋 top-1/top-5/FP32 대비 유사도
│   ├── bench_speed.py        속도 반복 측정
│   ├── summarize.py          evaluate.jsonl → 마크다운 표
│   │
│   │   [4. 자동 실행]
│   └── night_run.py          PRD 실험 전체를 순서대로 실행 (이어하기 지원)
│
├── results/                  ← 정리된 실험 결과 (날짜별)
│   └── 2026-10-07_night/     README.md 참고
│       ├── rn50/             evaluate.jsonl, baselines.jsonl, speed.json, tables/
│       └── rn50x4/           〃
│
├── siglip2/                  ← (준혁) SigLIP2 실험 코드 자리
└── hailo/                    ← (예정) Hailo DFC 컴파일·.hef·Pi 5 실측 스크립트
```

## 실험 규칙 (팀 공통으로 맞출 것)

- **데이터 구간**: 보정은 train 일부, 설정 비교(dev)는 train의 다른 부분(이미지 SHA-1 해시로 선택), test는 최종 1회만
- **보정**: 256장, 시드 0, 섞어서 사용
- **보고서 정확도**: ONNX Runtime 기본 설정(정수 커널) 값 + 측정 CPU 명시. fake-quant 값은 참고용
- 자세한 내용: `docs/실험계획표PRD_2026-10-07.md` 2절

## 현재까지 핵심 결과 (RN50, dev, 고화질 5종 평균)

| 모델 | 용량 MiB | 장당 ms (CPU 2코어) | top-1 변화 (FP32 대비) |
|---|---|---|---|
| RN50 FP32 | 146.1 | 54.5 | 0 |
| RN50 c3 (layer3~AttnPool INT8) | 41.5 (−72%) | 33.4 (−39%) | −0.72%p |
| RN50 c4 (layer2~AttnPool INT8) | 38.1 (−74%) | 25.5 (−53%) | −1.4 ~ −1.6%p |
| RN50x4 c5 (stem만 FP32, CIFAR 보정) | 84.9 | 70.1 | −0.54%p (RN50 FP32보다 +3~11%p) |

## 빠른 시작

```bash
cd rn50
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# 데이터·가중치 받기와 실행 순서는 rn50/README.md
```
