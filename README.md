# CLIP 이미지 인코더 양자화 — Raspberry Pi 5 + Hailo-10H

2026 모바일공학 하반기 공모전 **트랙 2 (임베디드 환경 On-device AI 최적화), 자유주제**.
CLIP 계열 모델의 **이미지 인코더**를 8비트 중심으로 양자화한다. 정확도·속도·용량·메모리의 트레이드오프 곡선을 만들고, 최종 지점은 팀이 상의해서 고른다.
예선은 로컬 PC에서 ONNX Runtime 정수 커널로 평가하고, 본선은 Hailo-10H(.hef)에서 실측한다.

> **현재 실험 계획은 [`PRD.md`](PRD.md)를 먼저 읽는다.** 이전 실험은 [`archive/`](archive/)에 참고용으로만 있다.

| 담당 | 계열 | 모델 | 폴더 |
|---|---|---|---|
| 정민 | CNN | CLIP-RN50 / RN50x4 | `rn50/` |
| 준혁 | 트랜스포머 | SigLIP2 (안 되면 CLIP ViT) | `siglip2/` |

## 폴더 구조

```
oslab_mobile/
├── PRD.md              ★ 현재 실험 계획서 (목표·요구사항·기술 스택·실험 E0~E9·일정·근거)
├── README.md           이 파일
├── .gitignore          데이터·가중치·onnx·hef 같은 큰 파일 제외
│
├── rn50/               현재 실험 코드 (CLIP-RN50 / RN50x4)
│   ├── README.md       설치·데이터 받기·실행 순서
│   ├── requirements.txt
│   ├── common.py       공통 규칙: 경로·전처리·데이터 구간·ONNX 세션·지표
│   ├── prompts/        프롬프트 템플릿
│   ├── baseline_rn50.py    [준비] PyTorch FP32 베이스라인 (정확도·크기·지연·메모리)
│   ├── export_onnx.py      [준비] PyTorch → float ONNX (+AttnPool q1, 동일성 검증)
│   ├── datasets.py         [준비] dev(train 분할)·보정 후보 / [최종] test 평가
│   ├── quantize_onnx.py    [양자화] ORT 정적 INT8, 층 선택, 보정 방법·출처
│   ├── evaluate.py         [측정] 모델 × dev 데이터셋 정확도·유사도
│   ├── bench_speed.py      [측정] 지연 반복 측정
│   └── summarize.py        [정리] 결과 → 마크다운 표
│
├── hailo/              본선 이식: DFC 스크립트(.alls)·층 이름 매핑·컴파일 로그 (E1, E8a, E8b)
├── results/            현재 실험 결과 (실험 ID별 폴더: E0/, E1/ ...)
├── docs/               팀 공통 문서 (코드 검토 기록)
├── siglip2/            준혁 작업 공간
│
└── archive/            이전 실험 (10/4~10/7). 참고용, 보고서 수치로 쓰지 않음
    └── 2026-10-07_night/   이전 PRD·결과·전용 코드
```

## 핵심 규칙 (자세한 내용: PRD 2·5절)

- **이미지 인코더만 양자화한다.** 텍스트는 FP32로 클래스 임베딩을 1회 계산해 저장한다.
- **정수 커널만 쓴다.** 보고 수치는 ONNX Runtime 기본 설정으로 측정하고, 측정 CPU를 함께 적는다. fake-quant는 쓰지 않는다.
- **예선 구성 = 본선 구성.** 예선 ONNX의 층별 정밀도를 DFC `.alls`로 그대로 옮길 수 있어야 한다.
- **데이터:** 보정·dev는 train 분할에서만 뽑는다. test는 최종 1회만 쓴다. 보정은 256장, 시드 0이다.

## 빠른 시작

```bash
cd rn50
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# 가중치·데이터 받기, 실행 순서: rn50/README.md
```
