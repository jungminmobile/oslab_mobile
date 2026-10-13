# 진행 로그 (SigLIP2 트랙)

세션마다 맨 아래에 추가한다. 형식:
```
## YYYY-MM-DD (세션 n)
- 단계:
- 한 일:
- 결과: (경로, 커밋)
- 3중 확인:
- 막힌 것 / 사람 결정 필요:
- 다음 할 일:
```

## 인수인계 (claude.ai 대화 → Claude Code)
- 단계: G0 시작 전
- 지금까지: 대회 규정 분석, 자유주제 확인, 노션 RN50 실험 로그 검토, SigLIP2 트랙 분담 결정(D-001), 로드맵 G0~G7 작성
- 팀 상태(노션 기준, 10/7): RN50 FP32 기준치 확보, AttnPool 그래프 최적화, 단계별 민감도(stem·layer1 민감), 보정 버그 수정 후 재측정 완료, 균형점 c3/c4, RN50x4 후보, Hailo DFC는 아직 미착수
- 팀 합의: 보고 정확도는 정수 커널만, fake-quant는 스크리닝 옵션 (D-005). 기록은 GitHub (D-006). PRD.md와 코드는 저장소에 업로드됨
- 다음 할 일: ROADMAP.md의 G0

## 2026-10-13 (세션 1)
- 단계: G0 (문서 작업만. 코드 수정·실험 없음)
- 한 일:
  - 로컬 환경 확인: 환경 변수 4개 정상, venv Python 3.11.9(패키지 없음), D 드라이브 176 GB 여유, `gh` 로그인(JJury2845, push 권한), CPU i7-1165G7 (avx2·avx512f·avx512_vnni 있음, avx_vnni 없음), NVIDIA GPU 없음, WSL Ubuntu-24.04·20.04에 Hailo venv 없음
  - 일정 점검: G0 5일, G1 판정일 1일 밀림. 일정 조정안 작성 (승인 대기)
  - 저장소 지도 `docs/repo_map.md` 초안: 공통 코드 재사용 가능/수정 필요 구분, PRD 핵심, requirements
  - `docs/g0_open_questions.md`: CLAUDE.md와 실제 상황 불일치 17건, 사람 결정 질문 Q1~Q14
- 결과: `siglip2/docs/repo_map.md`, `siglip2/docs/g0_open_questions.md` (브랜치 `siglip2/g0-handoff`)
- 3중 확인:
  1. 자기 검증: 문서의 핵심 사실(기본값, 입력 이름, 환경변수, PRD 문구, push 규칙)을 grep으로 원문과 대조 → 일치
  2. 독립 재계산: CPU 플래그를 WSL `/proc/cpuinfo`와 Windows `IsProcessorFeaturePresent`(avx2·avx512f) 두 경로로 확인 → 일치. CPU 이름은 WMI와 `/proc/cpuinfo` 일치
  3. 독립 검토: 맥락 없는 서브에이전트가 두 문서를 저장소와 대조 → 사실 오류 5건(라벨 수, `gh issue list` 동작, D-005 날짜 근거, G6·G7 일정 표기, 인용 위치)과 누락 7건(캐시 이름 충돌, 보정 풀 전처리, Pets 경로 불일치 등) 지적 → 모두 원문 확인 후 반영
- 막힌 것 / 사람 결정 필요: `docs/g0_open_questions.md` 3절 Q1~Q14 (특히 Q10 fake-quant 정책 충돌, Q4 DFC 머신, Q11 일정 조정). GitHub 라벨 `siglip2` 없음 → PR 라벨 미부착
- 다음 할 일: 질문 답변 받기 → 승인 항목을 DECISIONS·ROADMAP에 반영 → venv에 requirements 설치(RN50과 같은 버전 + timm) → `scripts/env_info.py` → SigLIP2 임베딩 1장 + CIFAR dev 100장 (G0 완료 기준)

## 2026-10-13 (세션 1 이어서)
- 단계: G0
- 한 일:
  - 질문 Q1~Q14 답변을 받아 D-008~D-017 확정, CLAUDE.md·ROADMAP·VERIFICATION 반영 (커밋 `772c578`, PR #3 본문 갱신)
  - DFC 준비 (D-011): `C:\Users\qaz43\.wslconfig`(memory=12GB, swap=16GB) 생성 후 `wsl --shutdown`. `wsl --install Ubuntu-22.04 --location D:\mokerton\wsl\ubuntu-22.04 --no-launch`로 설치(Ubuntu 22.04.5, Python 3.10.12, WSL 메모리 11GB 확인). 기본 사용자 `jjury`(비밀번호 없음, `/etc/wsl.conf`). root로 apt 설치: python3-dev, python3-venv, python3-pip, python3-tk, graphviz, libgraphviz-dev, build-essential, unzip
- 결과: vhdx `D:\mokerton\wsl\ubuntu-22.04\ext4.vhdx` (설치 후 2.1GB). 기존 Ubuntu-24.04·20.04는 그대로
- 3중 확인: 설치 확인만 (`/etc/os-release`, `python3 --version`, `dpkg -l`, `whoami`, `free -g`). 실험 결과 아님
- 막힌 것 / 사람 결정 필요: DFC 5.4.0 휠은 Developer Zone 로그인이 필요 → 쭈리가 받아 `D:\mokerton\files\`에 둔다. D-006 확정 날짜 미기입
- 다음 할 일: DFC 휠 설치(`~/hailo10_venv`) + 작은 모델 스모크 테스트 → Windows venv에 requirements 설치(RN50과 같은 버전 + timm) → `scripts/env_info.py` → SigLIP2 임베딩 1장 + CIFAR dev 100장 (G0 완료 기준)

## 2026-10-13 (세션 1, 밤 자동 진행 — 쭈리 취침 중)
- 단계: G0 완료 → G1(DFC 제외) → G2 일부
- 지시: "확인하지 않아도 되는 것부터 계속 진행, 크리티컬하지 않은 결정은 추천안으로 자동 결정" → 자동 결정은 DECISIONS D-019~D-023에 "확정(자동)"으로 기록
- 한 일:
  - venv 설치 (rn50/requirements.txt 참조 + timm 1.0.30, py-cpuinfo 9.0.0, transformers 5.19.0), `scripts/env_info.py`
  - **G0 완료**: `scripts/g0_smoke.py` — 임베딩 (1,768), CIFAR dev A 100장 top-1 76.0 (커밋 `1aed107`, 두 번 실행 일치) → `results/siglip2/G0_smoke`
  - Hailo Model Zoo 확인 (D-018 제안): SigLIP B/16 224·SigLIP2 B/32 256 공식 HEF 있음, **SigLIP2 B/16 224 이미지 인코더 HEF는 없음**
  - **G1 ONNX export**: `s2_export_onnx.py`, PyTorch 대비 코사인 0.9999998, 연산 목록, 재export sha256 일치 → `results/siglip2/G1_onnx_export`
  - **G2 dev 캐시**: `s2_build_dev.py` — 고화질 5종 장수가 RN50과 모두 같음(990/957/967/985/1024), FP32 top-1 Pets 93.33 · Flowers 86.73 · Cars 93.80 · Food 89.44 · Caltech 95.02 (커밋 전 dirty 기록 → G2 평가에서 다시 기록 예정)
  - G1 INT8 전부(MinMax, highres 256 s0, S8S8): 89.6 MiB(FP32 354.5 대비 −74.7%), 보정 256장 확인. 정수 커널 그래프: QLinearMatMul 66, QLinearSoftmax 13, MatMul 13 float, LayerNorm·Erf·patch Conv float
  - 독립 검토(서브에이전트) 지적 6건 반영 (`add8dc1`): 16장 버그를 실제로 막는 조건 assert, 실패 모델 삭제, 결과 경로 보호, exec 검사, export 재사용 검사, dirty 판정
- 실패·이상치 (숨기지 않음):
  - G0 첫 실행: `transformers` 없음 → 설치 (D-022)
  - `build_dev`가 rn50/export_onnx.py를 불러오는 이름 충돌 → `s2_` 접두사 (D-019). 데이터 처리 전 import 단계에서 멈춰 영향 없음
  - cp949 콘솔 인코딩으로 출력 중 멈춤 → stdout UTF-8 고정
  - CIFAR-100 공식 서버 약 65 KB/s, 169 MB에 48분 (정민 기록과 같음)
  - 첫 INT8 평가는 커밋 전 코드라 중단하고 재현 스크립트(`scripts/g1_pipeline.sh`)로 다시 기록
- 다음 할 일: g1_pipeline 결과 정리(README), G2 프롬프트 비교·FP32 S-A행·속도/메모리, DFC 휠 받으면 G1 DFC 단계

## 2026-10-13 (세션 1, 밤 자동 진행 이어서 → 세션 2 오전)
- 단계: G1(DFC 제외)·G2 완료, G3 민감도 지도 완료, G3 보정 방식 비교 중단
- 결과:
  - G1 INT8 전부(MinMax): 고화질 dev 91.66 → 1.37 (붕괴). 원인 진단: 정수 커널 문제 아님(fake-quant도 같음), 블록 5부터 잔차 흐름·fc2 입력 활성값 이상치(최대/99.9% 분위수 40~286배) → `results/siglip2/G1_int8_minmax_all`
  - G2 FP32 S-A행: 92,884,224 파라미터(PyTorch=ONNX), 17.71 GMACs, 354.45 MiB, peak RSS 433 MiB, 지연 403/348/265 ms(1/2/4스레드, 비공식), 고화질 dev 91.66. 프롬프트는 CLIP 공식 유지(D-025) → `results/siglip2/G2_fp32_baseline`
  - G3 민감도: 블록 5~11 하나만 INT8로도 붕괴, fc2만 INT8 cos 0.648, 블록 1~4·QKV·QKᵀ·AV는 cos ≥ 0.989 → `results/siglip2/G3_sensitivity`
- 실패·이상치:
  - MSYS(Git Bash) 경로 변환으로 `/trunk/...` 노드 이름이 망가짐 → `MSYS_NO_PATHCONV=1`. 그때 잘못 만들어진 only_head를 재사용 → 다시 만들고 스크립트는 재사용 금지로 수정
  - Windows에서 TaskStop이 손자 프로세스(python)를 끝내지 않아 이전 실행의 스크리닝이 계속 돌며 중복 행 기록(값은 같음)
  - Percentile 보정이 메모리 부족(bad allocation): ORT 1.25 HistogramCalibrater가 모든 배치의 중간 텐서를 쌓은 뒤 한 번에 히스토그램을 만듦 → 나눠서 누적하는 방식으로 진행(쭈리 승인, 동등성 확인 후)
  - 세션 전환 중 백그라운드 작업이 모두 중단됨
- 사람 결정(10/13 오전): 보정 OOM → 나눠서 누적 + 동등성 확인 / 단계별 PR 4개 / DFC 휠은 쭈리가 받음 / D-006 날짜 10/13
- 다음 할 일: 히스토그램 보정 나눠 누적(동등성 확인) → G3 보정 방식 비교 → 누적 구성 곡선 → DFC 설치·G1 컴파일 게이트
