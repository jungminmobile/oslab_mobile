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
