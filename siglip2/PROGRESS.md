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

## 2026-10-13 (세션 2, 오후~밤)
- 단계: G1 완료(판정: SigLIP2 계속, D-026) → G3 보정·누적 → G4 첫 구성
- 결과:
  - G1 DFC: 전 구간 컴파일 성공(HEF 69.2 MB, 18 컨텍스트, 2시간 16분), float 에뮬레이터 = ONNX(cos 1.0), 기본 8비트 붕괴(cos 0.12) → `results/siglip2/G1_dfc_gate`
  - G3 보정 비교: 블록 5~11 fc2 FP32 + Percentile 99.999가 최선(cos 0.959). Percentile·Entropy는 메모리 때문에 16장씩 나눠 누적(Percentile 동등, Entropy 근사) → `results/siglip2/G3_calib_method`
  - G3 누적: fc2(블록 5·9)만 FP32인 fc2L2(S8) 105.3 MiB(−70.3%), 고화질 Δtop-1 −0.50 [−1.01, +0.01](유의한 차이 없음). U8은 attention까지 정수로 돌지만 유의하게 나쁨(−0.41) → `results/siglip2/G3_cumulative`
  - G4 DFC: 잔차 합만 16비트 cos 0.72 → 공식 패턴(블록 0·5 MLP 추가) 0.963, 블록 5~11 fc2 0.936 → `results/siglip2/G4_dfc_first`
- 실패·이상치:
  - G4 B 에뮬레이션이 메모리 11.6 GB + 스왑 12 GB로 1시간 53분 걸림 → 배치 8 → 2로 줄여 다시 실행 중(D·E)
  - WSL 정리 후 첫 시작이 `HCS_E_CONNECTION_TIMEOUT`, 두 번째에 정상
  - 지연 측정은 PC가 바쁜 상태(다른 프로그램)라 미룸 → `scripts/run_when_idle.ps1`이 G4 종료 후 CPU 3분 연속 15% 미만이면 자동 측정
- 사람 결정: G1 SigLIP2 계속, 교차 확인은 G4 최종 후보로(D-027), Entropy는 근사로 표시, 열린 PR 브랜치 push는 묻지 않음, 지연은 한가할 때
- 다음 할 일: G4 D·E 결과 → 후보 조합 컴파일(HEF 크기·컨텍스트), 지연 측정 결과 정리, ORT 후보 + DFC 구성 후보표(팀 선택용), PR 검토 요청

## 2026-10-14 (세션 2 이어서, 밤샘 자동 진행 — 쭈리 수면 중, 결정은 모두 보류)
- 진행 방식: 개인연구 세션과 공용 잠금(`D:\mokerton\locks\heavy_lock.py`)으로 무거운 작업을 하나씩 번갈아 실행. 05:08부터 잠금에 "먼저 기다린 순서" 대기표 추가(한쪽 스크립트가 단계마다 바로 다시 잡아 다른 쪽이 몇 시간 밀리던 문제), 개인연구 세션에 알림
- 결과:
  - **G3 원인 분석** (`results/siglip2/G3_why`):
    - 1단계: 블록 5부터 고정 채널(67·301·145, 블록 9부터 483)·고정 위치 토큰(패치 55, 블록 5~7 패치 1, 블록 9~ 패치 13)에 거대 값, attention 28~52%가 그 토큰으로, 텐서별 8비트 한 칸 22~78배 → 84~99% 0 반올림
    - ORT 실제 scale과 1% 안팎 일치(독립 재계산), dev 6종(CIFAR A 포함) 모두 같은 구조
    - 2단계: 거대 값은 블록 5·9의 MLP가 써 넣음. 지우면 cos 0.50. 8비트에서 망가지는 쪽은 보통 토큰(블록 5 입력, 블록 9 출력)
  - 히스토그램 16장 단위 동등성: Percentile 동등(깨끗한 재실행 값 일치) → `G3_calib_method`
  - 캐시 결정성: Flowers·Cars·Caltech·Food 모두 sha256 일치(Caltech·Food는 깨끗한 커밋) → `G2_fp32_baseline` README
  - G4 DFC 스크리닝(dev 6×200, 같은 사진으로 ORT와 짝 비교, `s2_compare_dfc_ort.py`):
    - D: 고화질 cos 0.952, −1.1 [−2.3, 0.0]
    - E: 0.919, −2.3 [−3.7, −0.9]
    - ORT fc2L2: 0.956, +0.3. D는 ORT보다 1.4%p 유의하게 낮음
- 실패·이상치 (숨기지 않음):
  - DFC 로그 파일이 siglip2/에 남아 첫 원인 분석·히스토그램 확인이 dirty → 파일 옮기고 exclude, 깨끗한 재실행(값 완전 일치)
  - 실행 중 새 스크립트를 만들어 Flowers·Cars 캐시 확인이 dirty → 재실행 예약
  - **원인 분석 2단계의 분위수 버그(독립 검토 발견)**: 1/7 간격 표본이 196 = 28×7 때문에 같은 위치만 뽑아 거대 값을 놓침 → pq/clip 행·해석 1개 무효 표시, 정확한 정의 + ORT 실제 범위로 재실행 대기
  - "패치 55→13으로 옮겨 간다"는 추정이 틀림(실제로는 55 유지 + 13 추가) → v2로 정정
  - Hugging Face 요청 제한(429)으로 Flowers 재실행 실패, 원인 분석 v2는 재시도 끝에 성공 → 남은 재실행은 HF_HUB_OFFLINE=1
  - WSL 첫 밤샘 스크립트가 G4 5개(약 5.5시간)를 연달아 잡아 Windows 작업이 밀림 → 순서 바꾼 `overnight_wsl2.sh`(D 컴파일 앞당김)
  - G4 DFC 스크리닝 1구성 약 70분, WSL 메모리 11 GB + 스왑 5~8 GB
- **아침에 사람이 정할 것 (밤새 보류)**:
  1. ORT 최종 구성: fc2L7 / L4 / L2 (서로 동률, PRD 규칙상 cos는 L7, 크기는 L2) — 지연 측정 후 팀 후보표로
  2. S8 vs U8 (D-023 재검토): U8은 QKV 투영까지 정수로 돌지만(10/14 정정: attention 행렬곱 아님) 유의하게 나쁨(−0.41 [−0.78, −0.02])
  3. G4 최종 `.alls`(D가 현재 최선이지만 ORT보다 유의하게 낮음)와 정민 교차 확인 시점(D-027)
  4. 원인 분석 결과를 보고서 서사("CNN은 앞쪽, ViT는 거대 값을 쓰는 중간 MLP")로 쓸지
  5. 일정(내부 마감 10/31)
- 다음 할 일: 원인 분석 v2(writers) 결과로 README 6-2 정정 → 3중 확인 재검토, G4 C·B·A 스크리닝, D 컴파일·잡음 분석, 지연 측정(한가할 때), Flowers·Cars 깨끗한 재실행, STATUS.md·PR #7 본문 갱신

## 2026-10-14 (오전~14:30, 쭈리 복귀 후 → 잠시 중단)
- 사람 결정: 지연 측정을 WSL 작업보다 먼저(D-029), DFC 잡음 분석 보류(D-028), Windows 보안 업데이트가 끝나면 지연 측정을 이어서 하고 그동안 CPU와 상관없는 실험 먼저, 14:30 "지금 하던 것까지만 하고 정리·종료, 나중에 재개"
- 결과:
  - 원인 분석 2단계 v2(정확한 분위수·ORT 실제 범위·위치별 잔차 분해), 독립 검토 2회 반영 → `results/siglip2/G3_why` README 6-2절
  - 원인 분석 3단계(눈금 방식, 64장, **dirty 실행 — 깨끗한 재실행 필요**): fc2 5·9 8비트 cos 텐서 0.728 / high-norm 제외 0.995 / 토큰별 0.999 / 채널별 0.9995, fc2 12개 전부도 토큰·채널별이면 0.999 (`why_fix.jsonl` 1번째 줄)
  - G4: C·D·E 스크리닝 README, D 컴파일(HEF 99.8 MB, 컨텍스트 17), 잡음 분석 실패 기록, 독립 검토 반영
  - 후보표 초안(`results/siglip2/candidates`) + peak 메모리(fc2L2 227.5 MiB, −47.4%)
  - 고화질 5종 캐시 결정성 모두 깨끗한 재실행으로 확인
  - DECISIONS D-028·D-029(확정), D-030~D-033(팀 결정 제안), 보고서 SigLIP2 절 초안(`docs/report_section_draft.md`)
- 실패·이상치:
  - 지연 측정 5번 시작 → 값 없음 또는 오염: ① 10:50·10:52 잠금 도구가 WSL bash를 잡아 환경 변수 누락(→ Git Bash 전체 경로) ② 11:43~12:24 다른 세션 pip 설치·Defender 검사와 겹쳐 오염 ③ 14:07 Windows Update(5코어)와 겹쳐 중단. **아직 깨끗한 지연값 없음**
  - DFC 스크리닝 dirty 오표시(WSL git CRLF) → 판정 코드 수정(`3d5c120`), 재실행 여부는 팀 결정(D-033)
  - WSL 분리 실행이 한 번 바로 종료됨 → `Start-Process wsl.exe`(창 숨김)로 클라이언트를 띄워 둠
- **재개 방법 (다음에 시작할 때)**:
  1. `wsl -l --running`이 비었는지, `python D:/mokerton/locks/heavy_lock.py status`가 null인지, `D:\mokerton\locks\queue`가 비었는지 확인
  2. 지연 측정(PC를 비우고): `bash scripts/speed_when_idle.sh` (Windows Update 조용 → 잠금 → 스레드 1·2·4). 환경 변수는 `siglip2/.claude/settings.local.json`에서 오지만, Start-Process로 띄우면 `export SIGLIP2_PYTHON=...` 등을 명령에 직접 넣는다
  3. 원인 분석 3단계 깨끗한 재실행: `RN50_RESULTS=../results/siglip2/G3_why "$SIGLIP2_PYTHON" s2_why_fix.py --n 32` (잠금 안에서)
  4. 새 venv 재현: `bash scripts/g3_repro_venv.sh`
  5. WSL G4: 보정 256장 D의 최적화는 끝났다 (`~/siglip2_dfc/g4_calib256_20261014/opt_D256_official_b9fc2.har`). 남은 단계: D256 스크리닝 → B·A 스크리닝 → E·C 컴파일. `hailo/g4_more_wsl.sh`에서 최적화 단계를 건너뛰도록 고쳐 다시 실행한다
- 다음 할 일: 위 재개 1~5, 지연 채운 후보표로 팀 결정(D-030~D-033), PR #7 검토 요청

## 2026-10-14 오후 ~ 10-15 새벽 (재개 후 → 쭈리 수면 중 자동 진행)
- 사람 결정: 노트북은 고성능 모드에서 측정 (17:04 측정은 저소음 모드라 버림), 프로파일링·새 옵션·reduce_range 실험 진행 요청, 게임 중에는 정확도 실험만 (시간 측정은 자동으로 미뤄짐)
- 결과:
  - **지연 (세 번, 비공식)**: U8 계열은 세 번 모두 FP32의 약 절반이다. S8은 4스레드에서 FP32와 비슷하고 측정마다 흔들린다 (+2.5 / −10.6 / +0.5%). 절대값도 측정마다 다르다 (4스레드 FP32 148 / 211 / 245 ms) → `G3_cumulative/speed*`
  - **연산별 프로파일**: S8에서는 QKV 투영 12개 + 헤드 kv가 정수로 합쳐지지 않는다. 그 Q/DQ가 4스레드 시간의 36%이고, 스레드를 늘려도 줄지 않는다 (H1 채택). 정수 MatMul 확장성은 정상이다 (H2 기각) → `G3_cumulative/profile`
  - **정정**: "S8에서 attention 행렬곱 13개가 정수로 안 합쳐진다"는 틀렸다 (실제로는 QKV 투영). 문서 7곳 정정, `--attn-u8` 실험은 효과 없음으로 기록
  - **QKV만 U8 (`--u8-match`)**: U8과 dev 5,923장 예측이 완전히 같다 → S8과 U8의 정확도 차이(−0.41%p)는 전부 QKV 정수 경로에서 나온다 → `G3_cumulative/full_qkvu8`
  - **QKV 정수 경로 대책**: reduce_range −0.78, U8U8 −0.64. 둘 다 S8과 동률이지만 U8보다 유의하게 좋지는 않다. reduce_range는 U8과 같은 속도, U8U8은 느리다. x64quantprecision은 VNNI PC라 출력이 같다 → `G3_cumulative/qkv_precision`
  - **DFC**:
    - 보정 256장 D는 cos 0.955, −0.2 [−1.4, +1.0]로 ORT와 동률이다
    - 6개 구성 스크리닝 완료 (A −60, B −3.9, E −2.3, C −1.4, D −1.1, D256 −0.2)
    - C·E 컴파일 (97.1 MB·16, 95.1 MB·16 컨텍스트)
    - 최적화 수준 1은 GPU 없이 동작함 → D256L1 cos 0.967(최고), −0.4 [−1.4, +0.6], CIFAR −8.5 (10/15 07:32 완료). 컴파일 99.8 MB·17 컨텍스트
  - **새 venv 재현**: export·양자화 sha256과 dev 전체 예측이 비트 단위로 같다 → `G3_repro`
  - **원인 분석 3단계**: 토큰별·채널별 눈금이면 fc2를 전부 8비트로 해도 cos 0.999 (깨끗한 재실행 값 일치)
  - 독립 검토 3회 (G3_why 재검토, G4·후보표, 밤 결과): 지적 반영
- 실패·이상치 (숨기지 않음):
  - **스크립트 종료 코드 버그**: `echo "$(date) rc=$?"`가 date의 0을 찍었다 → 32곳 수정. 과거 결론은 결과 파일·값으로 확인해 왔으므로 영향 없음. G4_noise의 "종료 코드 0" 서술은 정정
  - 재현 테스트 pip 첫 시도가 requirements의 한글 주석(cp949) 때문에 실패 → PYTHONUTF8=1
  - 지연 측정 중 개인연구 업로드와 겹침, 저소음 모드, Windows Update, 메모리 부족(adb 시간 초과)이 있었다 → 잠금 안 CPU 확인, "needs idle PC" 표시, WSL과 Windows 무거운 작업 동시 실행 금지로 대응
  - 보정 1024장은 고화질 보정 후보 풀이 311장이라 불가 (풀 확대는 팀 결정)
- **아침에 정할 것**:
  1. ORT 최종(S-B): S8 fc2L7/L4/L2 (정확, 4스레드에서 FP32와 비슷) vs U8 / U8+reduce_range (약 2배 빠름, 0.3~0.4%p 낮음)
  2. 4스레드 속도 하한 판정을 정민 노트북 공식 측정으로 할지 (D-010)
  3. DFC 최적(S-D): C / D / D+보정 256장 (+ 최적화 수준 1 결과), 정민 교차 확인 시점
  4. DFC dirty 오표시 3행 재실행 여부 (D-033)
  5. 보정 후보 풀 확대 여부 (1024장 실험용)
  6. 4비트(a8_w4) 시작 시점 (8비트 최종 결정 뒤)
- 다음 할 일: DFC 수준 1 결과 정리, 독립 검토, 팀 결정에 따른 G6 동결 준비

## 2026-10-15 (오전~밤, 쭈리 복귀 후 → 22:20 게임으로 PC 작업 일시 정지)
- 사람 결정:
  - D-030·D-031 **확정**: ORT 최종 = fc2L2 U8 + reduce_range (정민·준용 찬성. 준용 측정: x86 Ultra 9 285K에서 S8 1.10배 / U8 2.16배, ORT 1.23.2·1.25 같음, ARM Orin은 둘 다 2.26배)
  - D-033 확정(dirty 3행 재실행 안 함), D-035 확정(보정 1024장 예선 제외), D-036 결정 위임(최종 점수가 좋아지는 방향, 사람 확인 유지 항목 명시)
  - **D-034 확정: 예선은 모델 하나 제출** (학술부 답 — CNN vs Transformer 비교 보고서는 탈락). RN50·SigLIP2를 따로 최적화하고 더 좋은 쪽 제출 → 문서의 서사 표현 수정, 보고서 초안 v2
- 결과:
  - **C256L1** (C + 보정 256장 + 수준 1): 스크리닝 cos 0.964, −0.8 [−1.8, +0.2], CIFAR −7.5. D256L1과 동률(예측 98.5% 일치). 컴파일 97.1 MB·16 컨텍스트(D256L1보다 2.7 MB 작음) → `G4_dfc_screen` 6-4, `G4_compile`
  - **D256L1 dev 전체**: −1.19 [−1.78, −0.62] (유의한 손실), cos 0.967. 200장 표본(−0.4)이 유리하게 뽑혔음. ORT S8보다 0.69%p 낮고 ORT U8+rr과 동률 → `G4_dfc_full`
  - 대회 최종 안내 원문을 읽고 요구사항 체크리스트 작성 → `docs/contest_requirements.md` (PRD의 "속도 하한"은 원문에 없는 팀 규칙)
  - 팀 질문지(Claude Docs) 작성, 다른 기기 측정용 묶음을 GitHub pre-release `siglip2-xplat-20261015`로 공유(FP32·S8·U8+rr ONNX, dev200, `scripts/portable_eval.py`)
  - 4비트 옵션 `--w4-regex`(기본 꺼짐) 구현, `onnx_ir 1.0.0` 설치(다른 패키지 변화 없음, dry-run 확인)
- 실패·이상치 (숨기지 않음):
  - D256L1 start=800 구간이 caltech에서 **OOM**(rc=247, python 11.7 GB) → `--sets`로 3종씩 나눠 재실행
  - README 2곳에 Windows 경로 `\a`가 제어 문자로 깨져 있었음 → 정정
  - w4_sweep의 바깥 잠금 모드 첫 구현이 `env`로 job 이름을 명령으로 실행하는 버그 → `nolock` 함수로 고침(실행 전 발견)
  - 개인연구 E20(메모리 2~3 GB) 동안 WSL을 내려 달라는 요청 → 22:02 WSL 종료, C256L1은 9조각 중 4조각 완료 상태로 멈춤
- **22:20 정지 상태** (쭈리 게임): WSL 내려감. 대기 작업(4비트, WSL 재시작) 취소. `speed_1016.sh`·portable_eval 검증 대기 스크립트는 "G4_MORE7 DONE"을 기다리며 대기(그 전에는 아무것도 안 함)
- 재개할 때 순서: ① 4비트 `W4_ONE_LOCK=1`로 `scripts/w4_sweep.sh`(약 40분) ② `scripts/yield_wsl_1015.sh` 없이 바로 `hailo/g4_more7b_wsl.sh`(C256L1 남은 5조각, 약 2시간 반) ③ portable_eval 검증 ④ `speed_1016.sh`(지연 2세션) — 개인연구 세션은 "SPEED_1016 DONE" 뒤에 E20 SigLIP2 분석을 넣기로 함
- 다음 할 일: 위 재개, 4비트 결과로 최종 포함 여부 결정(D-036), 3번(Hailo) 결정, 최종 동결 → test 1회, 재현 README

## 2026-10-16 00:40 재개 (쭈리: PC 작업 다시 시작, 개인연구 세션과 조율)
- **사고 (숨기지 않음)**: 10/15 22:20 정지 요청 때 백그라운드 작업(TaskStop)을 끝냈지만, Windows에서 `yield_wsl_1015.sh`의 자식 bash가 살아남아 22:55 개인연구 E20이 끝나자 WSL을 다시 띄웠다 → **게임 중(22:56~23:08) C256L1 start=600 a가 돌았다.** 23:09 재부팅으로 WSL이 죽으면서 공용 잠금이 남아 **00:40까지 개인연구 작업을 막았다**. 00:40 저희 잠금 해제(lock_log release 기록), 개인연구 세션에 사과·설명
- 대책: 한 스크립트(`scripts/night_1016.sh`)로 순서 실행, 단계 사이마다 `D:\mokerton\locks\siglip2.STOP` 파일을 확인해 멈춘다 (프로세스 종료에 기대지 않음). 다음부터 정지 요청 때는 STOP 파일을 만들고, 남은 bash·wsl 프로세스와 잠금 상태를 직접 확인한다
- 순서: 개인연구 E20(00:42~약 02:35) → 4비트 sweep → C256L1 남은 5조각(WSL) → portable_eval → 지연 2세션 → "SPEED_1016 DONE"(약 07:20)
- 결과 (10/16 01:46~07:52, 자동 진행):
  - **C256L1 dev 전체** −1.56 [−2.09, −1.03], D256L1과 +0.38 [−0.03, +0.76] 경계 동률 → **D-032 확정 C256L1** (D-038 사전 규칙, D256L1은 정확도 우선 대안) → `G4_dfc_full` 7절
  - **지연 3회 완성**: U8+rr 4스레드 −49.6 / −48.7 / −49.5% (모든 스레드 약 −50%), S8은 +0.5 / −13.7 / −11.9% → `G3_cumulative/speed_1016a`
  - portable_eval 검증: 기존 스크리닝과 정확히 같음 → `G3_xplat`
  - **4비트**: 크기 −30%·정확도 동률이지만 4스레드 지연 약 2배 → **D-039 넣지 않음** → `G3_w4`. 첫 세 번은 계산 전에 멈춤(sha256 비교 버그, 정규식 고정, ORT nodes_to_include 의미) — 모두 assert·확인 단계가 잡음
  - 4비트 0단계: 옵션을 끈 기본 경로가 이전과 비트 단위로 같음(sha256 SAME)
- 개인연구 세션과 조율: E20(00:42~01:42) 먼저, 이후 SigLIP2. 개인연구는 07:55까지 무거운 작업을 넣지 않고 기다려 줌
- **최종 동결 조건 충족**: ORT = fc2L2 U8 + reduce_range (D-030), Hailo = C256L1 (D-032), 4비트 제외 (D-039)
- 다음 할 일: **test 1회 평가 (쭈리 확인 필요)**, 재현 README 마무리(test 명령), 독립 검토, 준용 ARM 결과 비교, 정민께 공식 지연·교차 컴파일 요청

## 2026-10-16 밤 ~ 10-17 오전 (test 1회 평가, 쭈리 승인)
- 사람 결정: test 1회 평가 시작 승인, **D-032 Hailo 최종을 D256L1로 변경** (10/16 C256L1 판정이 난수 시드에 따라 뒤집히는 경계값 — 독립 검토 지적)
- Claude 결정 (D-036): D-040 Hailo는 예선 test로 평가하지 않음 (에뮬레이터 40시간+, 본선 실칩에서)
- 결과: **test 고화질 5종 Δ −0.83 [−1.01, −0.66]%p** (FP32 90.34 → 89.51), CIFAR-100 −4.71, ImageNetV2 −1.95 → `G6_test`
- 이상치: ImageNetV2 클래스 이름 중복(missile·sunglasses)으로 점수 동점 → 계산 방법에 따라 0.05%p 차이. RN50도 해당 → STATUS 정민 공유란
- 독립 검토 (10/16): 숫자는 모두 raw와 일치, 낡은 문서 문장과 C256L1 판정의 난수 민감성 지적 → 반영
- 다음 할 일: 정민께 공식 지연(x86)·Hailo 교차 컴파일(D256L1) 요청, 준용 ARM 결과 비교, 팀원 재현 확인, 보고서 다듬기, RN50과 비교·제출 모델 선택(팀)

## 2026-10-18 (G3R 레지스터 캐시 판별, 용 — 연구실 데스크탑 WSL Ubuntu-22.04, 브랜치 `siglip2/g3r-regcache`)
- 환경: uv Python 3.11 venv, dev 7종 재구축(준혁 G2와 장수·top-1 동일), FP32 직접 export가 Release `333cad08…`와 비트 단위 동일. identity는 5eee 불일치(AVX2 vs AVX-512 보정 눈금 차이) → c887a42 코드와 같아 코드 동일성 통과
- 결과: G1a·G1b 통과 (R1 FP32 Δ +0.10 [0.00, +0.22]). **G2 실패** (R1-INT8 − R0 −0.06 [−0.44, +0.33], 크기 −12.8%, t4 −4.1% 비공식). G3a는 토큰 축 Concat 미지원 → 제자리 마스크(D-041)로 파싱 통과, ORT INT8은 R1과 동률. **G3b 실패** (sink 잔차 0 → Hailo LN 분해 1/√ε 이상치, 마지막 시도 h_c 채움은 사전 확인 (c) 실패)
- 결론: G3R 종료. ORT 최종(D-030) 그대로, G3R은 Hailo에도 쓰지 않음 (Hailo 최종은 같은 날 D-042에서 rothad_D256L1로 바뀜). 원인 가설(공통 −0.6~0.7%p는 QKV/LN, D-031)은 `siglip2/g3q-rotate`에서
- 막힌 것·이상: 코드 버그 2개(stats hook, fp32 이중 감싸기) 수정 후 재실행. 1차 지연 측정이 DFC 로그 파일로 dirty → 재측정. WMI LoadPercentage 게이트 부정확 → 프로세스 합 게이트로 교체. DFC 스크리닝 RSS 18 GB로 WSL available 1 GB까지 → 중단, 이후 flock 공유
- 다음 할 일: PR 전 `siglip2/g3-sensitivity` 최신으로 rebase, 3중 확인(독립 검토) 후 PR (용 확인)
- (추가 17:30) 용 요청 4번째 Hailo 시도(분산 맞춤 h_c*): 사전 확인 통과·LN 이상치 제거됐으나 스크리닝 cos 0.733으로 실패 → G3R Hailo 종료. 다음: rebase·독립 검토·PR
- (추가 저녁) 용 요청 5번째(진짜 마지막) Hailo 시도: 진단(층별 정밀도·optimize 저장 SNR·limvals) → 주원인 잔차 위 8비트 마스크 연산 확인 → M2 처방. M2 사전 확인 (d) 실패(블록 5의 55번 행 1/√가 일반 범위 아래)로 optimize 안 함 → G3R Hailo 종료. `hailo analyze-noise`는 원래 D256L1에서도 DFC 내부 오류(768 vs 780)
