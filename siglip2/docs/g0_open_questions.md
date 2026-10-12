# G0 첫 세션: 일정 점검 · 규칙 불일치 · 결정 질문 (2026-10-13)

- 상태: **답변 반영** (2026-10-13). 확정된 항목은 `DECISIONS.md` D-008~D-017에 옮겼다. 결과는 맨 아래 4절.
- 근거 문서: `siglip2/CLAUDE.md`, `ROADMAP.md`, `PRD.md`, 루트 `CLAUDE.md`, `rn50/` 코드, `docs/repo_map.md`.

## 1. 일정 점검 (오늘 10/13 화)

ROADMAP 기준일은 10/7이다. 오늘까지 SigLIP2 트랙은 **코드·실험이 하나도 시작되지 않았다** (venv도 비어 있음).

| 단계 | 계획 기간 | 상태 | 밀린 정도 |
|---|---|---|---|
| G0 준비 | 10/7~10/8 | 이번 세션에서 문서 작업만 진행 중 | **5일** |
| G1 컴파일 게이트 | 10/8~10/12 (판정 10/12) | 시작 전 | 판정일 **1일 지남** |
| G2 FP32 기준치 | 10/10~10/14 | 시작 전 | 마감 내일 |
| G3 데스크탑 양자화 | 10/13~10/19 | 시작 전 | 오늘 시작 예정이었음 |
| G4 DFC 혼합 정밀도 | 10/15~10/23 | — | |

함께 볼 것:
- RN50 트랙도 E0(계획 10/8~10/11)이 아직 main에 병합되지 않았다 (10/12 재현 확인만 병합). 공통 코드의 기본값이 곧 바뀐다.
- DFC는 정민 노트북에만 있다 → G1(DFC parse/compile)은 DFC 머신이 정해져야 시작할 수 있다.

**일정 조정 제안 (승인 필요, 승인되면 ROADMAP·DECISIONS 수정)**

| 단계 | 제안 기간 | 메모 |
|---|---|---|
| G0 | 10/13~10/14 | venv 설치, env_info, SigLIP2 임베딩 1장 + CIFAR dev 100장 |
| G1 | 10/14~10/17 (판정 10/17) | ONNX export는 이 PC, DFC parse/compile은 DFC 머신. G2와 병행 |
| G2 | 10/15~10/18 | G1과 병행 (FP32 기준치는 DFC와 무관) |
| G3 | 10/18~10/23 | 블록 14회 + 연산 유형 스윕. 범위를 줄일지 함께 결정 필요 |
| G4 | 10/20~10/25 | G3와 일부 겹침 |
| G5 | 실기기 접근이 정해지면 (최소 범위) | |
| G6 | 10/26~10/28 (원래 10/23~10/27, 동결 10/27 → 10/26) | test 1회 |
| G7 | 10/29~10/31 (원래 10/28~10/31) | 내부 마감 10/31은 그대로 |

G0~G4가 4~5일씩 밀리므로, 대안으로 **G1 판정 전이라도 G2를 먼저 시작**하는 것과 **G3 범위 축소(블록 단위만, 연산 유형은 상위 블록에서만)**를 함께 제안한다.

## 2. `siglip2/CLAUDE.md` 규칙과 실제 저장소 상황이 맞지 않는 부분

| # | 위치 | CLAUDE.md 내용 | 실제 상황 | 제안 |
|---|---|---|---|---|
| 1 | 3절, D-005 | fake-quant는 **스크리닝 옵션**으로 허용 (팀 합의, D-005 확정. 날짜 미기입) | PRD R5와 루트 CLAUDE.md: "**fake-quant는 쓰지 않는다(디버깅에만 허용)**". 둘 중 어느 합의가 나중인지는 문서로 알 수 없음 | **충돌.** 팀이 어느 쪽인지 다시 확인. 정민과 합의 후 D-005를 유지하거나 대체 항목을 추가 |
| 2 | 4절 지표, D-007 | 헤드라인은 **CIFAR-100** (CLAUDE.md 4절과 D-007 둘 다) | PRD: 주 정확도 지표는 **고화질 5종 평균**, CIFAR·ImageNetV2는 별도 표기 | PRD에 맞춰 "헤드라인 = 고화질 5종 평균"으로 CLAUDE.md 4절과 D-007을 함께 고치는 것을 제안 |
| 3 | 3절 | 속도는 "예열 후 번갈아 **5회** 반복 (RN50과 같은 스크립트)" | `bench_speed.py` 기본은 워밍업 10·반복 3, PRD는 워밍업 20·200장×**3회** | RN50과 같은 조건(PRD 5절)으로 고치는 것을 제안 |
| 4 | 3절, G6 | U8S8 포화 대책으로 "`reduce_range` **또는 S8S8**" | PRD 3절: x86에서는 S8S8도 U8S8로 바뀌어 계산이 같다 → S8S8은 포화 대책이 아님. 대책은 `reduce_range`·U8U8·`session.x64quantprecision=1` | PRD 문구로 고치는 것을 제안 |
| 5 | 4절 측정 PC | "팀 공식 측정 PC 미정" | PRD 4-3: **주 측정 조건 = 정민 노트북 스레드 4**, 클라우드 1·2 보조 | 이 PC 수치는 "비공식" 유지. SigLIP2 속도도 정민 노트북에서 재야 RN50과 비교됨 (질문 Q1) |
| 6 | 6절 결과 파일 | `siglip2/results/<날짜>_<실험명>/` | 루트 CLAUDE.md·`results/README.md`: `results/<실험ID>/`, jsonl 필드 목록(`commit`, `dirty`, `options`, `cpu`, `vnni`, `ort` ...) 고정 | 위치를 정해야 함 (질문 Q8). 어느 쪽이든 jsonl 필드는 `results/README.md`를 따르는 것을 제안 |
| 7 | 7절, ROADMAP G5 | 실기기 = **Jetson Orin** + Hailo-10H | PRD·루트 문서에는 Orin이 없음 (Pi 5 + Hailo-10H는 본선 지급). PRD는 "NPU 실측은 본선 전에는 어렵다" | Orin 접근이 실제로 가능한지 확인 (질문 Q5) |
| 8 | 8절 | "본선 Pi HailoRT 5.1.1, **연구실 5.3.0**" | `hailo/DFC_SETUP.md`: 정민 DFC **5.4.0**, Model Zoo HEF도 5.4.0. Pi 버전은 주최측 확인 중. (PRD 3절 59행은 아직 "DFC 미설치"로 남아 있음 — PRD 쪽 문구가 오래됨) | 버전 표를 하나로 맞춤 (질문 Q4). PRD 문구는 정민에게 알림 |
| 9 | 6절 라벨 | `siglip2`, `experiment`, `decision`, `verification`, `blocked` | 저장소에는 GitHub 기본 라벨 10개뿐. **위 라벨 모두 없음**, 마일스톤 없음 | 생성 여부 확인 (질문 Q6) |
| 10 | 6절 브랜치 | `siglip2/<단계>-<주제>` | 루트 CLAUDE.md는 `exp/<실험ID>-<이름>`, 예시에 `siglip2/baseline` 포함 → 충돌 아님 | 그대로 사용 |
| 11 | 2절 세션 루틴 | `git log -- <RN50 디렉터리>` | RN50 디렉터리 = `rn50/`. 공통 조건은 `PRD.md`, `results/`, `hailo/`에도 있음 | `git log --oneline -10 -- rn50 PRD.md hailo results`로 구체화 제안 |
| 12 | 2절 세션 루틴 | `gh issue list --label siglip2` | 명령은 정상 종료(exit 0)하지만, 라벨이 없어서 **항상 빈 목록**이 나온다 → "열린 작업 없음"으로 잘못 읽을 수 있음 | 라벨 생성 전까지는 `gh issue list --search siglip2`도 함께 확인 |
| 13 | 4절 | "공통 코드는 import해서 쓴다" | 공통 코드는 `rn50/` 안에 있고 RN50 전용 상수(전처리, 토크나이저, 파일 이름)가 섞여 있다. 그대로는 SigLIP2 전처리를 쓸 수 없음 | `repo_map.md` 4-5절의 공통 코드 PR 제안 참고 (질문 Q9) |
| 14 | 10절 | 데이터 경로는 `SIGLIP2_DATA` | RN50 코드는 `RN50_DATA`(CIFAR), `RN50_EXTRA`(고화질 원본), `RN50_CACHE`, `RN50_ONNX`, `RN50_RESULTS`를 쓴다 | 매핑 규칙 필요 (질문 Q3) |
| 15 | ROADMAP G2 (64행), D-007 | "7개 데이터셋 dev 전체 FP32 정확도" | ImageNetV2는 test 전용이라 **dev가 없다** (dev는 6종: CIFAR A + 고화질 5종, CIFAR B는 최종 후보 확인용) | "dev 6종 + test 7종"으로 문구 정리 제안 |
| 16 | `siglip2/README.md` | "루트 README의 '실험 규칙'" | 루트 README의 절 이름은 "핵심 규칙" | 다음 문서 PR에서 문구 수정 제안 |
| 17 | 3절 / 루트 CLAUDE.md | (언급 없음) | 루트 CLAUDE.md: push 전에 "커밋 N개를 push할까요?" 확인 필수 | 이 트랙도 같은 규칙을 따름 (이번 PR도 push 전에 확인) |

## 3. 사람이 결정해야 할 질문

| # | 질문 | 제안 (Claude 의견, 미확정) | 근거 |
|---|---|---|---|
| Q1 | **팀 공식 측정 PC**를 무엇으로 할까? 이 PC(i7-1165G7)를 쓸까? | 공식 속도는 PRD대로 **정민 노트북 스레드 4**. 이 PC는 정확도 실험·스크리닝용, 속도는 "비공식". 최종 후보만 정민 노트북에서 측정 | CNN vs Transformer 비교는 같은 PC여야 성립 |
| Q2 | **GPU**를 쓸까? | 이 PC는 NVIDIA GPU가 없음. 예선 측정은 CPU만(PRD). DFC 최적화 레벨 2 이상·QFT에는 GPU 서버가 필요 → 쓸 수 있는 GPU 서버가 있는지 알려 주세요 | `hailo/DFC_SETUP.md` 7장 |
| Q3 | **데이터 경로**: `RN50_*`와 `D:\mokerton\data`를 어떻게 맞출까? | SigLIP2 스크립트가 import 전에 `RN50_DATA=D:\mokerton\data\cifar100`, `RN50_EXTRA=D:\mokerton\data\extra`, `RN50_CACHE=D:\mokerton\artifacts\siglip2\cache`, `RN50_ONNX=D:\mokerton\artifacts\siglip2\onnx`를 설정. 원본 데이터 폴더 구조는 **코드가 찾는 구조**(`pets/test-0.parquet` 등)에 맞춤 (`rn50/README.md`의 Pets 경로는 코드와 다름). 캐시는 전처리가 달라 RN50과 공유하지 않음. `RN50_CACHE`·`RN50_RESULTS`를 빼먹으면 RN50 파일을 덮어쓰므로 assert로 막음 | RN50 코드는 수정하지 않고 환경변수로만 맞춤 |
| Q4 | **DFC**를 어느 머신에서, 어느 버전으로 돌릴까? | 이 PC WSL에는 DFC가 없음. 선택지: (a) 정민 노트북 DFC 5.4.0 공유, (b) 이 PC Ubuntu-24.04에 같은 5.4.0 설치(Developer Zone 계정 필요, 수 GB), (c) 연구실 머신. 버전은 RN50과 **같게** | DFC 버전 = Pi HailoRT 버전 규칙 |
| Q5 | **실기기(Orin + Hailo-10H)** 접근 방법은? | 접속 방식(SSH 주소·계정·사용 시간), 설치된 HailoRT 버전, 지연 측정 권한을 알려 주세요. 기기 설정 변경·sudo는 하지 않음 | CLAUDE.md 7절 |
| Q6 | GitHub **라벨·마일스톤**을 만들까? | 라벨 `siglip2`, `experiment`, `decision`, `verification`, `blocked` 5개와 마일스톤 G0~G7. 공유 저장소 설정이라 **정민에게도 알리고** 생성 | 현재 라벨 없음. 이 계정은 push 권한(admin 아님) |
| Q7 | **대용량 모델 파일** 공유 방법은? | 제출용 ONNX(베이스라인 + 최적화)는 GitHub Release(파일당 2GB 한도) + sha256 기록. 중간 산출물은 공유 드라이브. RN50 트랙과 같은 방식으로 | 채점자 재현 (PRD R1) |
| Q8 | **결과 폴더 위치**: `siglip2/results/<날짜>_<실험명>/` vs 루트 `results/<실험ID>/` | `results/siglip2/<단계>_<주제>/` 또는 `siglip2/results/`. jsonl 필드는 `results/README.md`와 같게 | 2절 #6 |
| Q9 | **공통 코드 변경 PR**을 언제·누가 할까? | 정민의 E0 PR 병합 후, `repo_map.md` 4-5절 1~3번을 공통 코드 PR로 제안 (정민 리뷰). 그 전까지 SigLIP2는 `u_of`·`HIGHRES`·`topk_metrics` 등 모델 무관 부분만 import | CLAUDE.md 4절 |
| Q10 | **fake-quant 정책** (2절 #1): D-005와 PRD R5 중 어느 쪽? | 정민과 다시 합의. PRD를 따르면 G3 스크리닝도 정수 커널로 해야 해서 시간이 더 든다 | 충돌 |
| Q11 | **일정 조정**(1절 표)을 승인할까? G3 범위를 줄일까? | 1절 제안 | |
| Q12 | D-005·D-006의 "날짜는 첫 세션에서 기입" — 확정 날짜를 언제로 적을까? | 합의한 날짜를 알려 주세요 (Claude가 임의로 적지 않음) | |
| Q13 | 이 PC가 VNNI가 있어 비-VNNI 포화를 재현할 수 없음 — 정민에게 알릴까? | PRD E0 ①이 준혁 PC 확인을 요청했으므로 알리는 것을 제안. 학과 PC 등 다른 후보 필요 | 6절 (`repo_map.md`) |
| Q14 | RN50 쪽 문서에서 발견한 불일치 2건을 정민에게 알릴까? (① `rn50/README.md`의 Pets 저장 경로 ≠ 코드 경로, ② PRD 59행 "DFC 미설치" 문구가 오래됨) | 알리기만 하고 고치지 않음 (RN50 쪽 파일은 정민 담당) | `repo_map.md` 3·4-2절 |

## 4. 답변 결과 (2026-10-13, 쭈리)

| 질문 | 답 | 반영 |
|---|---|---|
| "RN50과 중복 안 되게"의 범위 | 코드: import만, 복사 금지 | D-009 |
| Q10 fake-quant | PRD R5 (정수 커널만, fake는 디버깅만) | D-008 (D-005 폐기) |
| Q1 측정 PC | 지금은 이 PC, 필요하면 쭈리가 정민에게 요청해 정민 노트북 | D-010 |
| Q4 DFC 머신 | 이 PC에 Ubuntu-22.04 새로 (D 드라이브) + DFC 5.4.0, 첫 1회 정민과 교차 확인. WSL 메모리 12GB/스왑 16GB | D-011 |
| Q11 일정 | 날짜 유지, 자동화로 따라잡기. G1 판정은 기한 없이 결과 나오는 대로 | D-014 |
| Q8 결과 위치 | `results/siglip2/<단계>_<주제>/` | D-012 |
| Q9 공통 코드 | 어댑터 먼저, 공통 코드 PR은 E0 뒤 | D-009 |
| Q3 데이터 경로 | 제안대로 | D-013 |
| 2절 #2·#3·#4·#15 PRD 정렬 | Claude 제안 방향으로 위임 → 네 가지 모두 PRD에 맞춤 | D-017 |
| Q2·Q5 GPU·실기기 | 예선은 PC(CPU)만, 실기기는 예선 통과 후 지급되는 Pi 5 + Hailo-10H | D-015 |
| Q7 대용량 파일 | 제출본 Release, 중간은 드라이브 | D-016 |
| Q6 라벨 | 5개 생성 (마일스톤 없음) | 완료 |
| Q13·Q14 정민 공유 | PR #3 본문에만 적기 | PR #3 |
| Q12 D-006 날짜 | 미답 | 남음 |
