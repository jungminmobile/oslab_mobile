# SigLIP2 트랙 — Claude Code 작업 지침

이 파일은 `siglip2/` 디렉터리에서 작업하는 Claude Code가 매 세션 따르는 규칙이다.
최종 결정권자는 사람(쭈리, 손준혁)이다. Claude Code는 실행·검증·기록을 맡고, 결정은 제안까지만 한다.

## 1. 프로젝트 요약

- 대회: 2026 모바일공학전공 하반기 공모전 트랙2, **자유주제**. 예선 제출 11/1 23:59 (내부 마감 10/31).
- 팀 주제: CLIP 계열 VLM의 이미지 인코더를 양자화해 Raspberry Pi 5 + Hailo-10H(본선 지급)에 올린다.
- 트랙 분담:
  - RN50(CNN) 트랙: 팀원 정민. 저장소의 다른 디렉터리. **건드리지 않는다.**
  - SigLIP2(Transformer) 트랙: 이 디렉터리. 모델 `timm/ViT-B-16-SigLIP2` (224 고정 해상도).
- 최종 서사: 두 트랙을 같은 조건으로 재서 "CNN과 Transformer는 양자화에 민감한 위치가 다르다"를 보인다.
  그래서 RN50 트랙과 조건을 맞추는 것이 이 트랙의 품질 그 자체다.

## 2. 세션 시작 루틴 (매번, 순서대로)

1. `git switch main && git pull` 후 `git status`가 깨끗한지 확인. 작업 브랜치로 이동.
2. 이 파일, `ROADMAP.md`, `DECISIONS.md`, `PROGRESS.md`의 마지막 항목을 읽는다.
3. `gh issue list --label siglip2 --state open` 으로 열린 작업 확인.
4. RN50 트랙의 최근 변경 확인: `git log --oneline -10 -- <RN50 디렉터리>` (공통 조건이 바뀌었는지).
5. 사람에게 3줄로 보고: 현재 단계 / 오늘 할 일 / 사람 결정이 필요한 것. 그 다음 시작.

## 3. 절대 규칙

- 보고 정확도는 **정수 커널**(제출할 INT8 모델을 ONNX Runtime으로 실제 실행한 값)만 쓴다.
  fake-quant는 선택 옵션이며 **많은 구성의 1차 스크리닝**에만 쓴다. 상위 후보(기본 5개)는 반드시 정수 커널로 재측정하고,
  두 방식의 순위 상관(Spearman)을 같이 기록한다. (DECISIONS D-005)
- 정수 커널 정확도는 CPU 명령어 집합에 따라 달라질 수 있다. 모든 결과에 CPU 모델과 `avx2 / avx512f / avx512_vnni / avx_vnni` 플래그,
  ONNX Runtime 버전을 기록한다. 활성 UInt8 + 가중치 Int8(U8S8) 구성은 VNNI가 없는 x86에서 포화로 정확도가 떨어질 수 있으므로,
  최종 후보는 G6에서 `reduce_range` 또는 S8S8 대안과 비교해 학술부 PC에서도 같은 값이 나오는 형식을 정한다.
- test 분할은 최종 1회만 평가한다. 설정 비교와 선택은 dev 분할로만 한다.
- 보정 사진은 train에서 dev와 겹치지 않게 뽑고, **실제로 보정에 쓰인 장수를 로그로 assert** 한다
  (RN50 트랙의 "256장 중 16장만 사용" 버그 재발 방지).
- 속도는 1회 측정 결과를 쓰지 않는다. 예열 후 비교 모델을 번갈아 5회 반복한다 (RN50 트랙과 같은 스크립트).
- 실행하지 않은 결과, 추정치, 기억에 의존한 수치를 결과처럼 적지 않는다. 모든 숫자는 raw 파일 경로와 커밋 해시를 함께 남긴다.
- 결과를 본 뒤에 기준을 바꾸지 않는다. 바꿔야 하면 DECISIONS에 이유를 적고 사람 승인을 받는다.
- 실패와 이상치도 숨기지 않고 PROGRESS에 기록한다.

## 4. RN50 트랙과 맞출 공통 조건

공통 코드(데이터셋 로더, 분할, 프롬프트, 평가, 속도 측정)는 RN50 트랙의 것을 **import해서** 쓴다. 복사해서 고치지 않는다.
공통 코드를 바꿔야 하면 별도 PR을 열고 팀원 리뷰를 받는다.

| 항목 | 조건 |
| --- | --- |
| 범위 | 이미지 인코더만 양자화. 텍스트 분류기는 FP32로 한 번 계산해 캐시 |
| 데이터셋 | CIFAR-100, Oxford Pets, Flowers-102, Stanford Cars, Food-101, Caltech-101, ImageNetV2 (RN50과 같은 소스·같은 분할·같은 seed) |
| 지표 | top-1 (헤드라인 CIFAR-100, 설정 선택은 고화질 5종 dev 평균), FP32 대비 이미지 임베딩 코사인, 모델 크기, 지연, peak 메모리, 파라미터·GMACs |
| Caltech | background·얼굴 클래스 처리를 RN50과 동일하게 |
| 측정 PC | 팀 공식 측정 PC (미정이면 사람에게 확인하고, 그 전 수치는 "비공식"으로 표시) |

SigLIP2만의 차이: 점수는 sigmoid(logit_scale × cos + logit_bias)지만 top-1 순위는 코사인 순위와 같다.
프롬프트 템플릿은 CLIP 공식 템플릿과 SigLIP 계열 권장 템플릿을 dev에서 비교해 정한다 (DECISIONS에 기록).

## 5. 3중 확인

모든 코드·결과·결정은 `VERIFICATION.md`의 3단계를 거친다. 요약:

1. 자기 검증: 실행하고, 기대 불변식(shape, 개수, 범위)을 assert로 확인.
2. 독립 재계산: 핵심 숫자를 다른 경로로 한 번 더 구한다 (예: PyTorch vs ONNX, 두 도구로 파라미터 수, raw에서 요약 재계산).
3. 독립 검토: 작업한 맥락을 모르는 서브에이전트(또는 새 세션)에게 diff와 raw 결과만 주고 체크리스트로 검토시킨다.
   결정 사항은 여기에 더해 사람 승인.

3단계를 마치지 않은 것은 "완료"로 표시하지 않는다. PR 본문에 세 단계의 기록을 남긴다.

## 6. GitHub 기록 규칙

- 브랜치: `siglip2/<단계>-<주제>` (예: `siglip2/g1-onnx-export`). main 직접 push 금지, force push 금지.
- 작업 단위: Issue 생성 → 브랜치 → PR. PR은 사람이 확인 후 병합한다.
- PR 본문 형식: 목적 / 변경 / 실행 명령 / 결과(표) / 3중 확인 기록 / 남은 문제.
- 라벨 `siglip2`를 모든 Issue·PR에 붙인다. 그 외 `experiment`, `decision`, `verification`, `blocked`.
  라벨·마일스톤이 저장소에 없으면 만들기 전에 사람에게 확인한다 (공유 저장소 설정이므로).
- `DECISIONS.md`: 결정마다 항목 추가. 상태는 제안 → 확정 / 폐기. 확정은 사람 승인 후에만.
- `PROGRESS.md`: 세션이 끝날 때마다 항목 추가 (한 일, 결과, 막힌 것, 다음 할 일).
- 결과 파일: `siglip2/results/<날짜>_<실험명>/` 에 raw(jsonl/csv), 요약 md, 실행 명령, 환경 정보(`env.json`), 커밋 해시.
- 커밋하지 않는 것: 데이터셋, 가중치, ONNX, HEF 등 대용량 파일. `.gitignore`에 넣고 경로와 sha256만 기록한다.
  최종 제출 모델의 공유 방법(Release, 드라이브 등)은 사람이 정한다.

## 7. 반드시 멈추고 사람에게 물을 것

- 게이트 판정(G1), 모델·해상도·범위 변경, 평가·측정 프로토콜 변경
- RN50 트랙 디렉터리, 공통 코드, 저장소 설정(라벨, 브랜치 보호, Actions) 변경
- main 병합
- 결과가 예상과 크게 다를 때 (원인 가설 2개 이상과 함께 보고)
- 같은 문제로 2회 이상 실패했을 때 (세 번째 시도 전에 보고)
- 실기기(Jetson Orin + Hailo-10H) 작업: 펌웨어·전력 모드·클럭 등 **기기 설정 변경 금지, sudo 금지**, 홈 디렉터리 안에서만 실행.
  실기기 접근 방법은 사람에게 확인한다.

## 8. 팀이 이미 겪은 함정

- ORT `MinMaxCalibrater`는 `CalibMaxIntermediateOutputs`를 넘는 배치를 버린다 → 보정 장수 assert 필수.
- CIFAR-100은 해상도가 낮아 초기 층 변화에 과민하다 → 설정 선택 지표로 쓰지 않는다.
- 같은 모델끼리도 1회 속도 측정 차이가 크게 났다 → 반복 측정.
- `hailortcli monitor`를 켜면 지연 꼬리가 오염된다 → 지연 측정 시 끈다.
- 호스트 OpenBLAS 스레드 스핀 → `OPENBLAS_NUM_THREADS=1`.
- 본선 Pi OS의 HailoRT(2026년 6월 기준 5.1.1)와 연구실(5.3.0) 버전이 다르면 HEF가 실행되지 않을 수 있다 → DFC 버전을 기록하고, Pi 버전용 컴파일 경로를 확보.

## 9. 참고 문서

- `ROADMAP.md` 단계별 계획 (G0~G7)
- `VERIFICATION.md` 3중 확인 체크리스트
- `DECISIONS.md` 결정 로그, `PROGRESS.md` 진행 로그
- 팀 실험 기록(읽기 전용, 노션 블록 한도로 더 이상 쓰지 않음): 노션 "MWC 가야지 > 실험 로그"

## 10. 로컬 작업 환경 (Windows, D:\mokerton, Claude 데스크탑 앱 Code 탭)

이 PC는 Windows이고, Claude 데스크탑 앱의 Code 탭에서 Local 환경으로 실행된다. 프로젝트 폴더는 `D:\mokerton\oslab_mobile\siglip2`.
셸 명령은 Git Bash로 실행되므로 bash에서는 `/d/mokerton/...`, Python 코드에서는 환경 변수나 `pathlib`로 경로를 다룬다.

| 위치 | 용도 | 저장소에 커밋 |
| --- | --- | --- |
| `D:\mokerton\oslab_mobile\` | 팀 저장소 (코드, 문서, 작은 결과 파일) | O |
| `D:\mokerton\venvs\siglip2\` | Python 가상환경 (`SIGLIP2_PYTHON`) | X |
| `D:\mokerton\data\` | 데이터셋 (`SIGLIP2_DATA`) | X |
| `D:\mokerton\artifacts\siglip2\` | ONNX, HEF, 보정 캐시, 임베딩 등 대용량 (`SIGLIP2_ARTIFACTS`) | X (경로와 sha256만 기록) |
| `D:\mokerton\cache\` | Hugging Face(`HF_HOME`), torch(`TORCH_HOME`), pip(`PIP_CACHE_DIR`) 캐시 | X |

환경 변수는 `siglip2/.claude/settings.local.json`의 `env`로 매 세션 자동 설정된다. 이 파일은 개인 설정이라 커밋하지 않는다(`.git/info/exclude`에 등록됨).

규칙:
- Python은 항상 가상환경의 실행 파일로 실행한다: `"$SIGLIP2_PYTHON" script.py`, 패키지 설치는 `"$SIGLIP2_PYTHON" -m pip install ...`.
  그냥 `python`이나 `pip`를 쓰지 않는다 (시스템 Python으로 실행될 수 있음). 설치 후 `pip freeze`를 결과 폴더에 남긴다.
- 세션 시작 시 `echo $HF_HOME $SIGLIP2_DATA $SIGLIP2_ARTIFACTS $SIGLIP2_PYTHON`으로 확인하고, 비어 있으면 작업하지 말고 사람에게 알린다.
- 모든 다운로드, 캐시, 산출물은 `D:\mokerton` 아래에만 쓴다. C 드라이브의 사용자 캐시 폴더에 쓰지 않는다.
- 데이터 경로는 환경 변수에서 읽는다. 팀원 코드가 경로를 하드코딩하고 있으면 그 코드를 고치지 말고 인자·설정으로 넘기거나, 필요한 변경을 사람에게 보고한다.
- 2GB 이상 받기 전에 D 드라이브 여유 공간을 확인하고 예상 용량을 사람에게 알린다.
- Hailo DFC는 Linux에서만 동작한다. 이 PC에 WSL2 + DFC가 있으면 `wsl -e bash -lc "..."`로 호출하고 경로는 `/mnt/d/mokerton/...`을 쓴다.
  DFC를 어느 머신에서 돌릴지는 사람에게 먼저 확인한다.
- 이 PC에서 잰 속도는 팀이 공식 측정 PC로 정하기 전까지 "비공식"으로 표시한다.
