# AGENTS.md

이 저장소에서 작업하는 모든 AI 코딩 도구(Claude, Codex, Cursor, Copilot 등)를 위한 안내다.

**규칙 전체는 [`CLAUDE.md`](CLAUDE.md)에 있다. 작업 전에 `CLAUDE.md`와 [`PRD.md`](PRD.md)를 먼저 읽고 그대로 따른다.**

가장 중요한 다섯 가지:

1. 현재 실험 계획은 `PRD.md`다. `archive/` 수치는 쓰지 않는다.
2. `main`에 직접 커밋하지 않는다. 실험마다 `exp/<실험ID>-<이름>` 브랜치를 쓰고 PR로 합친다.
3. 새 양자화 기법은 **기본값이 꺼진 옵션**으로 추가한다. 옵션을 끄면 기존 결과가 그대로 나와야 한다.
4. 최선 설정은 `configs/<모델>_best.json`만 바꾼다. 같은 조건의 비교표(현재 best / +A / +B / +A+B)와 팀 확인이 있어야 한다.
5. 정확도·속도는 ONNX Runtime 정수 커널(기본 설정)로만 측정한다. 결과에 git 커밋 해시를 기록한다.
