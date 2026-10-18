# G3RQ 실칩 비교: rothad_D256L1 vs M2rot_D256L1 (AGX Hailo-10H, 설계 10/18 — 결과 전에 고정)

> 상태: **설계만, 실행 전.** 용 요청. 상위 설계 `../README.md` 4절 "다음 단계".

## 1. 비교
- A = **rothad_D256L1** (현재 Hailo 최종, D-042, HEF sha256 `7243b76c…`, 107.5 MB, 17 컨텍스트)
- B = **M2rot_D256L1** (`../README.md` 6절, HEF sha256 `94fff0976ac43480…`, 110.4 MB, 21 컨텍스트)
- AGX Orin + Hailo-10H 실칩 (HailoRT 5.3.0), dev 6종 전체 (CIFAR A 1000, Pets 990, Flowers 957, Cars 967, Food 985, Caltech 1024장), 두 HEF에 같은 사진.
- 입력: 데스크탑 dev 캐시 `<set>_x.npy` (G3Q 8절에서 보낸 AGX `~/ssd/junyong/hailo_g3q/devfull/`를 sha256 확인 뒤 재사용). `s2_hailo_chip.py eval --x-npy` (FLOAT32 입출력, NHWC).
- 계산(데스크탑): FP32 기준 `ref_fp32`·클래스 임베딩 `clf`는 데스크탑 dev 캐시 (`s2_hailo_chip_eval.py --k 0`).

## 2. 지표
- 정확도: 고화질 5종 Δtop-1 (B − A) 짝 부트스트랩 95% 구간 (데이터셋별 층화, 시드 0, 1000회, `s2_paired_ci.boot_hr`), 고화질 cos, CIFAR A Δtop-1. FP32 대비 Δtop-1도 적는다.
- 지연: 배치 1, 워밍업 20, 200회 × 3세션, 두 HEF를 번갈아 (A B A B A B, `s2_hailo_chip.py bench --rounds 1`을 세션마다). 3세션 중앙값(세션별 중앙값의 중앙값).
- **온도 관리:** 각 세션(eval 2회·지연 3세션·run2 2회) 시작 전에 칩 온도(`Device.control.get_chip_temperature()`, 읽기 전용 ts0·ts1)를 읽고 **max(ts0, ts1) < 60 °C가 될 때까지 기다린 뒤** 시작한다. 정확도 eval과 지연 측정 사이에도 같은 대기. 온도는 세션 전후로 기록 (`temps.jsonl`). 측정 중에는 읽지 않는다. `hailortcli monitor` 쓰지 않음.
- 처리량: `hailortcli run2` 각 10초 (A, B).

## 3. 판정 규칙
- 정확도: 구간 **하한 > 0 → "M2rot이 유의하게 좋음"**, **0을 포함 → "동률"**, **상한 < 0 → "나쁨"**.
- 지연: 3세션 중앙값으로 rothad 대비 증가율 (B/A − 1).
- 제안:
  - "유의하게 좋음" + 지연 증가 ≤ 10% → **Hailo 최종 교체를 팀에 제안**.
  - "유의하게 좋음" + 지연 증가 > 10% → 정확도·지연 트레이드오프 표로 **팀이 결정**.
  - "동률" 또는 "나쁨" → **rothad 유지**.
- 채택은 팀 결정 (siglip2/CLAUDE.md).

## 4. 공통
- AGX: `ssh agx`, `~/ssd/junyong/` 아래만, sudo·기기 설정 변경 금지, tmux 금지(nohup + 로그). 시작 전 다른 사용자·Hailo 프로세스 확인.
- HEF 전송 뒤 양쪽 sha256 확인. 임베딩(대용량)은 저장소에 넣지 않고 sha256만 기록.

## 5. 사후 변경 (실행 전, 결과 전 — 용 승인 10/18 23:20): 세션 시작 온도 기준
- AGX 칩이 아무 작업이 없는 상태(Hailo 프로세스 0, load 0.00)에서 23:13~23:18 6분 동안 **62.5~62.6 °C로 고정**(G3Q 15:26에는 50.5 °C) → "60 °C 미만" 조건을 채울 수 없다.
- **고친 기준: 쉬는 온도 + 2 °C = max(ts0, ts1) < 64.6 °C**가 되면 세션 시작. 의도(매 세션을 같은 식은 상태에서 시작)는 그대로. 최대 90분 기다리고, 넘기면 그 세션에 "hot" 표시하고 진행해 기록한다.
