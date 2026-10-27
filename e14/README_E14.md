# E14 — SigLIP2 B/32 예선 ORT 그래프 최적화 재현 안내

## 모델
| 이름 | 내용 | sha256 앞 8자리 |
|---|---|---|
| O | 현재 예선 제출본 O-AB (C3 세션 설정으로 실행) | `7a01044c` |
| F2 | O + GELU 융합 (`com.microsoft.FastGelu`, 13곳, E14-2) | `03f4de18` |
| P2 | F2 + 패치 임베딩 Conv → 자르기 + INT8 MatMul (E14-4, **채택 후보**) | `e465d10d` |
| P1 / P2b | E14-4 대조군 / 변형 (P1 탈락, P2b = P2와 비트 일치) | `450db00f` / `6df4ac09` |

양자화 값은 모두 같고 ORT 그래프 표현만 다르다. **본선 HEF에는 쓰지 않는다** (FastGelu는 Hailo DFC 비표준 연산, 6차원 Reshape·Transpose는 Hailo 파싱/속도 위험. HEF는 FP32 ONNX → DFC 경로 유지).

## 결과 (정민 노트북 Core Ultra 7 258V, ORT 1.25.0, O 대비 P2)
| 항목 | O | P2 | 변화 |
|---|---|---|---|
| 4T p50 | 16.36 ms | 14.61 ms | −10.3% |
| 1T p50 | 38.72 ms | 34.33 ms | −11.6% |
| peak 메모리 | 193.2 MiB | 161.2 MiB | −16.6% |
| 고화질 top-1 | 89.67% | 89.88% | +0.21%p (CI상 차이 없음) |
원자료: `results/E14-4_패치임베딩MatMul/` (계획서 v3 + 11절 결과).

## 다른 PC에서 재현
**1. 환경** (Windows x64, Python 3.11)
```bat
python -m venv e14venv
e14venv\Scripts\pip install -r e14\requirements_e14.txt
```
**2. 데이터 받기** (GitHub에 없음, 약 1.2 GB — 팀 드라이브로 공유): O ONNX 1개 + `dev256\`, `clf\` 폴더. 받은 뒤 `e14\E14_DATA_SHA256.txt`와 sha256 비교 (`certutil -hashfile <파일> SHA256`).

**3. 경로 지정** (기본값은 정민 노트북 경로)
```bat
set E14_O_ONNX=D:\e14data\onnx\siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
set E14_ART=D:\e14data            & rem 아래에 dev256\, clf\
set E14_WORK=D:\e14data\e14        & rem 만든 ONNX 저장
set E14_OUT=D:\e14data\results_%COMPUTERNAME%   & rem 원본 결과에 섞지 않기 (cmd에서는 영문 경로 권장)
set OPENBLAS_NUM_THREADS=1
set PYTHONIOENCODING=utf-8
```
**4. 모델 만들기** (O에서 F2·P1·P2·P2b 생성, sha가 원본과 같은지 출력)
```bat
e14venv\Scripts\python e14\make_models_e14.py
```
**5. 측정** (다른 프로그램 끄고 전원 연결. 각 단계는 앞 단계 결과 파일을 쓴다)
```bat
set PY=e14venv\Scripts\python -u e14\run_e14_4.py
%PY% --step 0
%PY% --step rewrite
%PY% --step g2
%PY% --step g3 --session 1 & timeout /t 120 & %PY% --step g3 --session 2
%PY% --step acc
%PY% --step mem
%PY% --step speed --session 1 & timeout /t 120 & %PY% --step speed --session 2 & timeout /t 120 & %PY% --step speed --session 3
%PY% --step summary
```
약 30분. 결과는 `%E14_OUT%\summary.json`.

## 비교 방법
- **정확도**: 노트북과 같아야 한다 (같은 ORT 버전·x86이면 같은 값이 기대됨. 다르면 CPU 명령어(VNNI 등) 차이 → 보고).
- **속도·메모리**: CPU가 다르면 절댓값이 달라진다. **O 대비 %**로 비교한다 (summary는 F2 대비, O 대비는 `abs` 값으로 계산).
- 4T 이상 코어가 없는 PC는 speed 단계에서 1·2T만 의미 있다.
- 지연 입력은 `C:\s2art\cache\pets_dev_x.npy`(224 캐시)가 없으면 dev256 첫 장으로 대체된다 (`speed.jsonl`의 `input_src`로 확인). 지연은 입력 내용과 거의 무관.
- 환경 기록은 `oslab_s2` 저장소가 없으면 간이판(`env_source: EnvLite`)으로 남는다.
