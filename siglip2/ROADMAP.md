# SigLIP2 트랙 로드맵 (G0~G7)

기준일 2026-10-07. 예선 제출 11/1 23:59 (내부 마감 10/31). 날짜가 밀리면 이 파일을 고치고 DECISIONS에 이유를 남긴다.
각 단계는 "완료 기준"을 3중 확인(VERIFICATION.md)까지 통과해야 끝난 것이다.

| 단계 | 기간 | 핵심 결과 |
| --- | --- | --- |
| G0 준비 | 10/7~10/8 | 공통 조건 확인, 환경, 저장소 구조 |
| G1 컴파일 게이트 | 10/8~10/12 | SigLIP2 계속 / ViT-B/32 전환 판정 |
| G2 FP32 기준치 | 10/10~10/14 | 결과표 S-A행 |
| G3 데스크탑 양자화·민감도 | 10/13~10/19 | 민감도 지도, S-B0·S-B행 |
| G4 Hailo DFC 혼합 정밀도 | 10/15~10/23 | S-C·S-D행, 최종 HEF |
| G5 실칩 실측 | 10/20~10/24 | S-E행 |
| G6 동결·재현·보고서 절 | 10/23~10/27 | test 1회 평가, 재현 테스트, 보고서 절 |
| G7 통합·제출 | 10/28~10/31 | 팀 보고서 제출 |

## 결과표 (SigLIP2)

| 행 | 구성 | 환경 | 쓰임 |
| --- | --- | --- | --- |
| S-A | FP32 | 데스크탑 PyTorch, ONNX Runtime | 기준점 |
| S-B0 | ONNX Runtime INT8 전부 (MinMax) | 데스크탑 CPU, 정수 커널 | 단순 양자화 손실 |
| S-B | ONNX Runtime 혼합 정밀도 최종 | 데스크탑 CPU, 정수 커널 | 본문 주 수치 (재현 경로) |
| S-C | DFC 기본 INT8 | DFC 에뮬레이터 | Hailo 기본 대비 |
| S-D | DFC 최적 (16비트 보호, 부분 4비트, 보정·최적화 레벨) | DFC 에뮬레이터 | 기법 깊이 |
| S-E | S-D의 HEF | Orin + Hailo-10H 실칩 | 부록, 이식 결과 |
| S-G | 같은 HEF | Pi 5 + Hailo-10H | 본선 |

열은 RN50 결과표와 같게 한다.

## G0. 준비 — 10/7~10/8

- [ ] 저장소 구조 파악: RN50 트랙의 디렉터리, 공통 코드 위치, PRD.md, requirements, 데이터셋 경로 규칙을 읽고 `siglip2/docs/repo_map.md`로 정리
- [ ] RN50의 공통 코드를 import해서 모델만 바꿔 끼울 수 있는지 확인. 어려우면 필요한 변경을 목록으로 만들어 사람에게 보고 (직접 고치지 않음)
- [ ] 환경: open-clip-torch 2.31 이상, timm 1.0.15 이상. onnx·onnxruntime은 RN50 트랙과 같은 버전
- [ ] `siglip2/scripts/env_info.py`: CPU 모델, 명령어 플래그(avx2, avx512f, avx512_vnni, avx_vnni), 코어 수, OS, 패키지 버전을 `env.json`으로 저장
- [ ] Hailo Model Zoo의 Hailo-10H 문서에서 SigLIP 계열 공식 HEF가 있는지 확인하고 결과를 DECISIONS에 기록
- [ ] DFC가 설치된 머신과 버전, Pi OS hailo-h10-all 버전을 사람에게 확인
- [ ] 라벨 `siglip2` 등과 마일스톤 G0~G7 생성 여부를 사람에게 확인 후 생성

완료 기준: SigLIP2를 불러와 이미지 한 장의 임베딩이 나오고, PyTorch 기준 FP32 zero-shot이 CIFAR-100 dev 100장에서 돈다.

## G1. 컴파일 게이트 — 10/8~10/12

목적: 튜닝에 시간을 쓰기 전에 SigLIP2가 Hailo-10H에 올라가는지 판정.

- [ ] 이미지 인코더만 ONNX export (opset 17, 입력 1×3×224×224 고정). PyTorch 대비 임베딩 코사인 ≥ 0.9999, 최대 절대 오차 기록
- [ ] ONNX 연산 목록 표: patch embedding, LayerNorm, attention(QKV, QKᵀ, Softmax, AV, proj), MLP의 GELU 종류(tanh 근사/erf), pooling head 구조
- [ ] DFC parse → optimize(기본, 보정 64장) → compile (Hailo-10H). 성공 여부, 미지원 연산, 컨텍스트 수, 호스트로 떨어진 구간, 컴파일 시간, DFC 버전 기록
- [ ] FP32 빠른 확인: CIFAR-100 dev 1,000장 + 고화질 1종 dev, 공개 보고치와 대조
- [ ] 손실 감 잡기: ONNX Runtime INT8 전부(MinMax), 정수 커널, dev 1,000장

판정 (10/12, 사람이 결정):

| 결과 | 다음 |
| --- | --- |
| 전 구간 컴파일 성공 | G2 진행 |
| pooling head에서만 실패 | head 직전에서 그래프를 잘라 head는 호스트 실행, 비용 기록 후 진행 |
| 블록 내부에서 실패 | ViT-B/32로 전환 (공식 HEF·실칩 동작 확인된 Transformer), G2~G7을 ViT-B/32로 |

## G2. FP32 기준치 — 10/10~10/14

- [ ] 프롬프트 템플릿 결정 (CLIP 공식 vs SigLIP 계열 권장, dev 비교)
- [ ] 7개 데이터셋 dev 전체 FP32 정확도. test는 기준치 기록용 1회
- [ ] FP32 이미지 임베딩 저장 (충실도 비교용)
- [ ] 파라미터 수, GMACs, ONNX 크기, peak 메모리, 지연 (공식 측정 PC, 팀 속도 스크립트)

완료 기준: S-A행이 RN50 FP32 행과 같은 열로 채워진다.

## G3. 데스크탑 양자화와 민감도 지도 — 10/13~10/19

먼저 확인: ONNX Runtime이 LayerNorm, Softmax, GELU를 실제로 정수로 계산했는지 양자화된 그래프의 노드 목록으로 확인한다.
Hailo는 모든 연산을 8/16비트로 계산하므로 두 런타임의 민감도 지도는 다를 수 있고, 그 차이도 분석 결과다.

- [ ] 보정 방식 비교: MinMax / Percentile / Entropy, 보정 256장 (장수 assert)
- [ ] 블록 단위 민감도: patch embedding, 블록 1~12, pooling head를 하나씩만 INT8 (14회)
- [ ] 연산 유형 민감도: QKV / attention 행렬곱 / proj / MLP fc1·fc2 각각만 INT8
- [ ] 활성값 이상치: 블록별 최대값 / 99.9% 분위수 비율
- [ ] 누적 구성: 둔감한 곳부터 INT8을 넓혀 크기·지연·정확도 곡선, 균형점 2개 선정
- [ ] 스크리닝은 fake-quant 허용, 상위 5개는 정수 커널로 재측정 + 순위 상관 기록

완료 기준: 10/19까지 S-B 구성 확정, 새 venv에서 같은 정수 커널 정확도가 재현된다.

## G4. Hailo DFC 혼합 정밀도 — 10/15~10/23

- [ ] S-C: 기본 INT8, 최적화 레벨 비교
- [ ] DFC 층별 노이즈 분석으로 층별 SNR → Hailo 민감도 지도, G3 지도와 비교
- [ ] 16비트 보호: Softmax, LayerNorm, attention 행렬곱, pooling head 중 SNR 낮은 곳
- [ ] 4비트: MLP fc1·fc2부터 둔감한 블록 순으로 a8_w4 확대 → 정확도-크기 곡선
- [ ] 보정 사진 수 64 / 256 / 1024
- [ ] 구성마다 에뮬레이터 dev 정확도, 컨텍스트 수, HEF 크기

막히면: 4비트가 무너지면 INT8 + 16비트 보호를 최종으로, 4비트는 한계 분석.

## G5. 실칩 실측 — 10/20~10/24

- [ ] S-E: 실칩 test 정확도, 에뮬레이터와 차이
- [ ] 지연 p50/p99 (monitor off), cold/hot, 컨텍스트 수와 지연 관계
- [ ] 같은 조건의 RN50 HEF와 비교
- [ ] 여력 있으면 QD/SB 스윕, J/inference

최소 범위 (중간고사와 겹칠 때): 실칩 정확도와 단일 지연. run 수 정의는 시작 전에 팀과 확정.

## G6. 동결·재현·보고서 절 — 10/23~10/27

- [ ] 10/27 최종 구성 동결, test 1회 평가 (정수 커널)
- [ ] U8S8 포화 점검: 최종 INT8 모델을 `reduce_range` 또는 S8S8 대안과 비교하고, 가능하면 VNNI 없는 CPU에서도 실행해 정확도 일치 확인
- [ ] 스크립트를 팀 저장소 구조에 맞춰 정리, README에 SigLIP2 재현 명령어 한 줄과 보고서 표 번호 대응
- [ ] 코드를 작성하지 않은 팀원이 새 환경에서 재현
- [ ] 보고서 절 초안: 선정 사유, 민감도 지도(ORT vs Hailo), 결과표, CNN vs Transformer 비교 그림

## G7. 통합·제출 — 10/28~10/31

- [ ] 팀 보고서 통합, 표·그림 최종화, 10/31 제출
- [ ] 11/2~11/5 연락 가능 상태 유지

## 본선 메모

- 키트 수령 직후 Pi의 HailoRT 버전에 맞는 DFC로 재컴파일, S-G 측정
- Orin vs Pi 지연 분해(NPU / 전송 / 호스트)에 SigLIP2 추가
- 데모 기본은 RN50, SigLIP2는 비교 시연 옵션
