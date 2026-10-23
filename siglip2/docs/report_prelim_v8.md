<!--
초안 v8 메타 정보 (PDF로 옮길 때 이 주석 블록은 지운다)

[쓴 자료] (경로는 D:\mokerton\wt_report 기준, results/siglip2/ 는 R/ 로 줄임)
- 양식·요구사항: siglip2/docs/contest_requirements.md, scratchpad review/tmpl.txt·guide.txt (대회 원문 추출)
- 방법 설명 재사용: siglip2/docs/report_prelim_v7.md (B/16 중심 초안)
- 결정: siglip2/DECISIONS.md D-048·D-049, R/G8_candidates/DECISION_B16_vs_B32.md
- B/32 CPU: R/G8_candidates/b32/README.md, b32/G3RQ/dev_full/paired_ci.jsonl, b32/O_AB/dev_full/paired_ci.jsonl, b32/block0/README.md,
  b32/submit_prep/README.md (+ model_stats/model_stats.jsonl, H5_qdq_snr/why_after_qdq_summary.json, m2rot_fp32_devfull/paired_ci.jsonl,
  input_shift/input_shift_summary.json, seed/dev_full/paired_ci.jsonl), b32/test/README.md·test_summary.json, b32/xplat/README_bundle.md·i7-1165G7/result.json·bundle_SHA256SUMS.txt,
  b32/G1_onnx_export/export.jsonl·onnx_ops.json
- 지연·메모리: R/G8_candidates/latency_this_pc/all8_20261022/cross_latency.json, all8_20261022_rerun_t24/cross_latency_recovered.json
  (이상치 규칙은 R/G8_candidates/b16/O_AB/README.md 2절 — 세션 값에서 직접 다시 계산해 README 표와 같음을 확인),
  latency_this_pc/b32_submit_20261022/cross_latency.json (저녁, 발열 — 메모리·비율 교차 확인용으로만)
- 기전: R/G6_pre/README.md (B/32 거대 활성값·sink 위치·attention 질량)
- Hailo·O-AB (준용 브랜치, git show): origin/siglip2/g6e-b16-g:R/G6b/README.md, origin/siglip2/g6d-speed2:R/G6d/README.md,
  origin/siglip2/g6f-ceiling:R/G6f/README.md (GF·ALL8F — 파일에서 확인됨), origin/siglip2/o1-ort-speed:R/O1/README.md, origin/siglip2/o2-unified-b16:R/O2/README.md, R/G6/README.md
- B/16 일반화: R/G8_candidates/b16/A1_ablation/README.md, b16/O_AB/README.md, b16/E5_calib_seed/README.md, R/G6_m2rot_ort/test/test_summary.json
- 제출 ONNX 입출력·opset: 로컬 파일 D:\mokerton\artifacts\siglip2_b32\onnx\*.onnx 를 onnx로 읽어 확인 (읽기만)
- Release 목록: gh release view siglip2-b32-submit-20261023 (읽기만)

[확인 필요로 남긴 것] 본문의 [확인 필요: …] 전부. 요약:
 팀명·학번·준용 성명·대표 연락처·제출일 / 저장소 공개 방법 / prelim-submit 태그 / 최종 제출 Release 태그 / 모델 라이선스 /
 B/32 HEF의 호스트 실행 층 0개 문구 / 본선 Pi HailoRT 버전 / 실칩 결과를 예선 보고서에 넣어도 되는지(학술부) /
 채점 PC·실행 방식(onnxruntime 사용 여부, 학술부 6번) / 다른 PC(VNNI 없는 x86·ARM) 재현 결과(정민·준용) /
 siglip2/README.md B/32 갱신 / 새 환경에서 9장 B/32 명령 그대로 실행 확인 /
 디스크 필요량(B/32 기준) / 참고문헌 5번 채택·9~17 서지 확인 (본선 HEF·O-A Hailo 대응은 미정 사항으로 본문 3.3·7.1·7.2에 적음)

[불확실했던 점]
- 지연 대표값: 낮 all8 측정(1T 3세션, 2T 5세션, 4T 4세션 — 이상치 규칙 적용)을 썼다. 변화율은 "세션마다 같은 실행 안 비율의 중앙값"(README 규칙)이라
  중앙값 ms끼리 나눈 값과 0.1~0.7%p 다르다(4T −56.9% vs −56.2%). 2장 각주에 적었다.
- 메모리: 낮 측정 453.7 → 193.0 MiB(D-049 문장은 저녁 측정 453.9 → 192.8). 둘 다 −57.5%. 본문은 낮 값.
- O-A/O-B 단독 효과는 준용 데스크탑(285K, 같은 바탕 그래프) 값이다. 이 PC의 O-AB−M2rot(+0.52)은 바탕 FP32 빌드가 달라 단독 효과로 쓰지 않았다.
- B/32 오류 클래스 분석(B10)은 O-AB가 아니라 이전 주 후보(M2rot) 기준이다. 6.6에 그렇게 적었다.
- 보정 시드 흔들림(폭 0.10)은 M2rot(이 PC 빌드) 기준이고 O-AB로는 재지 않았다.
- B/16 test −0.34는 M2rot_ort 파일 값이다(B/16 O-AB는 test 없음). 10장에 구분해 적었다.
-->

# 2026학년도 모바일공학전공 하반기 공모전 [트랙 2] 예선 결과 보고서 — 자유주제

## 1. 팀 정보

| 구분 | 내용 |
|---|---|
| 팀명 | [확인 필요: 팀명] |
| 선택 주제 | 자유주제 — 자유 모델 최적화 |
| 대상 모델명 | SigLIP2 ViT-B/32 (256×256) 이미지 인코더 (`timm/ViT-B-32-SigLIP2-256`) |
| 팀원 (성명 / 학번 / 역할) | 손준혁 / [확인 필요: 학번] / 원인 분석, CPU(ONNX Runtime) 양자화·평가, 측정·재현, 보고서 |
| | [확인 필요: 준용 성명] / [확인 필요: 학번] / Hailo-10H 컴파일·실칩 평가, 회전·레지스터 캐시 이식, ORT 커널 최적화(O-A·O-B) |
| | 전정민 / [확인 필요: 학번] / 공통 평가 코드(데이터 분할·평가·속도 측정), 다른 PC 교차 측정 |
| 대표자 연락처 | [확인 필요: 연락처] |
| 제출일 | 2026년 10월 [확인 필요: 일] |

---

## 2. 결과 요약

정확도 지표: **제로샷 분류 Top-1 정확도, 고화질 5종 평균**(Pets·Flowers·Cars·Food·Caltech). 효율은 같은 PC에서 원본(FP32)과 같은 실행 안에서 번갈아 잰 값이다 (Intel i7-1165G7 노트북, 배치 1, ONNX Runtime 1.25.0, **비공식**). 같은 PC에서 3번 잰 개선율이 4스레드 −55.9~−56.9%, 1스레드 −60.2~−63.8%, 메모리 −57.4~−57.5%로 비슷했다(6.5, 그림 F3b).

| 항목 | Baseline (FP32) | 최적화 후 (INT8, O-AB) | 변화율 |
|---|---|---|---|
| 모델 크기 (MB) | 378.36 | 102.76 | **−72.8%** |
| 파라미터 수 (M) | 94.55 | 94.55 (그중 MatMul 가중치 92.10M을 INT8로 저장, 나머지 float. 레지스터 캐시 상수 약 0.16M 추가) | 0% |
| 연산량 (GFLOPs) | 11.50 | 11.50 (INT8 정수 연산으로 실행) | 0% |
| 추론 Latency (ms, 4스레드 / 1스레드) | 61.5 / 132.8 | 27.0 / 47.9 | **−56.9% / −63.8%**¹ |
| Throughput (FPS, 4스레드 / 1스레드) | 16.3 / 7.5 | 37.1 / 20.9 | **+128% / +177%** (2.3배 / 2.8배) |
| 피크 메모리 사용량 (MB) | 475.7 (453.7 MiB) | 202.4 (193.0 MiB) | **−57.5%** |
| 정확도 — test (Top-1, 고화질 5종) | 88.51% | 88.28% | **−0.23%p** [95% 구간 −0.36, −0.08] |
| 정확도 — dev (Top-1, 고화질 5종) | 89.96% | 89.69% | −0.27%p [−0.62, +0.09] (유의하지 않음) |

<!-- src: 크기 bytes·파라미터·GMACs = b32/submit_prep/model_stats/model_stats.jsonl, latency_this_pc/all8_20261022/cross_latency.json(files);
지연 = all8_20261022 + all8_20261022_rerun_t24/cross_latency_recovered.json 세션 값에서 이상치 규칙으로 다시 계산(1T 132.81/47.89, 4T 61.48/26.95);
메모리 = all8_20261022 peak_mem peak_wset; test = b32/test/test_summary.json; dev = b32/O_AB/dev_full/paired_ci.jsonl -->

¹ 지연 변화율은 세션마다 같은 실행 안의 비율을 낸 뒤 중앙값이다(사전 규칙). 중앙값 ms끼리 나누면 4스레드 −56.2%, 1스레드 −63.9%다. FPS = 1000 / 중앙값 지연(배치 1)이고, FPS 변화율은 중앙값 지연에서 계산했다(−56.2% / −63.9%에 해당). 주 조건은 4스레드다(6.4).

**요약.**
- 이미지-텍스트 대조 학습 모델 **SigLIP2 B/32의 이미지 인코더**를 재학습 없는 **INT8 정적 사후 양자화(PTQ)**로 최적화했다. 단순 INT8은 정확도가 무너졌다(dev −61.0%p).
- 원인(고정 위치 토큰 2개의 거대 활성값)을 찾아 **레지스터 캐시(토큰 축) + Hadamard 회전(채널 축)**으로 고쳤고, ORT 커널에 맞춘 **O-A(행렬곱 출력 재반올림 생략)·O-B(SkipLayerNorm 융합)**를 더했다. 행렬곱 79개를 **예외 없이 모두 INT8**로 했다.
- 결과: 크기 −73%, CPU 지연 약 −57%(4스레드), 메모리 −57%, test 정확도 −0.23%p. 같은 처방이 SigLIP2 B/16에서도 통했다(10장).

---

## 3. 대상 모델 및 선정 사유

### 3.1 대상 모델

| 구분 | 내용 |
|---|---|
| 모델명 / 버전 | SigLIP2 ViT-B/32, 입력 256×256 (`hf-hub:timm/ViT-B-32-SigLIP2-256`, open_clip 3.3.0 / timm 1.0.30) |
| 모델 유형 | 멀티모달(이미지-텍스트 대조 학습). 최적화 대상은 **이미지 인코더** |
| 저장소 URL | 우리 코드: https://github.com/jungminmobile/oslab_mobile [확인 필요: 현재 비공개 — 학술부 접근 방법] / 원 모델: https://huggingface.co/timm/ViT-B-32-SigLIP2-256 |
| 커밋 해시 또는 릴리스 태그 | 제출 시점 main의 `prelim-submit` 태그 [확인 필요: 태그·커밋]. 모델 파일 Release [확인 필요: 최종 제출 Release 태그] (지금은 내부용 pre-release `siglip2-b32-submit-20261023`) |
| 사전학습 가중치 출처 | Google SigLIP2 (Tschannen et al., 2025) 공개 가중치의 timm/open_clip 변환본 (Hugging Face, 로컬 캐시 스냅샷 `cd1efb47…`) |
| 라이선스 | [확인 필요: 모델 카드 라이선스, Apache-2.0로 알고 있음] |

- 구조: 패치 32×32 → 토큰 64개(8×8), 폭 768, Transformer 블록 12개, MLP GELU는 tanh 근사. CLS 토큰 없이 attention pooling(MAP) 헤드로 768차원 임베딩을 낸다.
- 파라미터 94,552,320개(PyTorch = ONNX), 5.75 GMACs = **11.50 GFLOPs**(fvcore), FP32 ONNX 360.83 MiB. <!-- src: b32/submit_prep/model_stats/model_stats.jsonl -->
- **범위**: 이미지 인코더만 양자화한다. 텍스트 쪽은 클래스 문장 임베딩을 FP32로 한 번 계산해 저장하고 다시 쓴다. 기기에서 사진마다 도는 것은 이미지 인코더뿐이다.

### 3.2 선정 사유

1. **엣지 쓰임새가 넓다.** 이미지 임베딩 하나로 제로샷 분류·이미지 검색·필터링을 한다. 클래스가 바뀌어도 재학습 없이 텍스트만 바꾸면 된다.
2. **최신 모델이고 최적화할 가치가 크다.** SigLIP2(2025)는 CLIP 계열 최신판이다. FP32 이미지 인코더가 378 MB, 노트북 CPU 1스레드에서 한 장 133 ms다.
3. **원본 대비 개선율·정확도 유지가 좋다.** 같은 레시피를 쓴 같은 계열 B/16과 같은 실행에서 비교하면 원본 대비 개선율은 거의 같고(지연 4스레드 −56.9% vs −57.9%, 크기 −72.8% vs −74.1%), 메모리는 B/32가 더 줄며(−57.5% vs −52.9%), dev 손실도 둘 다 유의하지 않다(−0.27 vs −0.18). 예선 채점이 원본 대비 개선율이므로 두 모델은 예선에서 비슷하다. <!-- src: DECISION_B16_vs_B32.md 3-1·4절, D-048 -->
4. **본선에서 벤더 기준과 직접 비교할 수 있다.** Hailo Model Zoo에 Hailo-10H용 SigLIP2 B/32 공개 HEF가 있다. 우리 방식으로 만든 HEF는 이 공개 HEF보다 **빠르면서 더 정확하다**(구성 GF: 배치 1 지연 −9.5%, 정확도 +2.67%p, 3.3절). B/16은 Zoo에 공개 HEF가 없어 이런 비교가 안 된다. 본선은 예선 모델을 그대로 이어서 써야 하므로(대회 원문 2·5.2절), 본선까지 고려해 B/32를 골랐다(팀 결정 D-048).

### 3.3 Hailo-10H 호환성 사전 검토

본선용 HEF는 예선 ONNX와 **같은 PyTorch 모델(같은 회전 행렬 `3a6391a2…`, 같은 sink 규칙 `15@6:5, 7@10:9`, 같은 레지스터 캐시 상수)**에서 배치 1로 export한 FP32 ONNX(`a0eb502b…`)를 Hailo DFC로 따로 양자화·컴파일한다. 예선 ONNX는 같은 모델을 배치 가변으로 export한 FP32(`23236123…`)에 O-B 융합을 더해 양자화한 것이다. 두 FP32 파일은 PyTorch 대비 cos 최소가 모두 0.999999997 이상이다. O-A·O-B는 ORT CPU 커널용 변경이라 HEF에는 들어가지 않는다(7.1). <!-- src: G6f/G6 export.jsonl, G6b export.jsonl·8절, O1 6-2 -->

| 항목 | 검토 결과 |
|---|---|
| 지원 연산 | Conv(패치 임베딩), MatMul, Add, Mul, Softmax, GELU(tanh), LayerNorm, Reshape/Transpose. 모두 DFC가 지원한다(LayerNorm은 DFC가 분해) |
| 비표준 층 | 없다. 레지스터 캐시는 상수 Mul·Add 마스크로 표현해 토큰 수를 유지하므로 표준 연산으로 컴파일된다(토큰을 지우는 형태는 LayerNorm 분해와 충돌해서 쓰지 않음) |
| 실제 컴파일 | Hailo DFC 5.4.0, `--hw-arch hailo10h`로 HEF 컴파일 성공(여러 구성, 아래 표). 입력은 UINT8 픽셀(NHWC) 그대로, **정규화도 HEF 안**, 출력 768차원(UINT16) <!-- src: R/G6/README.md 6-1 --> |
| CPU Fallback | 후처리 연산 없음. 호스트 정규화 불필요(HEF 안). [확인 필요: B/32 HEF `parse-hef`에서 호스트 실행 층 0개라는 문구 확인] |
| 혼합 정밀도 | 구성 G·GF: 패치 경로(`norm1`·`conv1`·`format_conversion1`)와 잔차 덧셈 26개만 16비트(a16_w16), 나머지 8비트(a8_w8). Zoo 공식 설정보다 16비트 층이 적다 |
| 실칩 검증 [확인 필요: 실칩 결과를 예선 보고서에 넣어도 되는지 학술부] | NVIDIA AGX Orin + Hailo-10H(HailoRT 5.3.0), dev 전체 5,923장 |
| INT4 | Hailo-10H는 INT4를 지원한다. CPU에서는 4비트가 느려 넣지 않았다(5.3). Hailo INT4는 본선에서 시험한다 |
| 버전 위험 | 팀 DFC 5.4.0, 본선 Pi의 HailoRT는 [확인 필요: 주최측]. 버전이 다르면 그 버전용 DFC로 다시 컴파일한다 |

**B/32 Hailo-10H 실칩 결과** (준용, dev 전체, FP32 89.96% 대비. 배치 1 지연의 Zoo 대비 비율은 각 실험 세션 안의 Z 기준 — M은 G6b, G는 G6d, Z·GF·ALL8F는 G6f 세션. 같은 G6f 세션에서 G는 15.63 ms, −5.2%)

| HEF | 고화질 Δ [95%] | Zoo 공개 HEF 대비 정확도 | 배치 1 지연 (Zoo 대비) | 크기 |
|---|---|---|---|---|
| Z: Hailo Model Zoo 공개 HEF | −4.22 [−4.95, −3.45] | — | 16.49 ms | 102.6 MB |
| M: 우리 방식, 16비트 보호 넓게 | −0.84 [−1.31, −0.37] | +3.38 [+2.62, +4.17] | 21.16 ms (+28.2%) | 113.0 MB |
| G: 16비트를 패치 경로·잔차만 | −1.55 [−2.09, −1.03] | +2.67 [+1.92, +3.46] | 15.66 ms (−5.0%) | 104.7 MB |
| **GF**: G + 컴파일 설정 2줄 | = G (출력 1,200장 비트 동일) | +2.67 (= G) | **14.93 ms (−9.5%)** | 99.9 MB |
| ALL8F: 전부 8비트 + 컴파일 설정 | −2.42 [−3.04, −1.77] | +1.80 [+1.05, +2.63] | 13.38 ms (−18.9%) | 91.8 MB |

<!-- src: origin/siglip2/g6e-b16-g:R/G6b/README.md 6-2·6-3 (Z·M), origin/siglip2/g6d-speed2:R/G6d/README.md 5-2·5-3 (G), origin/siglip2/g6f-ceiling:R/G6f/README.md 5-2·9-3 (GF·ALL8F, 같은 세션 Z 16.494) -->

- 본선 HEF는 아직 정하지 않았다(GF·ALL8F 등, 정확도와 속도의 교환점은 팀이 정함). 어느 쪽이든 Zoo 공개 HEF보다 빠르고 정확하다.

---

## 4. 실험 환경

| 구분 | 사양 및 버전 |
|---|---|
| CPU | Intel Core i7-1165G7 (4코어 8스레드, AVX2·AVX-512F·AVX-512 VNNI). 교차 측정: Intel Core Ultra 9 285K (AVX-VNNI, 준용), NVIDIA Jetson AGX Orin(Hailo 호스트) |
| GPU (모델 / VRAM) | 사용 안 함 (양자화·평가·속도 측정 모두 CPU) |
| RAM | 16 GB |
| OS | Windows 11 (빌드 26300). Hailo 작업은 Linux / WSL2 Ubuntu 22.04 |
| Python 버전 | 3.11.9 |
| 딥러닝 프레임워크 / 버전 | PyTorch 2.14.0 (CPU), open_clip_torch 3.3.0, timm 1.0.30, transformers 5.19.0 (토크나이저) |
| CUDA / cuDNN 버전 | 사용 안 함 |
| 양자화·최적화 도구 / 버전 | ONNX 1.23.1 (opset 17), **ONNX Runtime 1.25.0** (`quantize_static`, 실행 `ORT_ENABLE_ALL`), numpy 2.4.4, Hailo Dataflow Compiler 5.4.0, HailoRT 5.3.0 |

<!-- src: b32/test/env.json -->
- 결과 jsonl 한 줄마다 CPU 명령어 집합, ORT 버전, 커밋 해시, 커밋 안 된 변경 여부(dirty)를 남겼다. 보고서 수치는 모두 dirty가 아닌 실행이다.
- 지연은 노트북 고성능 모드·충전 상태에서만 쟀다. 공용 잠금으로 다른 무거운 작업과 겹치지 않게 했고, 측정 직전 CPU 30초 평균 25% 미만일 때 시작했다.

---

## 5. 적용한 최적화·양자화 기법

### 5.1 기법 요약

"단독 효과"는 같은 양자화 설정(MatMul 전부 INT8, U8 + reduce_range, Percentile 99.999, 보정 256장)에서 한 가지만 켜고 끈 비교다. 값은 dev 전체 고화질 Top-1의 FP32 대비 손실과 짝 차이 [95% 구간]이다. 예외: ①의 MinMax 결과는 연산 전부 INT8·S8, ②는 MatMul만 INT8·S8·fc2 일부 보호에서 dev 6×200 cos 스크리닝 값이다. ⑥⑦은 준용 PC(285K, WSL)에서 준용 바탕 그래프(M2rot −0.87)로 잰 값이다.

| 적용 기법 | 적용 대상 | 주요 설정값 | 단독 적용 효과 |
|---|---|---|---|
| ① 정적 PTQ (INT8) | 이미지 인코더 MatMul 79개 | ORT `quantize_static`, QDQ, 가중치 채널별 S8, 보정 256장 | 이것만 하면 **붕괴**: −60.96%p (cos 0.627). MinMax·연산 전부 INT8도 붕괴(고화질 0.4~3.8%) |
| ② Percentile 보정 | 활성값 범위 | 99.999 백분위, 비대칭, 히스토그램 16장씩 누적 | fc2를 보호한 상태 cos: MinMax 0.931, Entropy 0.937, P99.99 0.815 → **P99.999 0.963** (dev 6×200) |
| ③ 레지스터 캐시 M2 (토큰 축) | sink 토큰 15·7의 K·V와 MLP 입력 | 규칙 `15@6:5, 7@10:9`, 보정 256장 평균 상수, 토큰 64개 유지 | **−60.96 → −1.58%p** (+59.38 [+58.14, +60.72]) |
| ④ 오프라인 직교 회전 (채널 축) | 잔차 흐름 768채널 | 무작위 부호 Hadamard × Householder, LayerNorm γ·β 흡수 | 단독: −60.96 → −54.52 (+6.44, 여전히 붕괴). **M2 위에: −1.58 → −0.79** (+0.79 [+0.30, +1.34], 유의) |
| ⑤ 활성값 U8 + reduce_range | x86 정수 커널 선택 | 활성값 UInt8, 가중치 7비트 범위 | B/16에서 S8은 QKV가 정수로 합쳐지지 않아 4스레드 속도가 FP32 수준, U8은 약 2배. 정확도는 구분 안 됨 |
| ⑥ O-A: MatMul 출력 재반올림 생략 | 가중치 MatMul 53 + QKᵀ 13 | ORT `OpTypesToExcludeOutputQuantization=["MatMul"]` → `MatMulIntegerToFloat`, Softmax float | 같은 바탕: +0.67 [+0.28, +1.08], 지연 −4.3 / −6.0 / −7.7% (1/2/4스레드). 단, CIFAR A −1.1로 사전 기준 미달(단독 미채택) |
| ⑦ O-B: SkipLayerNorm 융합 | 잔차 Add + LayerNorm, bias Add, 모양 계산 | LN 26개 중 24개 → `SkipLayerNormalization`(bias 23개 흡수), Reshape 모양 상수화 28개. FP32 cos ≥ 0.99999988 | 같은 바탕: +0.20 [−0.14, +0.55], 지연 −0.7 / −1.7 / −2.7% (단독으로는 3% 기준 미달) |
| **⑥+⑦ O-AB** | 위 둘 함께 | — | 같은 바탕: **+0.57 [+0.17, +0.97]**, 지연 **−4.9 / −8.4 / −11.4%**. 이 PC 같은 실행: M2rot 대비 −5.2 / −11.1 / −14.3% |

<!-- src: ①③④ b32/G3RQ/dev_full/paired_ci.jsonl (b32/README.md B8), ① MinMax b32/README.md B3, ② b32/README.md B6,
⑤ v7 5.3 단계 5 (B/16 G3_cumulative), ⑥⑦ origin/siglip2/o1-ort-speed:R/O1/README.md 6-3·6-4·7 (285K, 비공식), 이 PC 값 b16/O_AB/README.md 2절 -->

**최종 구성(O-AB, 제출 파일) = ①②③④⑤⑥⑦.** 본선 HEF는 ③④를 적용한 같은 모델(배치 1 export)을 Hailo DFC로 혼합 정밀도 컴파일한다(3.3).

### 5.2 양자화 상세

| 구분 | 내용 |
|---|---|
| 양자화 방식 (PTQ / QAT) | **PTQ** (정적 사후 양자화, 재학습 없음) |
| 정밀도 (INT8 / INT4 / 혼합) | **INT8**: MatMul 79개 전부 입력·가중치 INT8 (가중치 채널별 INT8 7비트 범위, 활성값 UINT8). O-A로 가중치 MatMul 53개와 QKᵀ 13개는 정수 행렬곱 뒤 float 출력(`MatMulIntegerToFloat`), AV 13개는 `QLinearMatMul`. Hailo HEF는 8/16비트 혼합 |
| 캘리브레이션 데이터셋 | 고화질 5종의 **train 분할**(Flowers는 train+validation)에서 dev·test와 겹치지 않게 뽑은 보정 후보 풀 311장. 256×256 Zoo 방식 전처리 캐시(같은 사진·같은 순서) |
| 캘리브레이션 샘플 수 | **256장** (시드 0). 실제로 보정에 들어간 장수를 코드에서 assert한다(ORT 보정기가 일부 배치를 조용히 버린 사고 재발 방지) |
| 양자화 제외 레이어 | MatMul은 **제외 없음**. MatMul 밖의 연산은 float으로 계산: 패치 임베딩 Conv, LayerNorm 2개·SkipLayerNorm 24개, GELU(tanh), 잔차 덧셈, 레지스터 캐시 마스크 연산, Softmax 13개(O-A 때문에 float) |

<!-- src: O1 6-2 노드 표(B32 O-AB), 제출 ONNX를 onnx로 읽어 확인: QuantizeLinear 105, DequantizeLinear 158, MatMul 79, SkipLayerNormalization 24, LayerNormalization 2, Softmax 13, Conv 1 -->

### 5.3 적용 근거 및 과정

판정 기준은 결과를 보기 전에 정해 기록했다(`siglip2/DECISIONS.md`, 각 결과 폴더 README). 과정은 B/16에서 먼저 찾은 원인·처방을 B/32에 같은 순서로 다시 확인한 것이다.

**단계 1. 단순 INT8은 무너진다.** MinMax·모든 연산 INT8이면 고화질 Top-1 0.4~3.8%(cos 0.38~0.42). MatMul만·Percentile이어도 −60.96%p다.

**단계 2. 민감도 지도** (블록·연산 하나만 INT8, 나머지 FP32, dev 6×200, 고화질 cos). <!-- src: b32/README.md B4 -->

| 블록 | 0 | 1~4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|
| cos | **0.521** | 0.986~0.997 | **0.519** | **0.475** | **0.470** | **0.495** | **0.551** | **0.620** | **0.722** |

- 블록 5부터 무너진다. 연산별로는 **MLP fc2만 붕괴**(0.620), QKV·attention은 둔감(≥ 0.997).
- B/32만의 특징: **블록 0도 민감**(0.521). 블록 0을 FP32로 남기는 시도는 아래 실패 표.

**단계 3. 원인: 고정 위치 토큰의 거대 활성값(attention sink).** <!-- src: R/G6_pre/README.md 7-2·7-4, b32/README.md B5 -->
- 블록 5의 MLP가 사진과 상관없이 **패치 15번 토큰**의 채널 67·301·145에, 블록 9의 MLP가 **패치 7번 토큰**의 채널 483에 아주 큰 값을 써 넣는다. 블록 5 출력에서 가장 큰 값이 보통 값(중앙값)의 약 990배다(128장).
- 15번 토큰은 128장 모두, 7번은 95%에서 고정 위치다. 블록 6~11에서 다른 토큰들이 attention의 26~48%를 15번에 몰아준다(균등 분배면 1.6%). CLS 토큰이 없는 ViT가 패치 일부를 "register"처럼 쓰는 현상이다(Darcet et al. [3], Sun et al. [4]).
- 텐서당 눈금 하나(per-tensor) 8비트에서 이 값이 눈금을 넓혀 그 층 출력이 망가진다. 실제 배포 눈금으로 잰 8비트 출력 SNR이 **블록 9 fc2 0.8 dB**(신호와 오차가 거의 같음)다.
- 메커니즘 확인(fake-quant, 64장): fc2를 8비트로 하되 거대 값 토큰만 빼고 눈금을 정하면 cos 0.997, 토큰별 눈금 0.999, 채널별 1.000으로 회복된다. **"8비트가 부족한 것이 아니라 눈금 단위가 맞지 않는 것"**이다.

**단계 4. 원인을 직접 고친다.**
- **토큰 축 — 레지스터 캐시(M2)** (RegCache [5], 학습 없는 register [6]의 생각을 고정 위치 sink에 맞춤): sink 토큰의 K·V를 보정 사진 평균 상수로 바꾸고, 그 토큰 행의 MLP 입력도 평범한 값(u_c)으로 바꾼다. 규칙 `15@6:5, 7@10:9` = K·V는 블록 6·10부터, MLP 입력은 블록 5·9부터. (15번의 K·V를 블록 5부터 바꾸면 그 토큰이 아직 그림 내용을 담고 있어 FP32 cos 0.998에 그쳤고, 한 블록 늦추자 0.9998이 되었다.) 원래 모델의 근사지만 **양자화 없이 dev 전체 +0.06 [−0.04, +0.19], cos 0.9999** — 근사 손실이 없다.
- **채널 축 — 오프라인 회전** (QuaRot [7]·SpinQuant [8] 계열): 잔차 흐름을 직교 행렬로 돌리고 역회전을 앞뒤 가중치에 흡수한다. LayerNorm γ·β를 접어 원래 모델과 수학적으로 같다. 몇 채널에 몰린 값이 768채널로 퍼진다.
- **결과(2×2)**: M2가 결정적이고(−60.96 → −1.58), 회전은 M2 위에서 유의하게 더한다(−1.58 → −0.79). 회전만으로는 여전히 무너진다(−54.52). 보호 층 없이(예전 방식은 블록 5·9 fc2를 FP32로 남김, −2.36) **MatMul 79개 전부 INT8**이 더 정확하다(+1.57 [+0.98, +2.23]).

**단계 5. 무엇이 정확도를 정하나 — 배포 눈금의 층별 SNR.** 실제 배포 파일과 같은 눈금·정수 가중치(출력 재반올림 포함)로 MatMul 79개 각각의 8비트 출력 SNR을 쟀다(사진 64장, 층마다 FP32 입력에서 시작하는 시뮬레이션). <!-- src: b32/submit_prep/H5_qdq_snr/why_after_qdq_summary.json ("full") -->

| 그래프 | 최솟값 dB (가장 나쁜 층) | 평균 dB | dev 손실 %p |
|---|---|---|---|
| 원래 | 0.8 (블록 9 fc2) | 28.5 | −60.96 |
| 회전만 | 1.0 (블록 9 fc2) | 30.6 | −54.52 |
| 예전 방식(블록 5·9 fc2 FP32) | 13.1 | 29.0 | −2.36 |
| M2만 | 14.9 | 29.5 | −1.58 |
| **회전 + M2** | **20.5** | **31.6** | **−0.79** |

- **가장 나쁜 층의 SNR 순서가 정확도 순서와 5개 모두 같다.** 평균은 맞지 않는다(회전만 30.6 dB가 M2만 29.5 dB보다 높은데 정확도는 붕괴). 층 하나만 무너져도 전체가 무너진다는 뜻이고, 처방이 그 "가장 나쁜 층"을 고친 것이다.

**단계 6. ORT 커널에 맞추기(O-A·O-B, 준용 O1).** 프로파일에서 남은 비용은 정수 행렬곱 앞뒤의 변환(Q/DQ), 정수 Softmax, 합쳐지지 않은 LayerNorm이었다.
- O-A: 정수 행렬곱 결과를 다시 8비트로 반올림하지 않고 바로 float으로 낸다(ORT 최적화 뒤 독립 DQ가 O-AB에서 66 → 1개). 반올림 단계가 하나 줄어 출력 오차도 준다(FP32 대비 cos 0.977 → 0.989).
- O-B: ORT가 못 합치던 잔차 Add + LayerNorm(소비자가 둘이라 기본 융합이 안 됨)을 `SkipLayerNormalization`으로 합쳤다(FP32에서 수학적으로 같음).
- 같은 바탕 비교에서 O-AB가 정확도 +0.57 [+0.17, +0.97], 4스레드 −11.4%로 사전 기준 세 개를 모두 만족했다.

**시도했지만 넣지 않은 것** (결과 전에 정한 기준으로 판정, 실패도 기록)

| 시도 | 결과 | 넣지 않은 이유 |
|---|---|---|
| 블록 0 FP32 보호 (X3) | −0.79 → −0.42, cos 0.977 → 0.991 | 짝 차이 +0.37 [−0.07, +0.80] 동률, 4스레드 지연 +5.9%, 크기 +20.6% → 사전 기준 두 개 미달 <!-- src: b32/block0/README.md --> |
| 가중치 4비트 (MatMulNBits, 블록 32) | 크기 최대 −28%(예전 방식 R0 구성 111.7 → 80.0 MiB, 제출 파일 98.0 MiB 기준이면 약 −18%), cos 유지 | B/16에서 x86 4스레드가 1.36~2.05배 느렸다(같은 커널). B/32 지연은 재지 않음 |
| 원인 층만 FP32 (예전 방식 R0) | −2.36 [−3.00, −1.77] | 회전+M2(전부 INT8)가 더 정확하고 더 작다 |
| O-A 단독 | +0.67, 지연 −4~−8% | CIFAR A −1.1%p로 사전 기준(−1.0) 미달. O-B와 함께일 때 통과 |
| O-B 단독 | 지연 −0.7~−2.7% | 사전 기준(모든 스레드 −3% 이상) 미달 |
| 활성값 S8 (B/16에서 확인) | 정확도 동률 | x86에서 QKV가 정수로 합쳐지지 않아 속도가 FP32 수준 |
| 토큰 삭제형 캐시 (B/16에서 확인) | 크기 −13% | Hailo LayerNorm 분해와 충돌 → 마스크형(M2)으로 대체 |
| Hailo 전부 8비트 (ALL8F) | Zoo보다 −18.9%·+1.80%p | GF보다 −0.87 [−1.46, −0.28] 열위(사전 기준 "정확도 유지" 미달). 본선 후보로만 남김 |

---

## 6. 벤치마크 설계 및 측정 결과

### 6.1 선정한 벤치마크

| 구분 | 내용 |
|---|---|
| 데이터셋 / 벤치마크명 | **제로샷 이미지 분류 7종**: Oxford-IIIT Pets, Oxford Flowers-102, Stanford Cars, Food-101, Caltech-101(고화질 5종, 주 지표) + CIFAR-100, ImageNetV2 matched-frequency(보조, 따로 표기) |
| 공개 여부 및 출처 URL | 모두 공개. Hugging Face `timm/oxford-iiit-pet`, `dpdl-benchmark/oxford_flowers102`, `tanganke/stanford_cars`, `ethz/food101`, `dpdl-benchmark/caltech101` / CIFAR-100: torchvision / ImageNetV2: https://github.com/modestyachts/ImageNetV2 |
| 평가셋 규모 | **test**: Pets 3,669 · Flowers 6,149 · Cars 8,041 · Food 25,250 · Caltech 6,084 · CIFAR-100 10,000 · ImageNetV2 10,000 (총 69,193장) / **dev**(구성 선택용, train에서 뗌): 고화질 5종 4,923장 + CIFAR 1,000장 |
| 평가 지표 | ① 제로샷 **Top-1** (고화질 5종 평균, FP32 대비 Δ와 짝 부트스트랩 95% 구간) ② FP32 대비 이미지 임베딩 cos ③ 크기·지연·FPS·피크 메모리 (FP32 대비 변화율) ④ (분석) 층별 8비트 출력 SNR |
| 지표 선정 사유 | 실제 쓰임새(텍스트로 정한 클래스에 사진 맞추기)를 그대로 잰다. 텍스트 분류기를 FP32로 고정해 차이가 이미지 인코더 양자화에서만 나오게 했다 |

### 6.2 벤치마크 선정 사유 및 타당성

- **분야가 다른 세밀 분류 5종 평균**: 동물·꽃·자동차·음식·일반 사물. 비슷한 클래스끼리 구분해야 해서 작은 임베딩 오차도 정확도에 드러난다.
- **프롬프트는 CLIP 논문 [2] 공식 템플릿**: SigLIP 권장 템플릿보다 dev에서 1.77%p 좋았다(−1.77 [−2.33, −1.22]). <!-- src: b32/README.md B2 -->
- **보조 지표는 따로 적는다**: CIFAR-100은 32×32를 키운 사진이라 앞쪽 층 변화에 지나치게 민감해 설정 선택에 쓰지 않는다. ImageNetV2는 일반성 확인용이다.
- **고르지 않은 대안**: ImageNet-1k 검증셋(등록제라 재현이 어려움 → 공개 ImageNetV2), 이미지-텍스트 검색(텍스트 인코더 효과가 섞임), linear probe(학습 필요, 쓰임새와 다름).
- **데이터 구간을 섞지 않았다**: 사진 선택은 이미지 바이트 SHA-1 해시로 정해 누가 돌려도 같다. 보정과 구성 선택은 train(dev)에서만 했고, **test는 제출 파일로 한 번만** 썼다(B/32 첫 사용, 결과는 고르는 데 쓰지 않음).
- **통계**: 후보 차이가 1%p 안팎이라 같은 사진끼리 짝지은 부트스트랩(데이터셋별 층화, 1,000회, 시드 0)으로 95% 구간을 냈다. 구간이 0을 포함하지 않으면 "유의"로 본다.
- **흔들림 점검**
  - 보정 시드 0·1·2 (M2rot, 이 PC): −0.79 / −0.80 / −0.70, 폭 0.10, 시드끼리 짝 차이 모두 유의하지 않음. <!-- src: b32/submit_prep/seed/dev_full/paired_ci.jsonl -->
  - 다른 PC 빌드: 같은 레시피를 준용 PC에서 만들면 −0.87, 이 PC는 −0.79 (0.08%p 차이).
  - 입력 변형 8종(변형 없음·좌우 반전·90도 회전·흑백·어둡게·저대비·잡음·JPEG 20, 고화질 5종 × 200장, 같은 변형의 FP32 대비): **O-AB는 8종 모두 유의한 손실 없음(−0.9 ~ +0.6%p)**. M2 근사 자체는 모든 변형에서 cos 0.9998 이상. 비교로 M2rot INT8은 흑백 −3.5·90도 회전 −2.2, 예전 방식은 흑백 −9.0으로 유의한 손실. <!-- src: b32/submit_prep/input_shift/input_shift_summary.json -->

### 6.3 재현 가능성 확보 방안

- **공개 데이터셋만 쓴다.** 실행할 때 Hugging Face Hub·torchvision에서 자동으로 받는다(`rn50/datasets.py`, 사진 선택 규칙 포함).
- **소규모 평가 묶음을 함께 낸다**: dev 6종 × 200장(전처리 끝난 256×256 입력, 정답, FP32 텍스트 분류기, FP32 임베딩) + numpy·onnxruntime만 쓰는 `portable_eval.py` + FP32·제출 모델 + sha256 목록. 몇 분이면 돈다(9.2 (가)).
- **모델 파일과 sha256 공개**: FP32 `be19a8ad…`, 제출 O-AB `7a01044c…`.
- **사진별 예측을 함께 낸다.** O-AB는 실행 환경에 따라 사진 몇 장의 결과가 다를 수 있다(9.4). 같은 파일이 i7(Windows)과 285K(WSL)에서 dev Cars만 91.93 / 91.83(순 약 1장)이고 나머지 데이터셋은 같았다(고화질 평균 89.69 / 89.67). 두 측정은 CPU뿐 아니라 OS·평가 코드·입력 캐시도 달라 원인은 아직 가리지 않았다. 같은 묶음으로 다른 PC에서 사진별 비교를 진행 중이다. <!-- src: b32/O_AB/dev_full/paired_ci.jsonl vs O1 6-3 표 -->
- **깨끗한 폴더에서 재현 절차를 확인했다.** 새 폴더에 Release 파일만 내려받아 sha256 13개 확인 → `portable_eval.py` 실행. FP32·제출 모델·M2rot 모두 기준 결과와 사진 1,200장의 예측이 비트 단위로 같고 cos 차이 0이었다(`b32/xplat/repro_check`). 같은 PC에서는 Release와 절차만으로 완전히 재현된다.
- **채점은 배포한 ONNX 파일로 한다.** 처음부터 다시 만들면 PC마다 float64 행렬 곱 반올림이 달라 파일이 달라진다(B/16에서 확인).

### 6.4 측정 프로토콜

| 측정 조건 | 설정값 |
|---|---|
| 입력 해상도 | 256×256 (open_clip 모델 기본 전처리, 정규화 평균·표준편차 0.5) |
| 배치 크기 | 1 (엣지 실시간 상황). 정확도 평가만 배치 50 |
| Warm-up 횟수 | 20회 (모델마다) |
| 측정 반복 횟수 | 사진 200장 × 3회를 **모델끼리 번갈아**(같은 실행에서 FP32와 최적화 모델 교대) × 세션 3번 이상 × 스레드 1·2·4 |
| 통계 방식 | 세션마다 장당 평균 → **세션 중앙값**. 변화율은 세션마다 같은 실행 안 비율의 중앙값. p50·p95도 기록. FPS = 1000 / 지연 |
| 측정 도구 및 방법 | ONNX Runtime 1.25.0 CPU, `ORT_ENABLE_ALL`, intra-op 스레드 고정·inter-op 1 (`siglip2/scripts/cross_latency.py`). 피크 메모리: 모델마다 새 프로세스, 4스레드, 세션 생성 + 추론 50회 중 최대 working set. 크기: ONNX 파일 바이트 |

- **이상치 규칙(측정 중 정함)**: 같은 스레드의 모든 세션 중앙값보다 8모델 실행의 기준 모델(B/16 FP32)이 10% 넘게 느린 세션은 뺀다. 그래서 1스레드 3세션, 2스레드 5세션, 4스레드 4세션의 중앙값이다. 재측정 때 같은 스크립트가 두 번 실행된 사고가 있었고, 첫 실행 값을 로그에서 복원했다(요약과 일치, 반복별 값은 잃음). <!-- src: b16/O_AB/README.md 2절 -->
- 주 조건은 결과를 보기 전에 정한 **4스레드**다.

### 6.5 측정 결과

**정확도 — test** (제출 파일로 1회, `b32/test`)

| 데이터셋 | FP32 | 최적화 후 | Δ (%p) [95%] | cos |
|---|---|---|---|---|
| Pets (3,669) | 93.54 | 93.21 | −0.33 [−0.76, +0.14] | 0.992 |
| Flowers (6,149) | 84.49 | 84.13 | −0.36 [−0.68, −0.02] | 0.991 |
| Cars (8,041) | 90.80 | 90.45 | −0.35 [−0.72, +0.00] | 0.988 |
| Food (25,250) | 88.90 | 88.90 | −0.00 [−0.17, +0.18] | 0.990 |
| Caltech (6,084) | 84.81 | 84.71 | −0.10 [−0.35, +0.13] | 0.985 |
| **고화질 5종 평균** | **88.51** | **88.28** | **−0.23 [−0.36, −0.08]** | 0.989 |
| (보조) CIFAR-100 (10,000) | 76.05 | 76.98 | +0.93 [+0.38, +1.44] | 0.964 |
| (보조) ImageNetV2 (10,000) | 65.85 | 65.65 | −0.20 [−0.61, +0.17] | 0.990 |

<!-- src: b32/test/test_summary.json (커밋 72784d1, dirty false) -->

**정확도 — dev 전체** (구성 비교, FP32 89.96%)
- 제출 O-AB: 89.69%, −0.27 [−0.62, +0.09](유의하지 않음), cos 0.989, CIFAR A −0.5. 데이터셋별 Pets −0.71, Flowers −0.63, Cars +0.31, Food −0.91, Caltech +0.59.
- 이전 단계 M2rot(회전+M2, 출력 재반올림 있음): −0.79 [−1.26, −0.34], cos 0.977.

**효율** (이 PC, 8개 모델 같은 실행에서 번갈아, 비공식. 재현 범위는 아래 '다른 조건' 표와 그림 F3b)

| 항목 | Baseline | 최적화 후 | 변화율 (%) |
|---|---|---|---|
| 정확도 지표 1 (Top-1, 고화질 5종, test) | 88.51% | 88.28% | −0.23%p |
| 정확도 지표 2 (임베딩 cos, 고화질 5종, test) | 1 | 0.989 | — |
| 추론 Latency (ms) 1 / 2 / 4스레드 | 132.8 / 74.0 / 61.5 | 47.9 / 31.3 / 27.0 | **−63.8 / −57.8 / −56.9** |
| Latency p50 / p95 (4스레드, ms, 1차 실행 세션 2·3) | 57.3~58.7 / 71.0~75.0 | 24.8~25.9 / 32.3~32.5 | 약 −56 / −56 |
| Throughput (FPS, 배치 1) 1 / 2 / 4스레드 | 7.5 / 13.5 / 16.3 | 20.9 / 32.0 / 37.1 | **+177 / +137 / +128** |
| 모델 크기 (MB) | 378.36 | 102.76 | **−72.8** |
| 피크 메모리 (MB) | 475.7 | 202.4 | **−57.5** |

<!-- src: 위 2장 src와 같음. p50/p95 = all8_20261022/cross_latency.json sessions(4T 세션 2·3) -->

**다른 조건의 개선율** (같은 파일, 각 조건에서 FP32와 같은 실행으로 번갈아)

| 기기 / 실행 | 4스레드 FP32 → 최적화 | 4스레드 | 1스레드 | 메모리 |
|---|---|---|---|---|
| i7-1165G7, 10/22 낮 (위 표) | 61.5 → 27.0 ms | −56.9% | −63.8% | −57.5% |
| i7-1165G7, 10/22 저녁 (연속 실행으로 발열, 2·4스레드 약 40% 느림, 이상치 규칙 미적용) | 88.6 → 37.3 ms | −56.9% | −60.2% | −57.5% |
| i7-1165G7, 10/23 새벽 (PC 쉬는 중, 3모델 같은 실행, 9구간 모두 정상) | 83.2 → 36.6 ms | −55.9% | −60.7% | −57.4% |
| Core Ultra 9 285K (준용, WSL2) | 27.01 → 12.26 ms | −54.6% | −64.6% | (측정 정의 다름) |

<!-- src: latency_this_pc/b32_submit_20261022/cross_latency.json summary·peak_mem; O1 6-4 (285K, 비율은 지연 중앙값에서 계산) -->
- 절대 지연은 기기와 PC 상태에 따라 크게 다르지만(같은 PC에서 4스레드 FP32 61.5~88.6 ms), **원본 대비 개선율은 4스레드 −55~−58%, 1스레드 −60~−65%로 비슷하다.** 채점이 원본 대비 개선율이므로 이 값을 주 결과로 둔다.

**Hailo-10H 실칩**(본선 이식 확인)은 3.3절 표다. 우리 HEF GF: 배치 1 14.93 ms(≈ 67 FPS), 정확도 −1.55%p, Zoo 공개 HEF 대비 −9.5%·+2.67%p.

### 6.6 결과 해석

**손실은 어디서 생기나.**
- 원인 분석이 가리킨 "거대 값이 눈금을 넓혀 그 층 출력이 무너지는" 손실은 M2와 회전으로 해소됐다. 2×2에서 M2만으로 −61.0 → −1.6%p가 됐고, 가장 나쁜 층의 SNR이 0.8 → 20.5 dB로 올랐다.
- 남은 손실은 MatMul 79개를 모두 8비트로 한 데서 오는 작은 임베딩 오차다(cos 0.989). O-A로 재반올림 한 단계를 없애자 이 오차가 더 줄었다(같은 바탕 비교, 준용 PC: 손실 −0.87 → −0.29, 짝 차이 +0.57 [+0.17, +0.97]).

**어떤 입력에서 취약한가.**
1. **원래 비슷한 클래스 쌍의 경계에 있는 사진**: 새 오답은 클래스마다 대부분 1~2장(최대 3장)씩 흩어지고, staffordshire bull terrier ↔ american pit bull terrier, ragdoll ↔ birman처럼 원래 헷갈리는 쌍에서 뒤집힌다(이전 주 후보 M2rot 기준 분석, `b32/G3_error_classes`). 특정 클래스가 망가진 것이 아니다.
2. **데이터셋별**: test에서 Flowers(−0.36, 유일하게 구간이 0을 포함하지 않음)·Cars(−0.35, 구간 상한 +0.00으로 경계)·Pets(−0.33)처럼 세밀한 구분이 필요한 쪽이 조금 더 잃었고, Food(−0.00)·Caltech(−0.10)는 거의 그대로다.
3. **입력 변형**: 90도 회전(−0.9)·저대비(−0.9)·흑백(−0.8)에서 점추정이 가장 낮았지만 모두 유의하지 않다. 처방 전 모델은 흑백에서 크게 무너졌다(6.2).
4. **저해상도(CIFAR-100)**: 오히려 +0.93%p(유의) 올랐다. 원인은 확인하지 않았다(7.1).

**dev와 test가 일치한다.** dev −0.27 [−0.62, +0.09], test −0.23 [−0.36, −0.08]로 같은 수준이다. dev에서 여러 구성을 고르며 생길 수 있는 과적합이 test에서 드러나지 않았다. test는 표본이 커서 구간이 좁아져 "작지만 유의한 손실"로 나왔다.

**CPU 속도에 대해.**
- 다른 PC 프로파일(285K, 4스레드, 30장 1회)에서 O-AB 커널 시간의 약 54%가 정수 행렬곱이고, 독립 DQ는 거의 0(0.01 ms)이 됐다. 남은 것은 입력 Q(정적 양자화라 필요, 1.30 ms), 잔차 Add(0.77 ms), 데이터 이동(Transpose 0.45 ms), SkipLayerNorm(0.40 ms)이다. <!-- src: O1 6-6 -->
- B/32는 토큰이 64개뿐이라 FP32로 남는 연산의 비중이 커서, 스레드가 늘수록 배속이 덜 오른다(1스레드 2.8배 → 4스레드 2.3배, 해석).
- 이 PC 프로파일(10/23 새벽, 4스레드, 200장): O-AB가 M2rot보다 −15%이고, 줄어든 곳은 Q/DQ(6.8 → 3.9 ms), 정수 Softmax 제거(3.5 ms → float Softmax), 잔차 Add(6.1 → 4.1 ms)다. 정수 행렬곱 시간은 같다(22.8 / 22.9 ms). <!-- src: latency_this_pc/b32_cool_20261023/README.md (10/22 저녁 프로파일은 게임 실행 중이라 무효) -->

---

## 7. 한계점 및 엣지 디바이스 이식 계획

### 7.1 현재 한계점

1. **속도는 노트북 한 대의 비공식 값이다.** 같은 PC에서도 실행에 따라 2·4스레드 절대값이 35~44% 달라졌다(4스레드 FP32 61.5 / 88.6 / 83.2 ms). 발열만으로는 설명되지 않아(같은 시각 B/16 확인값은 정상) 실행 구성에 따른 CPU 클럭 상태 차이로 추정한다. 그래서 같은 실행 안 비율로만 비교했고, 비율은 3번 모두 비슷했다.
2. **O-AB는 onnxruntime 전용이다.** `SkipLayerNormalization`(com.microsoft 연산)이 들어 있어 다른 ONNX 실행기에서는 돌지 않는다. [확인 필요: 채점 PC·실행 방식이 onnxruntime인지 학술부 답] — 아니면 표준 연산만 쓰는 M2rot INT8로 바꾼다. 본선 HEF와 같은 회전 행렬을 쓰려면 준용 바탕 파일(`3a46ae8a…`, dev −0.87)이어야 한다(이 PC 빌드 `9783d7de…`는 회전 행렬이 달라 본선 연속성이 깨짐).
3. **O-AB는 실행 환경마다 사진 몇 장의 결과가 다를 수 있다.** 같은 파일이 i7(Windows)과 285K(WSL)에서 dev Cars 순 약 1장 차이였다. 정수 → float 커널(`MatMulIntegerToFloat`)의 명령어 집합 차이일 수도 있고, OS·평가 경로 차이일 수도 있다(사진별 비교 진행 중). 출력 재반올림이 있는 M2rot은 B/16에서 x86 세 대 사진별 비트 동일이었다(B/32 M2rot은 비교한 적 없음).
4. **VNNI가 없는 x86과 ARM에서는 확인하지 않았다.** reduce_range는 VNNI 없는 x86의 포화를 막기 위한 설정이지만 효과는 시험하지 못했다. [확인 필요: 정민·준용 PC 재현 묶음 결과]
5. **O-A의 float 출력을 Hailo에서 어떻게 대응할지 정하지 않았다.** HEF는 같은 FP32 그래프에서 Hailo 방식으로 따로 양자화하므로, ORT의 "행렬곱 출력 float"과 Hailo의 8/16비트 출력이 다르다. 16비트로 맞출지(지연 증가 가능) 커널 형식 차이로 기록만 할지 본선 HEF와 함께 정한다.
6. **CIFAR-100이 +0.93%p로 오히려 올랐는데 원인을 모른다.** 저해상도 입력에서 양자화 잡음의 우연한 효과일 수 있다(보조 지표라 선택에 쓰지 않음).
7. **Hailo 최종 HEF가 정해지지 않았고, Hailo 손실이 CPU보다 크다**(GF −1.55, ALL8F −2.42 vs CPU −0.27).
8. **레지스터 캐시는 보정 데이터로 정한 근사다.** dev·test 7종과 입력 변형 8종에서는 근사 손실이 없었지만, 분포가 아주 다른 입력에서는 맞지 않을 수 있다.
9. **원인 분석 일부는 fake-quant다.** 토큰별 눈금 실험은 메커니즘 확인용이라 지금 정수 커널에서 그대로 쓸 수 없다.

### 7.2 본선 이식 계획 (11/6 ~ 12/4)

본선은 예선 모델(SigLIP2 B/32, 같은 회전 행렬·sink 규칙·레지스터 캐시 상수의 같은 모델)을 그대로 쓴다. 예선과 같은 벤치마크·측정 방법(배치 1 지연 주 지표)으로 서버(ORT INT8)와 엣지(Hailo)를 비교한다. 컴파일 과정의 조정은 "예선 대비 변경 사항"으로 기록한다.

| 시기 | 수행 내용 | 예상 리스크 |
|---|---|---|
| 1주차 (11/6~11/12) | Pi 5 + AI HAT+ 2 구성, HailoRT 버전 확인 → 맞는 DFC로 HEF 다시 컴파일, 본선 HEF 확정(GF 등)과 O-A 대응 결정, 실칩 dev 전체 1회 | HailoRT·DFC 버전 불일치 → 버전별 DFC 환경 미리 준비 |
| 2주차 (11/13~11/19) | Pi에서 dev·test 정확도, 배치 1 지연(워밍업 20, 200장 × 3회 × 3세션), 메모리·전력. 호스트 전처리(크기 맞춤) 시간 포함 | Pi CPU 전처리가 병목 → 크기 맞춤 최적화(정규화는 이미 HEF 안) |
| 3주차 (11/20~11/26) | 추가 최적화: Hailo INT4 가중치(MLP부터), 잔차 8비트화 재시도(ALL8 계열), 같은 판정 규칙(결과 전 고정)으로 평가 | INT4·8비트 정확도 손실, 컴파일 결과물 이상(B/16에서 실칩만 무너진 사례) → 에뮬레이터·실칩 모두 확인 |
| 4주차 (11/27~12/4) | 데모(카메라 → 제로샷 분류), 최종 보고서(11/30)·발표, 재현 코드 정리 | 발열에 따른 지연 변동 → 시작 온도 기록, 같은 조건 반복 |

---

## 8. 제출물 체크리스트

| No | 제출 항목 | 파일명 또는 링크 | 확인 |
|---|---|---|---|
| 1 | 결과 보고서 (PDF) | `[트랙2-예선]_자유_[확인 필요: 팀명]_보고서.pdf` | □ |
| 2 | 베이스라인 모델 파일 | `siglip2_image_fp32.onnx` (B/32, 378.36 MB, sha256 `be19a8ada01055df…`) | □ |
| 3 | 최적화 모델 파일 | `siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx` (102.76 MB, sha256 `7a01044c6ba32bf4…`) | □ |
| 4 | 모델 포맷 및 버전 정보 명시 | ONNX opset 17 + `com.microsoft` 1(SkipLayerNormalization). 입력 `image` float32 N×3×256×256(전처리 후), 출력 `embedding` 768차원. onnxruntime 1.25.0 CPU, `ORT_ENABLE_ALL` | □ |
| 5 | 벤치마크 실행 스크립트 | `siglip2/scripts/portable_eval.py`(소규모 묶음), `siglip2/s2_paired_ci.py`(dev), `siglip2/s2_test_eval.py`·`s2_test_summary.py`(test), `siglip2/scripts/cross_latency.py`(지연·메모리), `siglip2/s2_model_stats.py`(크기·파라미터·GFLOPs) | □ |
| 6 | 의존성 목록 | `siglip2/requirements.txt` (+ `rn50/requirements.txt` 공통 평가 코드) | □ |
| 7 | 실행 방법 README | `siglip2/README.md` [확인 필요: B/32 기준으로 갱신] | □ |
| 8 | 데이터셋 경로 또는 다운로드 방법 | 실행 때 자동 다운로드(`rn50/datasets.py`, 출처 6.1). 소규모 묶음 dev 6종 × 200장 | □ |

- 모델 파일·묶음은 GitHub Release로 낸다: [확인 필요: 최종 제출 Release 태그]. 지금 내부 확인용 pre-release `siglip2-b32-submit-20261023`에 FP32·제출 모델·비교용 M2rot·dev200 묶음·`portable_eval.py`·`SHA256SUMS.txt`가 있다.
- 저장소가 비공개라 학술부 접근(초대 또는 공개 링크)을 준비하고 심사 종료(11/5)까지 유지한다.

---

## 9. 재현 실행 절차

### 9.1 환경 구성

```bash
git clone https://github.com/jungminmobile/oslab_mobile.git
cd oslab_mobile
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
python -m pip install -r siglip2/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
gh release download <제출 Release 태그> --repo jungminmobile/oslab_mobile -D release
cd release && sha256sum -c SHA256SUMS.txt && cd ..
export S2_MODEL_ID=hf-hub:timm/ViT-B-32-SigLIP2-256
export SIGLIP2_DATA=$HOME/siglip2_data SIGLIP2_ARTIFACTS=$HOME/siglip2_b32_art HF_HOME=$HOME/hf_cache OPENBLAS_NUM_THREADS=1
mkdir -p $SIGLIP2_ARTIFACTS/onnx && cp release/*.onnx $SIGLIP2_ARTIFACTS/onnx/
```
- `<제출 Release 태그>` = [확인 필요: 최종 제출 Release 태그]. Windows(PowerShell)는 `export` 대신 `$env:S2_MODEL_ID="..."` 형식이다.
- **`S2_MODEL_ID`를 꼭 지정한다.** 지정하지 않으면 코드가 B/16(224)으로 동작한다.
- [확인 필요: 새 폴더·새 가상환경에서 이 절의 B/32 명령을 그대로 실행해 확인]

### 9.2 벤치마크 실행 명령

**(가) 빠른 확인: 소규모 묶음** (저장소 코드·데이터 다운로드 없이, `pip install numpy onnxruntime==1.25.0`만 필요, 실행 약 5~6분)
```bash
python release/portable_eval.py --data release --out out_quick --threads 4 \
  --models fp32=release/siglip2_image_fp32.onnx \
           oab=release/siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
```

**(나) 전체 정확도: dev** (Baseline·최적화 함께, 짝 비교)
```bash
cd siglip2
python s2_build_dev.py all --threads 4     # dev 캐시·텍스트 분류기·FP32 기준 (데이터 자동 다운로드, 처음 1~2시간)
RN50_RESULTS=../results/siglip2/repro_b32_dev python s2_paired_ci.py --models \
  siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
```

**(다) 전체 정확도: test 7종** (Baseline·최적화를 같은 실행에서)
```bash
for s in pets caltech flowers cars food cifar_test imagenetv2; do
  RN50_RESULTS=../results/siglip2/repro_b32_test python s2_test_eval.py $s --threads 4 \
    --models siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
done
RN50_RESULTS=../results/siglip2/repro_b32_test python s2_test_summary.py
```

**(라) 속도·메모리·크기** (Baseline·최적화를 같은 실행에서 번갈아)
```bash
python scripts/cross_latency.py --out out_speed --mem \
  --models b32_fp32=$SIGLIP2_ARTIFACTS/onnx/siglip2_image_fp32.onnx \
           b32_oab=$SIGLIP2_ARTIFACTS/onnx/siglip2_image_int8_g6b_b32_m2rot_o1b_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_moq0_all.onnx
```

### 9.3 기대 출력

| 명령 | 출력 파일 | 기대값 | 보고서 위치 |
|---|---|---|---|
| (가) | `out_quick/result.json` | FP32 고화질(200장 × 5) 88.9%, O-AB Δ **+0.4%p**, cos **0.9893**, CIFAR A 71.0 (FP32 71.5). 이 PC(AVX-512 VNNI) 값이며, 다른 x86 CPU에서는 사진 몇 장(0.1~0.5%p) 다를 수 있다 | 6.3 |
| (나) | `paired_ci.jsonl` | FP32 89.96%, O-AB 89.69% (Δ −0.27 [−0.62, +0.09]), cos 0.989, CIFAR A −0.5 | 6.5 dev |
| (다) | `test.jsonl`, `test_summary.json` | 고화질 88.51 → 88.28 (Δ −0.23 [−0.36, −0.08]), CIFAR-100 +0.93, ImageNetV2 −0.20 | 2, 6.5 |
| (라) | `out_speed/cross_latency.json` | FP32 대비 4스레드 약 −55~−58%, 1스레드 약 −60~−65%(절대 ms는 PC마다 다름), 피크 메모리 약 −57%, 크기 378.36 → 102.76 MB | 2, 6.5 |

<!-- src: (가) b32/xplat/i7-1165G7/result.json -->

### 9.4 재현 시 유의사항

1. **채점은 배포한 ONNX 파일로 해 주십시오.** 처음부터 다시 만들면 PC마다 float64 행렬 곱의 반올림이 달라 회전 행렬부터 파일이 달라지고, dev 정확도가 0.1%p 안팎 달라질 수 있습니다(B/32 같은 레시피 두 PC −0.79 / −0.87).
2. **onnxruntime으로 실행해 주십시오.** 제출 모델에 `com.microsoft` 연산(SkipLayerNormalization)이 있습니다.
3. **정수 커널 결과는 실행 환경(CPU 명령어 집합·OS)에 따라 조금 다를 수 있습니다.** 이 PC(AVX-512 VNNI, Windows)와 285K(AVX-VNNI, WSL)에서 dev 평균은 89.69 / 89.67%로 거의 같고 Cars에서 순 약 1장만 달랐습니다. 비교용으로 이 PC의 사진별 예측 파일을 함께 냅니다. VNNI가 없는 x86과 ARM은 [확인 필요: 확인 결과].
4. **지연 절대값은 PC·전원 설정·발열에 따라 크게 다릅니다.** 노트북은 고성능 모드·충전 상태에서, 다른 프로그램 없이, FP32와 같은 실행에서 **비율**로 비교해 주십시오.
5. **처음 실행 시간**: dev 캐시 생성 1~2시간, test 7종(FP32·INT8 함께) 이 PC에서 약 3시간 15분(대기 제외). 빠른 확인은 9.2 (가)를 써 주십시오.
6. **별도 다운로드**: 텍스트 분류기를 만들 때 Hugging Face `timm/ViT-B-32-SigLIP2-256` 가중치를 처음 한 번 받습니다. ImageNetV2는 `siglip2/scripts/test_g6.sh`가 받습니다. 디스크 [확인 필요: B/32 기준 필요 용량], GPU는 필요 없습니다.

---

## 10. 같은 처방이 다른 모델에도 통한다 — SigLIP2 B/16

원인 분석과 처방이 B/32 하나에만 맞춘 것이 아님을 보이려고, 같은 계열의 B/16(224×224, 토큰 196개)에 같은 레시피를 적용했다(제출 모델 아님).

| 항목 | B/16 결과 | 근거 |
|---|---|---|
| 기전 | 같은 채널(67·301·145·483)에, 고정 sink 토큰(55·13, 블록 5·9부터)에 거대 값. 민감도도 블록 5부터, 원인은 fc2 | G3_why, G6_pre |
| 2×2 (dev, 원래 / 회전만 / M2만 / 둘 다) | −63.49 / −48.97 / −0.62 / **−0.31** — M2가 결정적, 회전은 M2 위에서 보탬(+0.32, B/16에서는 유의하지 않음) | b16/A1_ablation |
| O-AB dev 전체 | −0.18 [−0.55, +0.18] (FP32 91.66%), cos 0.987 | b16/O_AB |
| 회전+M2(M2rot_ort) dev / test | −0.23 [−0.64, +0.18] / −0.34 [−0.50, −0.18] (test는 이 파일로 1회) | G6_m2rot_ort |
| 개선율 (이 PC, B/32와 같은 실행, O-AB) | 지연 1/2/4스레드 −62.1 / −60.5 / −57.9%, 크기 −74.1%, 메모리 −52.9% | latency_this_pc/all8 |
| 보정·빌드 흔들림 | 빌드 5개 dev −0.23 ~ −0.49 (폭 0.26) | b16/E5_calib_seed |

| 배포 눈금 층별 SNR (6구성) | 가장 나쁜 층 SNR 순서 = 정확도 순서 (원래 0.1 → 회전만 0.3 → R0 6.9 → M2만 12.9 → M2rot 18.1 → O-AB 18.2 dB). **B/32 5구성과 합쳐 11구성 모두 일치**, 평균은 일치 안 함 (그림 F4b) | b16/generalization/H5_qdq_snr |
| M2·회전 근사 손실 (FP32) | +0.10 [0.00, +0.22], cos 0.9999 (두 PC 빌드 사진별 예측 동일) | b16/generalization/m2rot_fp32_devfull |
| 입력 변형 8종 (O-AB) | 흑백·JPEG 20에서 −1.2 (유의), 나머지는 유의하지 않음. M2 근사는 모든 변형에서 cos ≥ 0.9998. B/32 O-AB(8종 모두 유의하지 않음)보다 덜 견고 | b16/generalization/input_shift |

- 두 모델 모두 단순 INT8은 붕괴하고, 토큰 축 처방(M2)이 결정적이며, 행렬곱을 전부 INT8로 해도 손실이 0.3%p 안팎이다. 정확도를 정하는 것이 "가장 나쁜 층"이라는 분석도 두 모델에서 같게 성립한다. **원인 → 처방이 모델 크기·토큰 수가 달라도 통한다**는 근거다.
- 차이: B/32는 블록 0도 민감하고(B/16은 아님), 회전의 추가 효과가 B/32에서 더 컸다(+0.79 유의 vs +0.32).

---

## 부록 A. 참고 문헌

**모델·기법**
1. M. Tschannen et al., "SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features," arXiv:2502.14786, 2025. — 대상 모델
2. A. Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021 (arXiv:2103.00020). — CLIP, 제로샷 프롬프트 템플릿
3. T. Darcet, M. Oquab, J. Mairal, P. Bojanowski, "Vision Transformers Need Registers," ICLR 2024 (arXiv:2309.16588). — 고정 위치 high-norm 토큰
4. M. Sun, X. Chen, J. Z. Kolter, Z. Liu, "Massive Activations in Large Language Models," COLM 2024 (arXiv:2402.17762). — 거대 활성값
5. S. Kim et al., "Activation Quantization of Vision Encoders Needs Prefixing Registers," arXiv:2510.04547, 2025. — RegCache, 레지스터 캐시(M2)의 바탕
6. N. Jiang, A. Dravid, A. Efros, Y. Gandelsman, "Vision Transformers Don't Need Trained Registers," NeurIPS 2025 (arXiv:2506.08010). — 학습 없이 register 역할 옮기기
7. S. Ashkboos et al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs," NeurIPS 2024 (arXiv:2404.00456). — 무작위 Hadamard 회전
8. Z. Liu et al., "SpinQuant: LLM Quantization with Learned Rotations," ICLR 2025 (arXiv:2405.16406). — 회전으로 이상치 퍼뜨리기
9. H. Wu, P. Judd, X. Zhang, M. Isaev, P. Micikevicius, "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation," arXiv:2004.09602, 2020. — Percentile 보정, 정적 PTQ
10. M. Nagel, M. van Baalen, T. Blankevoort, M. Welling, "Data-Free Quantization Through Weight Equalization and Bias Correction," ICCV 2019 (arXiv:1906.04721). — 바이어스 보정(Hailo optimization_level 1)

**데이터셋**
11. O. M. Parkhi et al., "Cats and Dogs," CVPR 2012. — Pets
12. M.-E. Nilsback, A. Zisserman, "Automated Flower Classification over a Large Number of Classes," ICVGIP 2008. — Flowers-102
13. J. Krause et al., "3D Object Representations for Fine-Grained Categorization," ICCV Workshops 2013. — Stanford Cars
14. L. Bossard, M. Guillaumin, L. Van Gool, "Food-101 – Mining Discriminative Components with Random Forests," ECCV 2014. — Food-101
15. L. Fei-Fei, R. Fergus, P. Perona, "Learning Generative Visual Models from Few Training Examples," CVPR Workshops 2004. — Caltech-101
16. A. Krizhevsky, "Learning Multiple Layers of Features from Tiny Images," Tech. Report, 2009. — CIFAR-100
17. B. Recht et al., "Do ImageNet Classifiers Generalize to ImageNet?," ICML 2019 (arXiv:1902.10811). — ImageNetV2

**도구**
18. ONNX Runtime 1.25.0 (Microsoft), https://onnxruntime.ai — 정적 양자화, CPU 정수 커널, `SkipLayerNormalization` 융합
19. Hailo Dataflow Compiler 5.4.0 · HailoRT 5.3.0 · Hailo Model Zoo, https://hailo.ai
20. OpenCLIP 3.3.0 / timm 1.0.30 — 모델 불러오기·export

[확인 필요: 5번 학회 채택 여부, 9~17번 서지 정보를 PDF 전에 한 번씩 확인]

## 부록 B. 그림 (넣을 위치)

그림은 모두 raw에서 바로 그린다(`siglip2/scripts/make_report_figs.py`, `make_report_figs_b32.py`, 그림에 쓴 숫자는 `figs_data*.json`).
1. `F1_sensitivity.png` 민감도 지도(B/16·B/32, 블록·연산별 cos) → 5.3 단계 2
2. `F4b_min_snr_vs_acc.png` 배포 눈금 층별 SNR(가장 나쁜 층·평균) vs dev 정확도 손실, B/32 5 + B/16 6구성 → 5.3 단계 5, 10장
3. `F2_ablation_2x2.png` 2×2 단독 효과(B/32·B/16) → 5.1, 10장
4. `F3b_improvement_b32.png` 제출 모델 FP32 대비 개선율(크기·메모리·지연 1/2/4T), 이 PC 3회 → 2장 또는 6.5
5. `F5_hailo_b32.png` Hailo-10H 실칩 배치 1 지연 vs dev 정확도(Zoo 공개 HEF Z, G, GF, ALL8F, 같은 세션) → 3.3
- `F3_improvement.png`·`F4_activation_snr.png`는 B/16 시절 그림이라 쓰지 않는다.

## 부록 C. 결정·실험 기록 위치

| 내용 | 위치 |
|---|---|
| 제출 모델·파일 결정 | `siglip2/DECISIONS.md` D-048, D-049 |
| B/32 CPU 실험 전체 | `results/siglip2/G8_candidates/b32/` (README, G3RQ, block0, O_AB, submit_prep, test, xplat) |
| 지연·메모리 원자료와 이상치 규칙 | `results/siglip2/G8_candidates/latency_this_pc/`, `b16/O_AB/README.md` 2·3절 |
| 기전(sink 위치·거대 활성값) | `results/siglip2/G6_pre/README.md` |
| O-A·O-B | 준용 브랜치 `siglip2/o1-ort-speed` `results/siglip2/O1/README.md` |
| Hailo 실칩 | 준용 브랜치 `siglip2/g6e-b16-g`(G6b), `siglip2/g6d-speed2`(G6d), `siglip2/g6f-ceiling`(G6f) |
