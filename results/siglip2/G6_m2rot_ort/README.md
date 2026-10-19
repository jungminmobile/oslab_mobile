# 예선 최종 M2rot_ort (D-044) — 재현·교차 측정·test (2026-10-19~)

예선 ORT 최종을 R0(fc2L2 U8+rr, `5eee7839…`)에서 **M2rot_ort**(`e84c3727f51e3b97cd59d9ab9604e01acc78f81493926a0696fbf96dc9c60f9e`, 91.72 MiB, MatMul 79개 전부 INT8)로 바꿨다 (D-044, test 보기 전에 확정). 파일은 GitHub pre-release `siglip2-m2rot-ort-20261019` (준용 생성, 10/19 쭈리 PC에서 받아 sha256 확인).

## 1. 정민 노트북 교차 측정 (10/19, 용 요청, 정민 실행)
- 환경: Intel Core Ultra 7 258V (AVX-VNNI), Windows 11, ORT 1.25.0, 충전 중. worktree `43cc1b8`(dirty 없음), 세션 전 CPU 부하 9~19%. 파일 sha256 세 개 모두 일치(Release SHA256SUMS·복사본까지), 모델 재생성 없음
- raw: `xcheck_jungmin/` (정민 zip 전체, 82 KB — 예측 npz, result.json, 지연 json 9개·runs.jsonl·env.json·gate.log, sha256 확인 로그 `VERIFY PASS`, 실행 스크립트). 준용은 같은 내용을 `results/siglip2/G3RQ_m2rot/ort/xcheck_jungmin/`에 PR로 받을 예정
- **원본 대조 (10/19 쭈리 PC)**: 지연 9개 json에서 다시 낸 3세션 중앙값과 감소율이 아래 표와 같다. 환경 json: Ultra 7 258V, ORT 1.25.0, 충전 중 98%, 커밋 `43cc1b8` dirty 없음
- **R0 사진별 비교**: 정민 노트북 R0 예측·cos가 쭈리 PC(i7, `../G3_xplat/x86_i7-1165G7/pred_u8rr.npz`)와 **비트 단위로 같다** (Pets·Food·CIFAR A 예측 100% 일치, cos 차이 0) → R0는 x86 세 세대(AVX-512 VNNI·AVX-VNNI 285K·258V)에서 출력이 같다

정확도 (portable_eval, dev 6×200, 4스레드)
| 모델 | 고화질 cos (정민 / 연구실 285K) | 고화질 Δtop-1 | CIFAR A Δ |
|---|---|---|---|
| R0 | 0.9545 / 0.9547 | +0.1 | −5.5 |
| M2rot_ort | 0.9756 / 0.9757 | +0.3 | −2.5 |
- FP32 top-1은 6종 모두 기준(`fp32_top1`)과 같음

지연 (s2_bench_speed, 3세션 × 스레드 1·2·4, 세션 평균의 중앙값, 괄호는 FP32 대비)
| 모델 | t1 ms | t2 ms | t4 ms |
|---|---|---|---|
| FP32 | 549.4 | 278.0 | 156.8 |
| R0 | 217.0 (−60.5%) | 115.4 (−58.5%) | 79.5 (−49.3%) |
| **M2rot_ort** | **204.7 (−62.7%)** | **109.8 (−60.5%)** | **76.5 (−51.2%)** |
- ① M2rot_ort는 모든 스레드에서 FP32보다 빠르다 ② R0보다도 빠르다 (t1 −5.7%, t2 −4.9%, t4 −3.8%, 9번 비교 모두)
- 절대값이 10/18 같은 노트북 측정(FP32 t1 305 ms)보다 약 1.8배 느렸다 (원인 미확인). 쭈리 확인: 제대로 측정된 값으로 쓴다. 비교는 같은 실행 안의 FP32 대비 비율로 한다
- 세션 3의 4스레드만 10~20% 느렸으나 순위 같음 (중앙값 정리)
- peak 메모리는 재지 않음 (`s2_model_stats.py`가 PyTorch 모델을 불러오므로 건너뜀)
- py-cpuinfo 9.0이 258V의 avx_vnni를 false로 찍음 (사양상 지원)

## 2. 쭈리 PC (i7-1165G7, AVX-512 VNNI) — dev 재현·test 1회·지연 3세션
- 스크립트 `siglip2/scripts/m2rot_ort_g6.sh` (sha256 `e84c3727…` 확인 뒤 실행, 커밋 `86554b0`, dirty 없음)

### 2-1. dev 전체 재현 (10/19 20:05~20:29, `dev_full/paired_ci.jsonl`)
| 모델 | 고화질 Δtop-1 (FP32 91.66% 대비) [95%] | 고화질 cos |
|---|---|---|
| R0 (`5eee…`) | −0.78 [−1.26, −0.31] | 0.9547 |
| **M2rot_ort** (`e84c…`) | **−0.23 [−0.64, +0.18]** | **0.9757** |
| M2rot_ort − R0 | **+0.56 [+0.10, +1.01]** | |
- **준용 연구실(285K) dev 전체 결과와 구간까지 완전히 같다** (`G3RQ_m2rot/ort/README.md` 6절).
- 사진별: 이 PC의 M2rot_ort dev 각 앞 200장 예측·cos가 정민 노트북(258V) 결과와 **비트 단위로 같다** (6종 모두 예측 100% 일치, cos 차이 0) → M2rot_ort도 x86 세 세대(AVX-512 VNNI i7, AVX-VNNI 285K·258V)에서 출력이 같다. 마스크용 float 원소 연산이 늘었지만 x86 재현성에는 영향이 없다
- [test 1회, 지연 3세션은 아래에 추가]
