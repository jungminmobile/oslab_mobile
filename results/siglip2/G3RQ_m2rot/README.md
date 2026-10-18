# G3RQ: M2 + 회전(rothad) 조합을 Hailo에서 평가 (설계, 2026-10-18)

> 상태: **실칩 비교까지 완료 (10/18 23:52).** 실칩 dev 전체 M2rot − rothad +0.53 [+0.03, +1.02]%p (유의), 지연 +3.6% → Hailo 최종 교체를 팀에 제안 (`hailo/README.md` 6절, D-043 제안). 판정 기준(4절)은 결과를 보기 전에 고정한다. 브랜치 `siglip2/g3rq-m2rot` (`siglip2/g3r-regcache` + `origin/siglip2/g3q-rotate` merge), worktree `~/oslab_m2rot`. 담당: 용 (용 요청).

## 1. 질문
회전 모델 rothad(D-042 Hailo 최종) 위에 G3R의 M2 그래프(sink K·V 캐시 교체 + sink 행 MLP 입력을 u_c로 교체, 잔차 마스크 없음)를 얹으면,
Hailo 에뮬레이터에서 rothad_D256L1·M2_D256L1보다 좋은가? 좋은 쪽을 컴파일해 HEF 크기·컨텍스트를 본다.

## 2. 구성 M2rot
- 회전: `rotq_rothad.npy` (sha256 `3a6391a2091db60d…`, G3Q `build.jsonl`과 같음)로 `s2_rotate.rotate_trunk` → FP32 가중치만 바꾼 회전 모델.
- 그 위에 M2 (`s2_regcache.py --graph mask --mask-form bcast --mask-site mlp`)를 **기본 꺼진 옵션 `--rot rothad`**로 얹는다.
- 캐시는 회전된 모델에서 다시 계산 (보정 train 고화질 256장, 시드 0, dev 안 씀):
  - K_c·V_c: qkv가 LN γ·β와 Qᵀ를 흡수하므로 원래 캐시와 같아야 한다 → 검사 (b).
  - u_c: 회전된 공간의 norm2 출력(γ=1, β=0) 평균이라 새로 계산. fc1이 diag(γ)·Qᵀ와 β를 흡수하므로 fc1(u_c^rot) = fc1(u_c)가 되어 M2와 동치.

## 3. 사전 확인 (하나라도 실패하면 optimize 하지 않고 보고)
- **(a) M2rot vs M2(회전 없음) FP32 임베딩 cos 최소 ≥ 0.99999** (dev 64장, Pets·Caltech 32장씩).
  - 용의 요청 문구는 "vs 원래 FP32"였지만, M2 자체가 상수 캐시 근사라 원래 FP32 대비 cos 최소가 0.99940(`../G3R_regcache/g3a_mask/maskcheck.jsonl`)이어서
    회전과 무관하게 반드시 실패한다. 회전 동치를 보려는 의도에 맞게 비교 대상을 M2로 정했다 (실행 전, 설계 단계). M2rot vs 원래 FP32 값은 기록만 한다 (기대 ≈ 0.9994).
- (b) 회전 모델에서 다시 계산한 K_c·V_c가 원래 캐시와 같음: 층·헤드 전체 최대 상대오차 ‖ΔK‖/‖K‖ ≤ 1e-4.
- (c) 그래프: 잔차 경로 위에 새 연산 없음(잔차 Add의 h 입력이 잔차 Add·위치 임베딩뿐), 연산 종류·개수가 M2 b1과 같음.
- (d) sink 행의 1/√(분산+ε) ≤ 일반 토큰 최댓값 (위쪽만, 블록 5~11 norm1·norm2와 마지막 norm, dev 64장).
- (e) `hailo parser onnx ... --hw-arch hailo10h` 통과 (rc=0, end node = 모델 출력).

## 4. 판정 규칙 (결과 전 고정)
- Hailo 구성 `.alls`: "M2 + D256L1 16비트 구성"과 같은 의미 — conv1, 잔차 ew_add 26개(위치 임베딩 1 + 잔차 24 + 풀링 1), 블록 0 fc1·fc2·블록 5 fc1·fc2·블록 9 fc2 16비트.
  층 이름은 이 HAR에서 ONNX 노드 기준으로 다시 매칭하고 매칭 표를 남긴다. 나머지: 최적화 수준 1, 보정 256장 s0, LN 분해, equalization 끔 (D256L1과 같음).
- optimize → 출력 SNR 기록 → 에뮬레이터 스크리닝 (dev 6종 × 앞 200장, SDK_QUANTIZED).
- 스크리닝 표 (같은 PC): rothad_D256L1 0.9775 · M2_D256L1 0.9803 · M2rot_D256L1 (고화질 cos).
- **컴파일 대상: M2_D256L1과 M2rot_D256L1 중 고화질 cos가 높은 쪽 하나. 차이가 0.002 미만이면 연산이 적은 M2_D256L1.**
- 컴파일 뒤 HEF 크기·컨텍스트 수(profiler)를 rothad_D256L1(107,540,480 B, 17 컨텍스트, `../G3Q_rotate/hailo/README.md`)과 비교. **컴파일까지 하고 멈춘다.**
- 다음 단계 (용 확인 뒤, 이번에는 하지 않음): AGX 실칩 dev 전체, rothad_D256L1 vs 선택 후보 짝 부트스트랩 95% 구간(시드 0, 1000회). 하한 > 0이면 "유의하게 좋음"으로 팀에 제안. 지연도 같이.

## 5. 공통
- DFC는 모두 `flock ~/siglip2_work/dfc.lock` 안, 시작 전 `free -g` available ≥ 15 GB. hailo 로그는 저장소 밖 폴더.
- 새 기능은 기본 꺼진 옵션만. 기본 export sha256이 이전과 같은지 확인한다.

## 6. 결과 (10/18, 연구실 데스크탑 WSL Ubuntu-22.04, DFC 5.4.0 CPU)
### 사전 확인 (`m2rotcheck.jsonl`, `graph_check.txt`, `parse.log`) — 모두 통과
| 확인 | 결과 |
|---|---|
| (a) M2rot vs M2 FP32 cos 최소 (dev 64장) | **0.9999998** (기록: vs 원래 FP32 0.99940, M2도 0.99940) |
| (b) 회전 모델 K_c·V_c vs 원래 캐시 최대 상대오차 | K 2.64e-7, V 5.46e-7 (≤ 1e-4) |
| (c) 그래프 | 잔차 경로 위 새 연산 없음. 연산 개수는 Identity만 달랐다(63 vs 15 — 회전으로 접은 LN γ=1·β=0 상수를 exporter가 합치고 Identity로 이음). 두 파일 모두 `s2_rotate.undedup_initializers`로 풀면 연산 종류·개수가 같다 |
| (d) sink 행 1/√(분산+ε) ≤ 일반 최대 | 15/15 통과 (sink 최대 1.91) |
| (e) DFC 파싱 | rc=0, end node `/trunk/Gather`, native cos 0.99993. HAR 324층 (M2 349층 — 회전으로 LN affine `normalization` 층 25개가 빠짐) |
- 캐시: `regcache_55at5-13at9_rothad_highres256_s0.npz` (`c205e3a0…`), `regcache_uc_55at5-13at9_rothad_highres256_s0.npz` (보정 train 256장 s0). 기본 `--rot none` export sha256은 이전 M2와 같음(`0e60fccc…`·`db66ec35…`).
- export: `siglip2_image_fp32_rcmaskbcmlp55at5-13at9_rothad(_b1).onnx` 354.39·354.36 MiB, ORT vs PyTorch 0.9999998.

### Hailo (`hailo/`, `.alls` = `siglip2/hailo/g3r_M2rot_D256L1equiv.alls`, 층 매칭 `layer_matching.csv` — ew_add 57 = 위치 1 + 잔차 25 + 마스크 31, conv 6개 역할 확인)
| 구성 (같은 PC, dev 6종 × 200장) | 출력 SNR | 고화질 cos | Δtop-1 | 출처 |
|---|---|---|---|---|
| rothad_D256L1 (D-042 현재 Hailo 최종) | — | 0.9775 | +0.4 | 회전 세션 `~/siglip2_work/hailo_g3q/screen_rothad_D256L1.log` |
| M2_D256L1 | 16.41 dB | 0.9803 | +1.1 | `../G3R_regcache/g3b_m2_dl/` |
| **M2rot_D256L1** | **18.03 dB** | **0.9866** | −0.2 | `hailo/screen.jsonl` (커밋 `0a6aa23`, dirty 아님) |
- M2rot 데이터셋별 cos: CIFAR A 0.960·Pets 0.990·Flowers 0.988·Cars 0.987·Food 0.984·Caltech 0.985 (rothad: 0.954·0.984·0.981·0.975·0.975·0.974 — 6종 모두 M2rot이 높음).
- Δtop-1은 200장 표본이라 구간이 ±1%p 정도다. 판정은 4절대로 cos로 한다.
- **컴파일 대상 판정: M2rot_D256L1** (M2보다 cos +0.0063 ≥ 0.002).

### 컴파일 (`hailo/compile/`)
| HEF | compile | 크기 | 컨텍스트 | sha256 |
|---|---|---|---|---|
| rothad_D256L1 (G3Q) | 25분 19초 | 107,540,480 B (107.5 MB) | 17 | `7243b76c…` |
| **M2rot_D256L1** | rc=0, 30분 55초 | **110,436,352 B (110.4 MB, +2.9 MB, +2.7%)** | **21 (+4)** | `94fff0976ac43480…` |
- 컨텍스트가 4개 늘어서 실칩 지연이 늘 수 있다 (추정, 확인 안 함). 다음 단계 실칩 비교에서 지연을 같이 잰다.
- HEF: `~/siglip2_work/g3b_20261018_215006/comp_M2rot_D256L1equiv/siglip2_m2rot.hef` (커밋하지 않음).

### 다음 (용 확인 뒤)
- AGX 실칩 dev 전체: rothad_D256L1 vs M2rot_D256L1, 짝 부트스트랩 95% 구간(시드 0, 1000회), 하한 > 0이면 "유의하게 좋음"으로 팀에 제안. 지연(배치 1, monitor 끔)도 같이.
