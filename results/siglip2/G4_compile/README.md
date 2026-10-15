# G4 컴파일: DFC 구성을 Hailo-10H용 HEF로 (2026-10-14, 진행 중)

**결론 (구성 D): 컴파일 성공. HEF 99.8 MB, 컨텍스트 17개 (프로파일러 기준), 2시간 0분.** G1 기본 8비트(HEF 69.2 MB, 컨텍스트 18개)보다 HEF가 44% 크다. 16비트로 둔 층(잔차 합 `ew_add` 전체, patch embedding, 블록 0·5 MLP, 블록 9 fc2)의 가중치·활성값 때문이다. 컨텍스트 수는 비슷하다.

## 1. 왜 했나

G4 스크리닝(`../G4_dfc_screen`)에서 구성 D(Hailo 공식 패턴 + 블록 9 fc2 16비트)가 DFC 구성 중 정확도가 가장 높았다 (고화질 cos 0.952, Δtop-1 −1.1%p [−2.3, 0.0], 200장×5종).

에뮬레이터 정확도만으로는 실기기에서 쓸 수 있는지 알 수 없다. 그래서 다음 세 가지를 확인했다.
- 실제로 `.hef`가 만들어지는지
- HEF 크기와 컨텍스트 수 (컨텍스트가 많을수록 칩 안팎 전환이 늘어 느려진다)
- 컴파일 시간

예선 요건("Hailo DFC로 `.hef` 컴파일이 가능해야 한다", CLAUDE.md)과도 직결된다.

## 2. 실행

- 스크립트: `siglip2/hailo/overnight_wsl2.sh`의 `compile` 함수 (WSL Ubuntu-22.04, DFC 5.4.0, `--hw-arch hailo10h`, 공용 잠금 안에서)
- 입력: `~/siglip2_dfc/g4_20261013_221931/opt_D_official_b9fc2.har` (최적화 수준 0, 보정 64장, `.alls`는 `siglip2/hailo/g4_D_official_b9fc2.alls`)

```bash
hailo compiler opt_D_official_b9fc2.har --hw-arch hailo10h --output-dir . --output-har-path compiled_D_official_b9fc2.har
hailo profiler compiled_D_official_b9fc2.har --hef siglip2_b16_224.hef --out-path profiler.html --csv profiler.csv
```

## 3. 결과

| 구성 | 컴파일 | 시간 | HEF 크기 | 컨텍스트 (프로파일러) | HEF sha256 |
|---|---|---|---|---|---|
| G1 기본 8비트 (`../G1_dfc_gate`) | 성공 | 2시간 16분 | 69.2 MB | 18 | `6733511…3a17540` |
| **D 공식 패턴 + 블록 9 fc2** | 성공 | 2시간 0분 (분할·배치 1시간 56분) | **99.8 MB** (99,840,000 B) | **17** | `033c7265ca1d50f3c04bc11492c9c038f825c8ec6e80d83752901736a919c3d5` |
| C 공식 패턴 (10/15) | 성공 | 1시간 59분 (재실행; 첫 시도는 지연 측정을 먼저 하려고 중단) | 97.1 MB (97,095,680 B) | 16 | `dc0d39de66485338c6215292f4310573d27f24d6d3d685ee19ec135f7b82b4f0` |
| E 블록 0·5·9 fc2 (10/14) | 성공 | 1시간 33분 | 95.1 MB (95,137,792 B) | 16 | `920ff42b14bc6b1b6d60e337c411d6c21b643ac94d8f0c228e4d9dd184004ede` |
| **D + 보정 256장 + 최적화 수준 1** (10/15) | 성공 | 1시간 25분 | 99.8 MB (99,831,808 B) | 17 | `b33a266f694ad40b1498ca4a19ec4dfccc96620e5127586ba5e55c987e81d99a` |
| C + 보정 256장 + 최적화 수준 1 (10/15 17:06~18:47) | 성공 | 1시간 41분 (분할·배치 1시간 35분) | 97.1 MB (97,083,392 B, C보다 12,288 B 작음) | 16 | `86e5aae5656841a0f81fc5bbef509a7ddf161fcc6680be429f18710e42d3e32b` |

- **C256L1 (10/15)**: 스크리닝(`../G4_dfc_screen` 6-4절)에서 D256L1과 top-1 동률이었던 구성이다. 컴파일 전 추정(C와 같은 16비트 층이라 약 97 MB·16 컨텍스트)이 맞았다: 97.1 MB·16 컨텍스트로 D256L1(99.8 MB·17)보다 2.7 MB(2.7%) 작고 컨텍스트가 1개 적다. 입력 HAR `~/siglip2_dfc/g4_lvl1_20261015/opt_C256L1_official.har`(sha256 `81221319…`), 스크립트 `siglip2/hailo/g4_more5_wsl.sh`, 파일 `C256L1_official/`.
- 단일 컨텍스트는 자원 부족으로 실패했다 (로그의 여러 분할 시도에서 `lcus` 118~149/80 — 처음 본 값 120/80은 전체 로그, 저장한 끝 200 KB에는 118·149). 그래서 다중 컨텍스트로 컴파일됐다.
- 프로파일러 모델 정보: 22.73 GMACs(연산 기준), 92.22 M 파라미터. 처리량·지연은 N/A다 (HailoRT 런타임 측정이 필요하고, 실칩 측정은 본선에서 한다 — D-015).
- **컨텍스트 수 표기 주의**: 컴파일 로그에는 "Applying selected partition to 5 contexts", "… 6 contexts"가 나온다. 로그에는 단일 컨텍스트 시도 → 다중 컨텍스트 분할 과정이 최소 두 번 있어, 이 숫자는 그 일부 단계의 값으로 보인다 (추정). 전체 컨텍스트 수는 프로파일러의 `Number of Contexts 17`을 쓴다 (프로파일러 csv의 context_0~16, G1도 같은 기준으로 18).
- 파일: `D_official_b9fc2/summary.txt` (로그 요약), `files.txt` (파일 크기·sha256), `profiler.csv` (층별 자원), `compile_tail.log` (로그 끝 200 KB).
- HEF·HAR 본체는 WSL `~/siglip2_dfc/compile_20261014/D_official_b9fc2/`에 있고 커밋하지 않는다 (D-016).

## 4. 해석

- 16비트 층이 늘어 HEF가 커졌다 (+30.7 MB, +44%). 예선 기준(정확도·속도·용량)에서 "용량" 축에 해당하는 비용이다.
- 컨텍스트는 18 → 17로 비슷하다. 실칩 지연은 컨텍스트 전환 횟수의 영향을 크게 받으므로, 실칩 지연은 G1과 비슷할 것으로 예상한다. 이것은 추정이고 본선 실기기에서 확인해야 한다.
- 참고: Hailo 공식 `siglip_b_16_image_encoder`(224)는 Model Zoo 표에서 33.0 FPS(배치 1)다 (D-018). 같은 구조의 SigLIP v1이라 직접 비교는 아니다.

## 5. 남은 것

- (완료 10/15) D + 보정 256장 + 최적화 수준 1 컴파일: 99.8 MB(D보다 8,192 B 작음)·17 컨텍스트로 D와 사실상 같다. 이 한 사례에서는 보정·최적화 수준이 크기·컨텍스트를 거의 바꾸지 않았다. 가장 정확한 DFC 구성(cos 0.967)이 실기기용 HEF로 준비됐다.
- (완료 10/15) C·E 컴파일: C 97.1 MB·16 컨텍스트, E 95.1 MB·16 컨텍스트. 16비트 층이 많은 D(99.8 MB·17)가 가장 크다. 컨텍스트 수는 16~18로 비슷하다
- 첫 컴파일 1회는 정민 노트북 결과와 교차 확인 (D-027: G4 최종 후보로) — 최종 구성은 팀 결정
- 본선 Pi의 HailoRT 버전 확인 후 그 버전용 컴파일 경로 (`hailo/DFC_SETUP.md` 8장)

## 6. 3중 확인

1. 자기 검증: 컴파일 rc=0, `Successful Compilation`, HEF 파일 존재·sha256 기록, 프로파일러 rc=0.
2. 독립 재계산: 프로파일러의 파라미터 수 92.22 M을 ONNX 파라미터 수 92,884,224와 비교했다. 차이는 661,860개(0.7%)다. 원인(상수로 접히거나 세지 않는 층)은 확인하지 않았다. 컨텍스트 수는 컴파일 로그 표기가 아니라 프로파일러로 확인했다.
3. 독립 검토: 10/15 서브에이전트 검토 — HEF 바이트·sha256·컨텍스트(16/16/17/17)·컴파일 시간 모두 원본과 일치.

입력 HAR·명령 (C·E·D256L1): C `~/siglip2_dfc/g4_20261013_165132/opt_C_official.har`, E `~/siglip2_dfc/g4_20261013_221931/opt_E_fc2_b0b5b9.har`, D256L1 `~/siglip2_dfc/g4_lvl1_20261015/opt_D256L1_official_b9fc2.har`. 명령은 D와 같다 (`hailo compiler <har> --hw-arch hailo10h`, 이어서 `hailo profiler`). 스크립트: `siglip2/hailo/g4_more_wsl.sh`(E), `D:\mokerton\artifacts\siglip2\logs\compile_c_after_latency.sh`(C 재실행), `compile_d256l1.sh`(D256L1).
