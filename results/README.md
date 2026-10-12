# results — 현재 실험 결과

`PRD.md` 6절의 실험 ID별로 폴더를 만든다. 보고서 수치는 이 폴더의 값만 쓴다.

```
results/
├── E0_measure/       S8S8·reduce_range·U8U8, 측정 환경 기록
├── E1_dfc_gate/      DFC parse·optimize·compile 로그, 노드↔층 매핑표
├── E2_calib_method/  MinMax·Percentile·Entropy
├── E3_calib_data/    보정 출처·장수·시드
├── E4_mixed/         블록 민감도, 용량-정확도 곡선
├── E5_dfq/           CLE(E5a) + Bias Correction(E5b), 선택: AdaRound(E5+)
├── E6_select/        최종 후보 선택 근거
├── E7_test/          최종 test (1회, 두 PC)
├── E8a_hailo_candidates/  후보 .hef 컴파일, 에뮬레이터 정확도
├── E8b_hailo_final/       최종 .hef 패키지
└── E9_4bit/              (여유 시) ORT AttnPool 4비트, Hailo a8_w4
```

## 기록 규칙

- 모든 jsonl 한 줄에 다음 필드를 넣는다: `date`, `commit`(git 짧은 해시), `dirty`(커밋 안 된 변경이 있으면 true, 보고서에 쓰지 않음), `options`(사용한 명령줄 옵션), `model`, `data`, `n`, `top1`, `top5`, `cos_mean`, `MiB`, `exec`(항상 `int`), `qtype`(S8S8 등), `calib`(방법·출처·장수·시드), `cpu`(모델명), `vnni`, `ort`, `os`, `threads`
- 결과 위치: 코드의 기본 저장 위치(`rn50/results/`)는 gitignore 대상이다. 실행 때 `RN50_RESULTS=results/<실험 폴더>`(예: `results/E2_calib_method`)로 지정해 이 폴더에 직접 쌓는다.
- 폴더 이름은 위 목록 그대로 쓴다(예: E7 결과는 `results/E7_test/`).
- 실험마다 `README.md`를 둔다. 내용: 실행 명령, 결론 3줄, 표
- 데이터·onnx·hef 같은 큰 파일은 커밋하지 않는다(`.gitignore`). 다시 만드는 명령만 남긴다.
