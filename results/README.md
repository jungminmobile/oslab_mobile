# results — 현재 실험 결과

`PRD.md` 6절의 실험 ID별로 폴더를 만든다. 보고서 수치는 이 폴더의 값만 쓴다.

```
results/
├── E0_measure/       S8S8·reduce_range·U8U8, 측정 환경 기록
├── E1_dfc_gate/      DFC parse·optimize·compile 로그, 노드↔층 매핑표
├── E2_calib_method/  MinMax·Percentile·Entropy
├── E3_calib_data/    보정 출처·장수·시드
├── E4_mixed/         블록 민감도, 용량-정확도 곡선
├── E5_dfq/           CLE + Bias Correction
├── E6_select/        최종 후보 선택 근거
├── E7_test/          최종 test (1회, 두 PC)
├── E8a_hailo_candidates/  후보 .hef 컴파일, 에뮬레이터 정확도
├── E8b_hailo_final/       최종 .hef 패키지
└── E9_4bit/              (여유 시) ORT AttnPool 4비트, Hailo a8_w4
```

## 기록 규칙

- 모든 jsonl 한 줄에 다음 필드를 넣는다: `date`, `model`, `data`, `n`, `top1`, `top5`, `cos_mean`, `MiB`, `exec`(항상 `int`), `qtype`(S8S8 등), `cpu`, `vnni`, `ort`, `threads`
- 실험마다 `README.md`를 둔다. 내용: 실행 명령, 결론 3줄, 표
- 데이터·onnx·hef 같은 큰 파일은 커밋하지 않는다(`.gitignore`). 다시 만드는 명령만 남긴다.
