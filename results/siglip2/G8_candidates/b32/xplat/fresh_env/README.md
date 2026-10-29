# B/32 제출 모델 — 새 clone·새 가상환경에서 보고서 9장 명령 그대로 재현 (2026-10-23 10:03~11:15, 이 PC)

- 스크립트 `siglip2/scripts/g8_b32_fresh_env_repro.sh` (커밋 6593254). 폴더 `D:\mokerton\repro_fresh_b32_1023_1003`.
- 순서 (학술부가 할 순서):
  1. GitHub에서 새로 clone (브랜치 `siglip2/report-b32`, 커밋 b7891cd)
  2. 시스템 Python 3.11.9로 새 venv → `pip install -r siglip2/requirements.txt` (`freeze.txt`: onnxruntime 1.25.0, torch 2.14.0+cpu, numpy 2.4.4, open_clip 3.3.0, timm 1.0.30)
  3. Release `siglip2-b32-submit-20261023` 받기 → `sha256sum -c` 13개 OK
- (가) 빠른 확인: O-AB Δ +0.4, cos 0.9893 — 이 PC 기준값과 같고 **사진 1,200장 예측 비트 동일**
- (나) dev 전체: 새 산출물 폴더에 `s2_build_dev.py all`로 캐시·텍스트 분류기·FP32 기준을 처음부터 만듦.
  - 고화질 5종 학습 parquet은 Hugging Face에서 **새로 내려받음**: Pets 1, Flowers 2, Cars 2, Food 8, Caltech 1개 파일. CIFAR·HF 모델 가중치는 이 PC 것을 다시 씀.
  - 결과: O-AB −0.27 [−0.62, +0.09], cos 0.9893 — 기존 결과와 같고 **6개 데이터셋 사진별 예측 모두 비트 동일**
- 결론: 저장소 코드 + requirements + Release 파일 + 공개 데이터만으로 같은 PC에서 dev 결과가 사진 단위까지 그대로 재현된다. 다른 CPU는 팀원 결과로 확인 예정.
