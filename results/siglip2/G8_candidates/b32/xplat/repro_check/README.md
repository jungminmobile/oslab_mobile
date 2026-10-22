# B/32 제출 모델 — 깨끗한 폴더 재현 절차 확인 (2026-10-23 00:21~01:00, 이 PC)

- 스크립트 `siglip2/scripts/g8_b32_repro_check.sh` (커밋 6f47a85). 학술부가 하는 순서 그대로: 새 폴더 `D:\mokerton\repro_check_b32_1023_0021`에 Release `siglip2-b32-submit-20261023`만 내려받음 → `sha256sum -c` (13개 파일 모두 OK, `sha_check.txt`) → `portable_eval.py`(numpy+onnxruntime만) 실행. 저장소 코드·데이터 사용 안 함
- 결과: FP32·O-AB·M2rot 모두 이 PC 기준값(`../i7-1165G7/result.json`)과 **사진별 예측이 비트 단위로 같고 cos 차이 0** (dev 6종 × 200장)
  - O-AB 고화질 Δ +0.4, cos 0.9893 / M2rot −0.1, cos 0.9772
- 뜻: Release 파일과 절차만으로 같은 PC에서는 완전히 재현된다. 다른 CPU(정민 258V, 준용 285K·AGX ARM)는 팀원 결과를 받아 비교 예정
