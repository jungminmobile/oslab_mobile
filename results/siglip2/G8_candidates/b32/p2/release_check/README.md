# 최종 제출 후보 Release `siglip2-b32-p2-submit-20261027`로 새 폴더 빠른 확인 (2026-10-27 21:14~21:20, 이 PC)
- 스크립트 `siglip2/scripts/g8_b32_p2_release_check.sh` (커밋 `c5e000c`): 빈 새 폴더 `D:/mokerton/repro_p2_release_1027_2114`에 Release만 받아 Release 설명·보고서 9.2 (가) 명령 그대로 실행
- `sha256sum -c SHA256SUMS.txt`: **15개 모두 OK** (`sha_check.txt`)
- `portable_eval.py --no-mem-pattern` (Release 안 스크립트, 4스레드): P2 고화질 Δ +0.4, cos 0.9884, `mem_pattern: false` 기록 — Release 안 기준 결과(`result_i7-1165G7_p2.json`)와 같음
- **FP32·P2 사진별 예측·cos가 Release 안 `pred_i7-1165G7_*.npz`와 비트 동일** → 같은 PC에서 Release와 절차만으로 완전히 재현된다
