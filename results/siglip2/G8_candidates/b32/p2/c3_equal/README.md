# P2 출력이 C3(enable_mem_pattern=False)에서도 같은지 (2026-10-27 20:41~20:43, `g8_b32_p2_c3_check.sh` ①②, 커밋 `4e2897b` dirty 0)
- ① E12-1 `check_equal.py`(정민 스크립트, `exp/E12-1-jjury` 7228d87 그대로): P2 C0 vs C3, 고화질 5종 × 200장 → **max_abs_diff 0.0 (비트 동일)**, G1_pass True (`equal.json`)
- ② `portable_eval.py --no-mem-pattern` (dev200 1,200장, `../xplat_i7_c3/`) vs 기본 세션 (`../xplat_i7/`) → FP32·P2 모두 **사진별 예측·cos 비트 동일** (`../xplat_i7_c3/compare_default.txt`)
- → C3는 출력에 영향이 없다. 기본 세션으로 잰 dev·test 정확도는 제출 실행 설정(C3)에서도 같다.
