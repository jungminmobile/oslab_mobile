# RN50 B2 깨끗한 재측정 (2026-10-22, 쭈리 PC i7-1165G7, 비공식)

- 이유: 보고서 3.2·부록 C의 RN50 값이 정민 브랜치의 dirty 기록(4a8fa10, dirty true)과 raw 없는 258V 값이었음.
- 묶음: 정민 Release `rn50-b2-xcheck-20261021` (코드 511de3c, sha256 26개 확인). zip에 빠진 `rn50/prompts/cifar100_templates.json`은 정민 브랜치(= main과 같음)에서 넣음.
- 실행: `siglip2/scripts/g8_rn50_clean.sh` (커밋 4b23052, 시작 dirty=0). 정확도는 정민 코드 그대로(`evaluate.py --exec int --threads 4`, `bootstrap.py`), 묶음 폴더에서 실행해 저장소는 건드리지 않음. 정민 코드의 jsonl에는 우리 커밋이 안 찍히므로(`"git": null`) 커밋은 실행 로그(`g8_e12_rn50lat_console.log` 등)와 이 README로 남긴다.

## 정확도 (`evaluate.jsonl`, `bootstrap.jsonl`, 10/22 02:21~02:24)

| | FP32 | B2 | Δ [95%] |
|---|---|---|---|
| 고화질 5종 평균 | 71.51 | 70.48 | **−1.02 [−1.78, −0.28]** (유의) |
| Pets / Flowers / Cars / Food / Caltech | | | −2.02 / −0.84 / −1.65 / −0.51 / −0.10 |
| CIFAR A | 40.7 | 39.7 | −1.0 |
| cos | | 0.957~0.961 | |

- 정민 노트북 258V −1.43, 클라우드 Xeon −1.05와 다른 것은 CPU마다 정수 커널이 다르기 때문(정상 범위). B/16 M2rot_ort(−0.23, 유의 아님)를 고른 결론(D-045·D-046)은 그대로.

## 지연·메모리
- 03:08 첫 시도는 재부팅 직후 1.6배 느린 상태(B16 FP32 4T 274 ms)라 무효: `latency_run_INVALID_powermode_0308.log`, `latency/cross_latency_INVALID_powermode_0308.json`. 보고서에 쓰지 않는다.
- 유효 값은 8모델 같은 실행 측정 `../latency_this_pc/all8_20261022*` (O-AB 포함). RN50: FP32 107.3 / 63.0 / 47.5 ms → B2 46.1 / 28.4 / 20.9 ms (1/2/4T, FP32 대비 −58.0 / −56.3 / −56.0%), peak 316.2 → 113.0 MiB (−64.3%), 파일 146.1 → 37.7 MiB (−74.2%). 자세한 규칙은 `../b16/O_AB/README.md` 2절.
