# G3R 0-b: `--src` 옵션을 끈 기본 경로 identity (연구실 데스크탑, 2026-10-18)

## 결론
- **코드 동일성: 통과.** 같은 PC·같은 패키지에서 지금 코드(`ef8328c`, `--src` 추가 후)와 추가 전 코드(`c887a42`, 준혁 마지막 커밋, git worktree)가
  비트 단위로 같은 ONNX를 만들었다: 둘 다 sha256 `410d7483d64d71548a9a50e529c679aaec12f9119d0709810aa1fcd413538eca`.
- **Release 값 `5eee7839…`와는 다르다. 원인은 PC(CPU) 차이로 기록한다.** 가중치·zero point·그래프 구조는 같고, 활성값 눈금(scale)만 아주 조금 다르다 (아래 표).
- 판정 방식은 G3R README 7절 "이 PC에서 5eee가 안 맞을 때의 대체 확인"(용 승인, 10/18)을 따랐다.

## 실행
```bash
# 지금 코드 (~/oslab_mobile_yong, ef8328c)
RN50_RESULTS=../results/siglip2/G3R_regcache/identity "$SIGLIP2_PYTHON" s2_quantize.py --calib-source highres --calib-n 256 --op-types MatMul \
  --method percentile --percentile 99.999 --hist-chunk 16 --act-type u8 --reduce-range \
  --exclude "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/" --tag idcheck_fc2L2
# --src 추가 전 코드 (git worktree ~/g3r_base_wt, c887a42), 같은 인자, --tag idcheck_base
```
- 로그: `idcheck_run.log`, `quantize_log.jsonl`(지금 코드), `quantize_log_base_c887a42.jsonl`(추가 전 코드). 두 실행 모두 보정 실제 사용 장수 `[256]`.
- 입력 FP32 ONNX: Release `siglip2_image_fp32.onnx` (sha256 `333cad08…`, Downloads에서 복사). 보정 후보 `calib_*_f32.npy`는 이 PC에서 `s2_build_dev.py`로 만든 것.

## 이 PC 결과(`410d…`)와 Release(`5eee…`) 비교
| 항목 | 결과 |
|---|---|
| 노드 수, 연산 종류별 개수 | 같음 (921개) |
| 가중치(양자화된 int8)·zero point | 모두 같음 |
| 활성값 눈금 (QuantizeLinear/DequantizeLinear 입력 1) | 127쌍이 다름. 상대 차이 중앙값 3.3e-7, 최대 5.5e-6 |
| 값이 같은 shape 상수 이름 | 10개가 다른 이름으로 남음 (값은 같음, 계산에 영향 없음) |
| pets_dev 앞 32장 출력 (ORT, 4스레드) | 비트 단위로는 다름, 정규화 전 임베딩 최대 절대 차 0.34 |

- 해석(가설): 보정 때 FP32 ORT 추론의 반올림 차이로 percentile 눈금이 아주 조금 달라졌다. 이 PC는 AVX2·AVX-VNNI만 있고 AVX-512가 없으며, 준혁 PC(i7-1165G7)는 AVX-512가 있다.
  보정 입력 사진 전처리(Pillow) 차이일 가능성도 완전히 배제하지는 못했다. dev 5종의 장수·보정 후보 수·FP32 top-1은 준혁 기록(`G2_fp32_dev`)과 같았다.
- 패키지: `../env_labpc/pip_freeze.txt`, `../env_labpc/requirements_vs_freeze.txt` (고정 버전 13개 모두 일치).

## G3R에 주는 의미
- G2 비교(R0 vs R1-INT8)는 **같은 PC에서 만든 INT8끼리** 해야 한다. 이 PC의 R0는 Release `5eee…` 파일 대신 이 PC에서 만든 `410d…`(= idcheck 파일)를 쓰거나,
  Release 파일을 쓰려면 R1도 같은 조건이어야 하는데 그건 불가능하다. 어느 쪽을 R0로 쓸지는 2단계 전에 용이 정한다.
