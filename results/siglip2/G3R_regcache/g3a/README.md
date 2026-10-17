# G3a: R1 b1 ONNX의 Hailo DFC 파싱 (2026-10-18, 연구실 데스크탑 WSL Ubuntu-22.04)

## 결론: **실패** (README 5절 G3a 기준 "호스트로 떨어지는 구간 없이 파싱 성공"을 못 넘음)
- 대안 그래프(상수 K·V를 외부 입력으로)는 **아직 시도하지 않았다** (용 지시, 10/18). 막힌 연산만 기록한다.

## 실행
```bash
source ~/hailo10_venv/bin/activate && export CUDA_VISIBLE_DEVICES=-1
hailo parser onnx siglip2_image_fp32_rc55at5-13at9_b1.onnx --hw-arch hailo10h --net-name siglip2_rc55at5_13at9 < /dev/null
```
- 입력: `s2_regcache.py export`가 만든 `siglip2_image_fp32_rc55at5-13at9_b1.onnx` (354.48 MiB, sha256 `bd99eea74b7cc5da…`, `../export.jsonl`)
- Hailo DFC 5.4.0, `-y` 없음 (추천 end node 자동 수락으로 일부가 호스트로 빠지는 것을 막기 위해). 24초 만에 rc=1. 전체 로그 `parse.log`.

## 막힌 곳
| 순서 | DFC 메시지 | ONNX 노드 |
|---|---|---|
| 1 | shape 계산 중 `IndexError: list index out of range` (`hn_layers/matmul.py` `_calc_output_shape`, `input_shapes[1]`) | (MatMul 층, 이름은 로그에 없음) |
| 2 | `UnsupportedModelError: Unsupported dimensions at concat layer concat3 (translated from /trunk/Concat_2)` — `feature_splitter6 [-1,1,195,768]` + `const_input3 [-1,1,64,12]`, "concat은 feature 축(#3)으로만" | `/trunk/Concat_2` (axis 2): 블록 5 attention의 K `Squeeze_17`(Split 결과, 헤드×195토큰×64)에 캐시 상수 K `Expand_1`(init `[1,12,1,64]` → Where)를 **토큰 축으로** 붙이는 Concat |

- 해석: DFC는 (헤드, 토큰, 채널) 4차원 텐서의 **토큰 축 Concat**을 지원하지 않고, 상수 입력을 `[1,64,12]` 모양으로 다르게 해석했다.
  b1 파일에도 `Expand` 17개·`Where` 17개가 남아 있다 (README 7절에서 "b1에서는 상수로 접힐 것"이라고 예상했으나 접히지 않음, `../export.jsonl`).
- 처음 막힌 곳에서 파서가 멈추므로, 뒤쪽 블록(6~11·풀링 헤드)의 같은 종류 Concat도 같은 이유로 막힐 가능성이 크다 (확인 안 함).
