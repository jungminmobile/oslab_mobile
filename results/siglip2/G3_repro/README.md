# G3 재현 테스트: 새 가상환경에서 export·양자화·평가 (10/14 18:19~23:47, `siglip2/scripts/g3_repro_venv.sh`)

ROADMAP G3 완료 기준: "새 venv에서 같은 정수 커널 정확도가 재현된다".

## 방법
1. 새 가상환경 `D:\mokerton\venvs\siglip2_repro`에 `siglip2/requirements.txt`를 그대로 설치했다 (pip 캐시 `D:\mokerton\cache\pip`, `PYTHONUTF8=1`). 첫 시도는 requirements의 한글 주석을 pip 24.0이 cp949로 읽다가 실패했다.
2. 새 venv로 FP32 ONNX를 다시 export했다 (`RN50_ONNX=$SIGLIP2_ARTIFACTS/onnx_repro`, 원본 폴더는 건드리지 않음).
3. 새 venv로 fc2L2(S8) 양자화를 같은 인자로 다시 했다 (보정 256장, `calib_seen=[256]`).
4. 새 venv로 그 모델의 dev 전체 사진별 예측(`s2_paired_ci.py`)을 구해 원래 결과(`../G3_cumulative/full`)와 비교했다.

## 결과
| 확인 | 결과 |
|---|---|
| FP32 ONNX sha256 | 원본과 같음 `333cad08…` (G1 export 기록과도 같음) |
| FP32 ONNX(배치 1) sha256 | 같음 `ce2b66b9…` |
| fc2L2 S8 양자화 모델 sha256 | 같음 `e34d39c3…` |
| dev 6종 5,923장 예측 | 모두 같음, 임베딩 cos 최대 차이 0 (`pred_compare.txt`) |
| 패키지 | `env.json`의 핵심 패키지 버전이 같다. `pip freeze` 차이는 `truststore` 0.10.4 → 0.10.5 하나뿐이다 (pip의 SSL 보조 패키지, 결과와 무관) |

- 커밋은 시작 `49495c5`이고, 모든 단계가 dirty=false다.
- 같은 결과로 `s2_quantize.py`의 새 옵션(`--attn-u8`, `--u8-match`, `--reduce-range`, `--weight-type`)을 모두 끈 기본 동작이 이전과 같은 모델을 만든다는 것도 확인됐다 (협업 규칙 2).

## 범위 (이 결과가 말하지 않는 것)
- **같은 PC**(i7-1165G7, AVX512-VNNI), 같은 pip 캐시, 같은 데이터·보정 캐시(`$SIGLIP2_ARTIFACTS/cache`)를 썼다.
- 다른 CPU(특히 VNNI 없는 x86)나 다른 OS에서의 재현은 확인하지 않았다 (G6에서 학술부 PC 대비, CLAUDE.md 3절).
