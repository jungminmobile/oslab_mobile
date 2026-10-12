"""
s2_evaluate.py — rn50/evaluate.py를 SigLIP2 경로로 그대로 실행하는 래퍼 (코드 복사 없음, D-009)
=========================================================================================

[목적]
  rn50/evaluate.py는 캐시(<key>_x/_y/_clf/_ref_fp32.npy)만 같은 형식이면 모델과 무관하다.
  이 래퍼는 s2common을 먼저 import해서 RN50_* 경로를 SigLIP2 쪽으로 맞추고(덮어쓰기 방지 assert 포함),
  rn50 evaluate.main()을 같은 명령줄 인자로 실행한다. 정확도는 정수 커널(--exec int, D-008)로만 잰다.

[사용]
  RN50_RESULTS=../results/siglip2/<단계>_<주제> "$SIGLIP2_PYTHON" s2_evaluate.py \
      --data cifar_A pets_dev flowers_dev cars_dev food_dev caltech_dev \
      --models fp32 siglip2_image_int8_minmax_highres256_s0_s8_all.onnx --exec int --threads 4
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S   # noqa: E402,F401  (RN50_* 경로 설정 + 확인)
import evaluate as RE  # noqa: E402  rn50/evaluate.py
assert Path(RE.__file__).resolve().parent == S.RN50_DIR.resolve(), RE.__file__

if __name__ == "__main__":
    assert "--exec" in sys.argv and sys.argv[sys.argv.index("--exec") + 1] == "int", \
        "정확도는 정수 커널로만 잰다 (D-008): --exec int 를 명시"
    print(f"[s2_evaluate] cache={S.RC.CACHE_DIR} onnx={S.RC.ONNX_DIR} results={S.RC.RESULTS_DIR}", flush=True)
    RE.main()
