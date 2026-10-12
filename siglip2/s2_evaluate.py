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

import argparse  # noqa: E402
import json      # noqa: E402
from datetime import datetime  # noqa: E402
from scripts import env_info   # noqa: E402

if __name__ == "__main__":
    # rn50 evaluate와 같은 argparse 규칙(같은 옵션이 여러 번이면 마지막 값, 축약형 허용)으로 --exec 값을 읽는다
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--exec", default="fake")
    exec_mode = ap.parse_known_args()[0].exec
    assert exec_mode == "int", f"정확도는 정수 커널로만 잰다 (D-008): --exec int (받은 값: {exec_mode})"
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    (out / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    # rn50 evaluate.jsonl 줄에는 커밋·옵션이 없으므로 실행마다 runs.jsonl에 따로 남긴다 (results/README.md 기록 규칙)
    with open(out / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
                            "dirty": env["git"]["dirty"], "argv": sys.argv[1:], "cpu": env["cpu"]["model"],
                            "vnni": env["cpu"]["flags"]["avx512_vnni"] or env["cpu"]["flags"]["avx_vnni"],
                            "ort": (env["onnxruntime"] or {}).get("version"), "os": env["os"]},
                           ensure_ascii=False) + "\n")
    print(f"[s2_evaluate] cache={S.RC.CACHE_DIR} onnx={S.RC.ONNX_DIR} results={out} commit={env['git']['commit']} "
          f"dirty={env['git']['dirty']}", flush=True)
    RE.main()
