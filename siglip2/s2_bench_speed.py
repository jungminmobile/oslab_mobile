"""
s2_bench_speed.py — rn50/bench_speed.py를 SigLIP2 경로로 그대로 실행하는 래퍼 (코드 복사 없음, D-009)
=====================================================================================================

[목적]  지연 측정은 RN50과 같은 스크립트로 한다 (번갈아 측정, 평균·표준편차·p50·p95).
  PRD 5절 조건(워밍업 20, 200장 × 3회, 배치 1)을 지키는지 인자를 검사하고, env.json·runs.jsonl(커밋·dirty·argv)을 남긴다.
  rn50 bench_speed는 CIFAR dev A 앞 N장을 CLIP 전처리로 넣는다. 속도는 입력 값과 무관하고 크기(224)가 같으므로 그대로 쓴다.
  이 PC 값은 "비공식"이다 (D-010).

[사용]
  RN50_RESULTS=../results/siglip2/<단계>_speed "$SIGLIP2_PYTHON" s2_bench_speed.py --warmup 20 --n 200 --repeats 3 --threads 4 \
      --models fp32=siglip2_image_fp32.onnx int8=<int8 파일>
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
import bench_speed as RB      # noqa: E402  rn50/bench_speed.py
from scripts import env_info  # noqa: E402
assert Path(RB.__file__).resolve().parent == S.RN50_DIR.resolve(), RB.__file__

if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    k = ap.parse_known_args()[0]
    assert k.warmup >= 20 and k.n >= 200 and k.repeats >= 3, f"PRD 5절 조건 미달: {vars(k)} (워밍업 20, 200장 × 3회)"
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    with open(out / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
                            "dirty": env["git"]["dirty"], "argv": sys.argv[1:], "cpu": env["cpu"]["model"],
                            "power": env["power"], "official": False}, ensure_ascii=False) + "\n")
    RB.main()
