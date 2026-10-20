"""
s2_g6b_bench.py — G6b ORT CPU 지연: 입력 크기가 다른 모델(B16 224, B32 256)을 한 세션에서 번갈아 (README 3-2·5)
================================================================================================================
`rn50/bench_speed.py`의 main()은 모든 모델에 같은 224 입력 하나를 넣어서, 256 모델을 같은 세션에 넣을 수 없다.
공통 코드(rn50/)는 고치지 않고(D-009), 측정 부분을 **그대로 옮기고 입력만 모델별 크기로** 바꿨다:
  - 세션: `rn50/common.session(path, threads)` 그대로 import (intra = threads, inter = 1, ORT_ENABLE_ALL)
  - 예열 --warmup번 → --repeats번 × 모델을 번갈아 --n장 (한 장씩 run, perf_counter) — bench_speed.main과 같은 순서·타이머
  - 결과 필드 (ms_per_img_mean·std·min, lat_p50·p95, MiB)도 같다
  - 입력: bench_speed는 CIFAR dev A 앞 n장(CLIP 전처리)이다. 여기서는 같은 CIFAR dev A 앞 n장의 Zoo 전처리 캐시
    (G6 dev_zoo<크기>/cifar_A_pix.npy → x/127.5 − 1). 속도는 입력 값과 무관하다
  - s2_bench_speed.py와 같이 PRD 5절 조건(워밍업 20, 200장 × 3회)을 assert하고, runs.jsonl(커밋·dirty·argv)과 env.json을 남긴다.
[사용]  RN50_RESULTS=../results/siglip2/G6b/speed python s2_g6b_bench.py --warmup 20 --n 200 --repeats 3 --threads 1 \
          --models b16_fp32=siglip2_image_fp32.onnx:224 b32_fp32=/abs/path.onnx:256 ...
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
from scripts import env_info  # noqa: E402

C = S.RC
WORK = Path(os.environ.get("G6_WORK", Path.home() / "siglip2_work" / "g6"))


def path_of(f):
    p = Path(f)
    return p if p.is_absolute() else C.onnx_path(f)


def main():
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="이름=파일:입력크기")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    assert a.warmup >= 20 and a.n >= 200 and a.repeats >= 3, f"PRD 5절 조건 미달: {vars(a)} (워밍업 20, 200장 × 3회)"
    out_dir = S.require_results_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out_dir, env)
    with open(out_dir / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
                            "dirty": env["git"]["dirty"], "argv": sys.argv[1:], "cpu": env["cpu"]["model"],
                            "power": env["power"], "official": False}, ensure_ascii=False) + "\n")
    models, xs = {}, {}
    for kv in a.models:
        k, spec = kv.split("=", 1)
        f, size = spec.rsplit(":", 1)
        models[k] = path_of(f)
        pix = np.load(WORK / f"dev_zoo{int(size)}" / "cifar_A_pix.npy", mmap_mode="r")[: a.n]
        xs[k] = (np.asarray(pix, np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2).copy()

    # ---- 아래는 rn50/bench_speed.main과 같은 측정 (입력만 xs[k]) ----
    sess = {k: C.session(v, a.threads) for k, v in models.items()}
    for k, s in sess.items():                     # 예열
        x = xs[k]
        for i in range(a.warmup):
            s.run(None, {"image": x[i:i + 1]})

    per_img = {k: [] for k in models}
    lat = {k: [] for k in models}
    for r in range(a.repeats):
        for k, s in sess.items():                  # 번갈아 측정
            x = xs[k]
            t0 = time.perf_counter()
            for i in range(a.n):
                t1 = time.perf_counter()
                s.run(None, {"image": x[i:i + 1]})
                lat[k].append((time.perf_counter() - t1) * 1000)
            per_img[k].append((time.perf_counter() - t0) * 1000 / a.n)
            print(f"repeat {r + 1}/{a.repeats} {k:16s} {per_img[k][-1]:7.2f} ms/img", flush=True)

    out = {"date": datetime.now().isoformat(timespec="seconds"),
           "env": {"python": platform.python_version(), "onnxruntime": ort.__version__,
                   "cpu": platform.processor() or platform.machine(), "threads": a.threads},
           "cond": vars(a), "results": {}}
    print(f"\n{'model':16s} {'MiB':>7s} {'mean':>8s} {'std':>6s} {'min':>7s} {'p50':>7s} {'p95':>7s}")
    for k, v in models.items():
        L = np.array(lat[k])
        res = {"file": v.name, "MiB": round(v.stat().st_size / 2**20, 1), "input": list(xs[k].shape[1:]),
               "ms_per_img_mean": round(statistics.mean(per_img[k]), 2),
               "ms_per_img_std": round(statistics.stdev(per_img[k]), 2) if a.repeats > 1 else 0.0,
               "ms_per_img_min": round(min(per_img[k]), 2),
               "lat_p50": round(float(np.percentile(L, 50)), 2), "lat_p95": round(float(np.percentile(L, 95)), 2)}
        out["results"][k] = res
        print(f"{k:16s} {res['MiB']:7.1f} {res['ms_per_img_mean']:8.2f} {res['ms_per_img_std']:6.2f} "
              f"{res['ms_per_img_min']:7.2f} {res['lat_p50']:7.2f} {res['lat_p95']:7.2f}")
    fn = out_dir / f"speed_{datetime.now():%Y%m%d_%H%M%S}.json"
    json.dump(out, open(fn, "w"), indent=1, ensure_ascii=False)
    print("saved", fn)


if __name__ == "__main__":
    main()
