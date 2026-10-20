"""chip_hostnorm.py — G6c: AGX에서 B16 입력의 호스트 정규화 시간 (README 3-2, 용 지시)
(a) eval 코드와 같은 구현: uint8 픽셀 → np.asarray(x, float32) / 127.5 − 1.0 (g6_hailo_chip.py eval, 정규화가 HEF 밖인 HEF)
(b) 참고: (a) + HEF 입력 quant_info로 UINT8 양자화 (g6_hailo_chip.py bench가 넣는 형식) — clip(round(v / s + zp), 0, 255)
배치 B = B장을 한 번에(NHWC 묶음) 처리한 시간. 워밍업 20번 → 200번 perf_counter, 중앙값. 세션 3번 (같은 프로세스 안, 세션 사이 1초).
사용: OPENBLAS_NUM_THREADS=1 python3 chip_hostnorm.py --data dev_zoo224 --scale <s> --zp <zp> --out out/hostnorm.json
"""
import argparse, json, platform, time
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--scale", type=float, required=True); ap.add_argument("--zp", type=float, required=True)
ap.add_argument("--out", required=True); ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
pix = np.load(Path(a.data) / "pets_dev_pix.npy", mmap_mode="r")[:8].copy()          # uint8 (8,224,224,3)
assert pix.dtype == np.uint8 and pix.shape[1:] == (224, 224, 3)
fa = lambda x: np.asarray(x, np.float32) / 127.5 - 1.0
fb = lambda x: np.clip(np.round(fa(x) / a.scale + a.zp), 0, 255).astype(np.uint8)
res = {"date": time.strftime("%F %T"), "machine": platform.machine(), "python": platform.python_version(), "numpy": np.__version__,
       "scale": a.scale, "zp": a.zp, "warmup": a.warmup, "n": a.n, "runs": []}
for s in (1, 2, 3):
    for b in (1, 8):
        x = pix[:b]
        for name, f in (("a_normalize", fa), ("b_normalize_quantize_u8", fb)):
            for _ in range(a.warmup):
                f(x)
            ts = []
            for _ in range(a.n):
                t0 = time.perf_counter(); f(x); ts.append((time.perf_counter() - t0) * 1000)
            row = {"session": s, "batch": b, "op": name, "median_ms": round(float(np.median(ts)), 4), "mean_ms": round(float(np.mean(ts)), 4),
                   "p90_ms": round(float(np.percentile(ts, 90)), 4)}
            res["runs"].append(row); print(json.dumps(row), flush=True)
    time.sleep(1)
Path(a.out).write_text(json.dumps(res, indent=1))
