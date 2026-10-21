"""check_fp32_equal.py — E12-2 관문 G2: 원본 FP32 vs 합친 FP32의 출력 같음 확인 (계획서 4절)

고화질 dev 5종 × 앞 N장 (s2common 캐시, SigLIP2 전처리 끝난 float32), batch 1, C0 세션.
통과: 사진별 cos(정규화 임베딩) 최소 ≥ 0.99999, top-1 예측 일치율 ≥ 99.5% (5종 전체).
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_opts as SO          # noqa: E402
sys.path.insert(0, str(SO.REPO.parent / "oslab_g8" / "siglip2"))
import s2common as S               # noqa: E402

RC = S.RC
DATA = ["pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]


def embed(sess, x):
    name = sess.get_inputs()[0].name
    e = np.concatenate([sess.run(None, {name: np.ascontiguousarray(x[i:i + 1], np.float32)})[0] for i in range(len(x))])
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    xs = {d: np.asarray(np.load(RC.CACHE_DIR / f"{d}_x.npy", mmap_mode="r")[:a.n]) for d in DATA}
    emb = {}
    for k, p in (("a", a.a), ("b", a.b)):
        t = time.perf_counter()
        s = SO.make_session(p, "C0", a.threads)
        emb[k] = {d: embed(s, x) for d, x in xs.items()}
        del s
        print(f"{k} done {time.perf_counter() - t:.1f}s", flush=True)
    rows, same, tot = {}, 0, 0
    for d in DATA:
        e0, e1 = emb["a"][d], emb["b"][d]
        W, y = np.load(RC.CACHE_DIR / f"{d}_clf.npy"), np.load(RC.CACHE_DIR / f"{d}_y.npy")[:a.n]
        p0, p1 = np.argmax(e0 @ W, 1), np.argmax(e1 @ W, 1)
        same += int((p0 == p1).sum())
        tot += len(p0)
        rows[d] = {"max_abs_diff": float(np.abs(e0 - e1).max()), "min_cos": float((e0 * e1).sum(1).min()),
                   "pred_same": round(float((p0 == p1).mean()), 4),
                   "top1_a": round(100 * float((p0 == y).mean()), 2), "top1_b": round(100 * float((p1 == y).mean()), 2)}
        print(d, json.dumps(rows[d]), flush=True)
    min_cos = min(r["min_cos"] for r in rows.values())
    agree = same / tot
    res = {"date": datetime.now().isoformat(timespec="seconds"), "a": Path(a.a).name, "b": Path(a.b).name,
           "n_per_dataset": a.n, "batch": 1, "threads": a.threads, "combo": "C0", "per_dataset": rows,
           "min_cos": min_cos, "pred_agree": round(agree, 5), "git": SO.git_state(),
           "G2b_pass": bool(min_cos >= 0.99999 and agree >= 0.995)}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("min_cos", min_cos, "agree", agree, "G2b_pass", res["G2b_pass"])


if __name__ == "__main__":
    main()
