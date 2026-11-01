"""check_equal.py — E12-1 관문 G1: 세션 설정 C1~C4가 C0과 비트 단위로 같은 임베딩을 내는지 (계획서 5-1)

입력: s2common 캐시(SigLIP2 전처리 끝난 float32)의 고화질 dev 5종 × 앞 N장 (E11b 스크리닝과 같은 '앞 200장').
실행: batch 1 (메모리·속도 측정과 같은 조건), 정수 커널(ORT_ENABLE_ALL).
통과: 모든 조합·모든 사진에서 최대 절대 차이 = 0. 아니면 정확도 재측정(예비 단계 7)으로 넘어간다.
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
sys.path.insert(0, str(SO.S2_SIGLIP))
import s2common as S               # noqa: E402  캐시 위치·전처리 규칙 (계획서 3절)
from scripts import env_info       # noqa: E402

RC = S.RC
DATA = ["pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]


def embed(sess, x):
    name = sess.get_inputs()[0].name
    e = np.concatenate([sess.run(None, {name: np.ascontiguousarray(x[i:i + 1], np.float32)})[0] for i in range(len(x))])
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--combos", nargs="+", default=list(SO.COMBOS))
    ap.add_argument("--n", type=int, default=200, help="데이터셋마다 앞 N장")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.combos[0] == "C0", "첫 조합은 기준 C0이어야 함"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env["git_oslab_mobile"] = SO.git_state()
    env_info.write_env(out.parent, env)
    xs = {d: np.asarray(np.load(RC.CACHE_DIR / f"{d}_x.npy", mmap_mode="r")[:a.n]) for d in DATA}
    emb = {}
    for c in a.combos:
        t = time.perf_counter()
        sess = SO.make_session(a.model, c, a.threads)
        emb[c] = {d: embed(sess, x) for d, x in xs.items()}
        del sess
        print(f"{c} done {time.perf_counter() - t:.1f}s", flush=True)
    rows = {}
    for c in a.combos[1:]:
        rows[c] = {}
        for d in DATA:
            e0, e1 = emb["C0"][d], emb[c][d]
            W, y = np.load(RC.CACHE_DIR / f"{d}_clf.npy"), np.load(RC.CACHE_DIR / f"{d}_y.npy")[:a.n]
            p0, p1 = np.argmax(e0 @ W, 1), np.argmax(e1 @ W, 1)
            rows[c][d] = {"max_abs_diff": float(np.abs(e0 - e1).max()), "min_cos": round(float((e0 * e1).sum(1).min()), 8),
                          "pred_same": round(float((p0 == p1).mean()), 4),
                          "top1_C0": round(100 * float((p0 == y).mean()), 2), "top1": round(100 * float((p1 == y).mean()), 2)}
        rows[c]["identical"] = all(r["max_abs_diff"] == 0.0 for r in rows[c].values())
        print(c, "identical" if rows[c]["identical"] else "DIFF", json.dumps({d: rows[c][d]["max_abs_diff"] for d in DATA}), flush=True)
    res = {"date": datetime.now().isoformat(timespec="seconds"), "model": Path(a.model).name, "n_per_dataset": a.n,
           "datasets": DATA, "batch": 1, "threads": a.threads, "combos": {c: SO.describe(c) for c in a.combos},
           "git": env["git_oslab_mobile"], **env_info.flat(env), "compare_vs_C0": rows,
           "G1_pass": all(r["identical"] for r in rows.values())}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("G1_pass:", res["G1_pass"])


if __name__ == "__main__":
    main()
