"""
s2_sessopt_check.py — ORT 세션 설정(session.x64quantprecision=1)이 정수 모델 출력을 바꾸는지 확인 (G3·G6 비-VNNI 대책)
==========================================================================================================

PRD 3절 기록: "session.x64quantprecision=1은 VNNI 없는 CPU에서만 U8U8 계산으로 바꾼다 (VNNI CPU에서는 출력 차이 0.0)".
이 PC(i7-1165G7)는 AVX512-VNNI가 있다. 그래서 같은 U8 모델을 기본 설정과 이 설정으로 돌려, 임베딩이 같은지 확인한다.
공통 평가 코드(rn50/common.py)는 세션 설정을 받지 않으므로, 여기서는 세션을 직접 만든다.

사용:
  RN50_RESULTS=../results/siglip2/G3_cumulative/qkv_precision "$SIGLIP2_PYTHON" s2_sessopt_check.py --model <onnx> --k 200
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
from scripts import env_info  # noqa: E402

RC = S.RC
DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]


def embed(path, x, entries, threads=4, bs=50):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    for k, v in entries.items():
        so.add_session_config_entry(k, v)
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    n = s.get_inputs()[0].name
    e = np.concatenate([s.run(None, {n: np.ascontiguousarray(x[i:i + bs], np.float32)})[0] for i in range(0, len(x), bs)])
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--k", type=int, default=200)
    a = ap.parse_args()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    rows = {}
    for d in DATA:
        x = np.load(RC.CACHE_DIR / f"{d}_x.npy", mmap_mode="r")[:a.k]
        e0 = embed(RC.onnx_path(a.model), x, {})
        e1 = embed(RC.onnx_path(a.model), x, {"session.x64quantprecision": "1"})
        W, y = np.load(RC.CACHE_DIR / f"{d}_clf.npy"), np.load(RC.CACHE_DIR / f"{d}_y.npy")[:a.k]
        p0, p1 = np.argmax(e0 @ W, 1), np.argmax(e1 @ W, 1)
        rows[d] = {"max_abs_diff": float(np.abs(e0 - e1).max()), "min_cos": round(float((e0 * e1).sum(1).min()), 6),
                   "pred_same": round(float((p0 == p1).mean()), 4), "top1_default": round(100 * float((p0 == y).mean()), 2),
                   "top1_x64qp": round(100 * float((p1 == y).mean()), 2)}
        print(d, rows[d], flush=True)
    res = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"], "dirty": env["git"]["dirty"],
           "model": a.model, "k": a.k, **env_info.flat(env), "datasets": rows,
           "identical": all(r["max_abs_diff"] == 0.0 for r in rows.values())}
    with open(out / "sessopt_check.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print("identical:", res["identical"])


if __name__ == "__main__":
    main()
