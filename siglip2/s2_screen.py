"""
s2_screen.py — 민감도 스크리닝: dev 6종 각 앞 K장으로 여러 INT8 구성의 cos·top-1 (정수 커널) (ROADMAP G3, D-024)
=========================================================================================================

[목적]
  블록·연산 유형 민감도처럼 구성이 많은 실험을 빠르게 잰다. 정수 커널(ORT_ENABLE_ALL)만 쓴다 (D-008).
  주 지표는 FP32 임베딩과의 cos (PRD 5절: "층 민감도 분석의 주 지표"). top-1은 참고.
  상위 후보는 rn50/evaluate.py(s2_evaluate.py)로 dev 전체를 다시 잰다.

[표본]  각 dev 캐시의 **앞 K장** (기본 200). dev 캐시는 해시 선택 순서라 클래스가 섞여 있다.
  "고화질 cos" = 고화질 5종 cos 평균의 단순 평균 (PRD 5절의 고화질 평균과 같은 방식).

[사용]
  RN50_RESULTS=../results/siglip2/G3_sensitivity "$SIGLIP2_PYTHON" s2_screen.py --k 200 --models a.onnx b.onnx --tag block
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2common as S          # noqa: E402
from scripts import env_info  # noqa: E402

RC = S.RC
DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HR = DATA[1:]


def screen(model, k, threads):
    row = {}
    for d in DATA:
        f = lambda s: RC.CACHE_DIR / f"{d}_{s}.npy"
        x = np.asarray(np.load(f("x"), mmap_mode="r")[:k])
        y, W, ref = np.load(f("y"))[:k], np.load(f("clf")), np.load(f("ref_fp32"))[:k]
        feat = RC.embed(RC.onnx_path(model), x, threads, fake_quant=False)       # 정수 커널
        m = RC.topk_metrics(feat, W, y, ref)
        fp_top1 = RC.topk_metrics(ref, W, y)["top1"]
        row[d] = {"n": int(len(y)), "top1": m["top1"], "fp32_top1": fp_top1, "cos_mean": m["cos_mean"], "cos_p1": m["cos_p1"]}
    row["hr_cos"] = round(float(np.mean([row[d]["cos_mean"] for d in HR])), 4)
    row["hr_dtop1"] = round(float(np.mean([row[d]["top1"] - row[d]["fp32_top1"] for d in HR])), 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default="", help="실험 묶음 이름 (block, optype, cumulative 등)")
    a = ap.parse_args()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    for m in a.models:
        t0 = time.time()
        p = RC.onnx_path(m)
        r = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"],
             "dirty": env["git"]["dirty"], "tag": a.tag, "model": p.name, "MiB": round(p.stat().st_size / 2**20, 2),
             "k": a.k, "exec": "int", "threads": a.threads, "options": vars(a), **env_info.flat(env),
             **screen(m, a.k, a.threads), "sec": None}
        r["sec"] = round(time.time() - t0)
        print(json.dumps({k: r[k] for k in ("model", "MiB", "hr_cos", "hr_dtop1", "sec")}, ensure_ascii=False), flush=True)
        with open(out / "screen.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
