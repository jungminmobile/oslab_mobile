"""
s2_g6b_ort.py — G6b ORT CPU 정확도: B16·B32의 FP32와 INT8을 같은 eval 코드·dev 전체로 (README 3-2)
====================================================================================================
- 실행: ORT 기본 설정 `ORT_ENABLE_ALL` (정수 커널), CPUExecutionProvider. 정확도용이라 스레드 수는 결과에 영향이 없다.
- 데이터: G6 Zoo 전처리 dev 전체 캐시 (`~/siglip2_work/g6/dev_zoo{224,256}`, uint8 픽셀 → x/127.5 − 1)
- 분류기: G6 `s2_g6.py fp32`가 만든 HF 텍스트 분류기 (`~/siglip2_work/g6/ref/<b16|b32>_<set>_clf.npy`)
- 지표: 고화질 5종 평균 top-1, 같은 모델의 ORT FP32 대비 Δtop-1(s2_paired_ci.boot_hr), cos, CIFAR A 따로
[사용]  (siglip2/)  RN50_RESULTS=../results/siglip2/G6b python s2_g6b_ort.py \
          --models b16_fp32=<onnx>:224:b16 b16_int8=<onnx>:224:b16 b32_fp32=<onnx>:256:b32 b32_int8=<onnx>:256:b32 \
          --pairs b16_int8-b16_fp32 b32_int8-b32_fp32
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s2common as S                     # noqa: E402
from s2_paired_ci import boot_hr         # noqa: E402
from scripts import env_info             # noqa: E402

DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HR = DATA[1:]
WORK = Path(os.environ.get("G6_WORK", Path.home() / "siglip2_work" / "g6"))
l2n = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def embed(path, size, threads, bs):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    out = {}
    for d in DATA:
        pix = np.load(WORK / f"dev_zoo{size}" / f"{d}_pix.npy", mmap_mode="r")
        e = []
        for i in range(0, len(pix), bs):
            x = (np.asarray(pix[i:i + bs], np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2).copy()
            e.append(s.run(None, {"image": x})[0])
        out[d] = np.concatenate(e).astype(np.float32)
        print(Path(path).name[:60], d, out[d].shape, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="이름=ONNX경로:크기:분류기모델(b16|b32)")
    ap.add_argument("--pairs", nargs="+", required=True, help="int8이름-fp32이름")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    out = S.require_results_dir()
    out.mkdir(parents=True, exist_ok=True)
    env = env_info.collect()
    env_info.write_env(out, env)
    base = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"], "dirty": env["git"]["dirty"],
            "exec": "ort_enable_all", **env_info.flat(env), "cpu_flags": env["cpu"]["flags"]}
    import onnxruntime as ort
    E, P, rows = {}, {}, []
    for kv in a.models:
        name, spec = kv.split("=", 1)
        path, size, ref = spec.rsplit(":", 2)
        size = int(size)
        t0 = time.time()
        emb = embed(path, size, a.threads, a.bs)
        np.savez_compressed(WORK / f"g6b_ort_emb_{name}.npz", **emb)
        E[name], P[name] = emb, {}
        r = {}
        for d in DATA:
            y, W = np.load(WORK / "ref" / f"{ref}_{d}_y.npy"), np.load(WORK / "ref" / f"{ref}_{d}_clf.npy")
            assert len(y) == len(emb[d]), (name, d)
            pred = np.argmax(l2n(emb[d]) @ W, 1)
            P[name][d] = (pred == y).astype(float)
            r[d] = {"n": int(len(y)), "top1": round(100 * float(P[name][d].mean()), 2)}
        rows.append({"what": "g6b_ort", "model": name, **base, "onnx": Path(path).name, "sha256": sha256(path),
                     "bytes": Path(path).stat().st_size, "size": size, "classifier": ref, "ort": ort.__version__,
                     "hr_top1": round(float(np.mean([100 * P[name][d].mean() for d in HR])), 2),
                     "cifar_A_top1": r["cifar_A"]["top1"], "per_set": r, "sec": round(time.time() - t0)})
        print(json.dumps(rows[-1], ensure_ascii=False)[:600], flush=True)
    for pr in a.pairs:
        q, f = pr.split("-")
        diff = {d: P[q][d] - P[f][d] for d in HR}
        cos = {d: float((l2n(E[q][d]) * l2n(E[f][d])).sum(1).mean()) for d in DATA}
        cmin = min(float((l2n(E[q][d]) * l2n(E[f][d])).sum(1).min()) for d in HR)
        rows.append({"what": "g6b_ort_pair", "int8": q, "fp32": f, **base,
                     "hr_dtop1": round(float(np.mean([100 * v.mean() for v in diff.values()])), 2), "ci95": boot_hr(diff),
                     "cifar_A_dtop1": round(100 * float((P[q]["cifar_A"] - P[f]["cifar_A"]).mean()), 2),
                     "hr_cos": round(float(np.mean([cos[d] for d in HR])), 4), "hr_cos_min": round(cmin, 4),
                     "cifar_A_cos": round(cos["cifar_A"], 4)})
        print(json.dumps(rows[-1], ensure_ascii=False)[:600], flush=True)
    with open(out / "ort_accuracy.jsonl", "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
