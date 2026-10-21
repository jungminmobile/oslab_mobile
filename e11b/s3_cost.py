"""
s3_cost.py — E11b S3 1단계: SigLIP2 R0 위 Linear 단위별 비용표 (E10 방식)
- 단위: 블록 0~11의 qkv·proj·fc1·fc2(블록 5·9 fc2는 R0에서 이미 FP32라 제외) + attn_pool의 q·kv·proj·fc1·fc2 = 51개
- 각 단위 u를 R0에서 추가로 FP32로 둔 모델을 만들고(s2_quantize, 같은 명령 + --exclude 추가)
  s_u = 스크리닝 cos(dev 6종 × 앞 200장, 정수 커널 4T) − R0 cos,  t_u = 4T 지연 − R0 (같은 묶음에서 번갈아, n=60, 3회 중앙값),
  m_u = 파일 크기 차이. 단위 모델은 측정 뒤 지운다(디스크).
사용: s2_venv python s3_cost.py [--start 0] [--group 6]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

S2 = Path(__file__).resolve().parents[2] / "oslab_s2" / "siglip2"
sys.path.insert(0, str(S2))
import s2common as S   # noqa: E402

RC = S.RC
R0 = "siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_fc2L2.onnx"
BASE_EX = "/blocks.5/mlp/fc2/,/blocks.9/mlp/fc2/"
Q = ["--calib-source", "highres", "--calib-n", "256", "--op-types", "MatMul", "--method", "percentile",
     "--percentile", "99.999", "--hist-chunk", "16", "--act-type", "u8", "--reduce-range"]
DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]


def units():
    U = []
    for b in range(12):
        for p in ("attn/qkv", "attn/proj", "mlp/fc1", "mlp/fc2"):
            if p == "mlp/fc2" and b in (5, 9):
                continue
            U.append((f"b{b}.{p.split('/')[1]}", f"/blocks.{b}/{p}/"))
    for p in ("q", "kv", "proj", "mlp/fc1", "mlp/fc2"):
        U.append((f"pool.{p.replace('mlp/', '')}", f"/attn_pool/{p}/"))
    return U


def quantize(tag, extra_ex):
    ex = BASE_EX + ("," + extra_ex if extra_ex else "")
    cmd = [sys.executable, "-u", str(S2 / "s2_quantize.py"), *Q, "--exclude", ex, "--tag", tag, "--overwrite"]
    r = subprocess.run(cmd, cwd=S2, capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr[-2000:]
    return f"siglip2_image_int8_percentile99999_highres256_s0_u8_ops-MatMul_hc16_rr_{tag}.onnx"


_X = {}


def screen_cos(model):
    cs = []
    for d in DATA:
        if d not in _X:
            _X[d] = (np.asarray(np.load(RC.CACHE_DIR / f"{d}_x.npy", mmap_mode="r")[:200]),
                     np.load(RC.CACHE_DIR / f"{d}_ref_fp32.npy")[:200])
        x, ref = _X[d]
        e = RC.embed(RC.onnx_path(model), x, 4, fake_quant=False)
        cs.append(float((e * ref).sum(1).mean()))
    return float(np.mean(cs)), cs


def lat(models, n=60, warm=10, rep=3):
    x = np.ascontiguousarray(np.load(RC.CACHE_DIR / "cifar_A_x.npy", mmap_mode="r")[:1], np.float32)
    so = ort.SessionOptions(); so.intra_op_num_threads = 4; so.inter_op_num_threads = 1
    ss = {m: ort.InferenceSession(RC.onnx_path(m).as_posix(), so, providers=["CPUExecutionProvider"]) for m in models}
    for s in ss.values():
        for _ in range(warm):
            s.run(None, {"image": x})
    res = {m: [] for m in models}
    for _ in range(rep):
        for m, s in ss.items():
            t = time.perf_counter()
            for _ in range(n):
                s.run(None, {"image": x})
            res[m].append((time.perf_counter() - t) * 1000 / n)
    return {m: statistics.median(v) for m, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--group", type=int, default=6)
    a = ap.parse_args()
    out = S.require_results_dir()
    U = units()
    c0, _ = screen_cos(R0)
    print("R0 screen cos", c0, flush=True)
    for g in range(a.start, len(U), a.group):
        grp = U[g:g + a.group]
        files = {}
        for name, pat in grp:
            files[name] = quantize("s3u_" + name.replace(".", "_"), pat)
        cos = {name: screen_cos(f) for name, f in files.items()}
        L = lat([R0, *files.values()])
        for name, pat in grp:
            f = files[name]
            row = {"unit": name, "pattern": pat, "cos": cos[name][0], "cos_by_data": cos[name][1], "s_u": cos[name][0] - c0,
                   "lat_ms": L[f], "lat_R0_ms": L[R0], "t_u": L[f] - L[R0],
                   "m_u_MiB": (RC.onnx_path(f).stat().st_size - RC.onnx_path(R0).stat().st_size) / 2**20,
                   "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
            with open(out / "s3_cost.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
            os.remove(RC.onnx_path(f))
        print(f"group {g // a.group + 1} done {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
