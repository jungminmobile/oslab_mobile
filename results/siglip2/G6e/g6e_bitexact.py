"""g6e_bitexact.py — G6e 4단계: optimized HAR을 DFC SDK_BIT_EXACT(와 SDK_QUANTIZED) 에뮬레이터로 돌려 저장된 실칩 출력과 비교한다 (칩·compile 안 씀)
입력은 g6c_emulate.py와 같다: dev_zoo224 `<set>_pix.npy`의 Zoo 전처리 픽셀을 x/127.5 − 1 (NHWC float32).
  probe : 층마다 acceleras bit_exact_supported를 본다
  run   : 고른 사진(--pick set:idx,...)의 최종 출력을 저장 (--ctx bit_exact|quantized)
사용 (hailo10_venv, CUDA_VISIBLE_DEVICES=-1, dfc.lock 안에서): python g6e_bitexact.py run <optimized.har> <dev_zoo224> <out.npz> --pick cifar_A:0,pets_dev:0 --ctx bit_exact
"""
import argparse
import json
import time

import numpy as np
from hailo_sdk_client import ClientRunner, InferenceContext

CTX = {"bit_exact": InferenceContext.SDK_BIT_EXACT, "quantized": InferenceContext.SDK_QUANTIZED}


def load_pick(data, pick):
    xs, tags = [], []
    for p in pick.split(","):
        d, i = p.split(":")
        xs.append(np.load(f"{data}/{d}_pix.npy", mmap_mode="r")[int(i)].astype(np.float32) / 127.5 - 1.0)
        tags.append(p)
    return np.stack(xs), tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "run"])
    ap.add_argument("har"); ap.add_argument("data", nargs="?"); ap.add_argument("out", nargs="?")
    ap.add_argument("--pick", default=""); ap.add_argument("--ctx", default="bit_exact", choices=list(CTX))
    a = ap.parse_args()
    r = ClientRunner(har=a.har)
    if a.cmd == "probe":
        m = r._sdk_backend.build_acceleras_model(InferenceContext.SDK_BIT_EXACT)
        bad = {n: type(l).__name__ for n, l in m.layers.items() if not bool(l.bit_exact_supported)}
        kinds = sorted({type(l).__name__ for l in m.layers.values()})
        print(json.dumps({"har": a.har, "n_layers": len(m.layers), "layer_kinds": kinds, "not_supported": bad}, ensure_ascii=False))
        return
    x, tags = load_pick(a.data, a.pick)
    res, sec = {}, []
    with r.infer_context(CTX[a.ctx]) as ctx:
        for i in range(len(x)):
            t0 = time.time()
            y = r.infer(ctx, x[i:i + 1], batch_size=1)
            sec.append(time.time() - t0)
            res[tags[i]] = np.asarray(y, np.float32).reshape(-1)
            print(f"{tags[i]} {sec[-1]:.1f}s", flush=True)
    np.savez_compressed(a.out, tags=np.array(tags), emb=np.stack([res[t] for t in tags]), sec=np.array(sec))
    print(json.dumps({"har": a.har, "ctx": a.ctx, "n": len(tags), "sec_each": [round(s, 1) for s in sec]}))


if __name__ == "__main__":
    main()
