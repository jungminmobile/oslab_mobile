"""
g6_hailo_chip.py — G6 실칩 (AGX Orin + Hailo-10H, HailoRT 5.3.0): dev 전체 임베딩(eval) / 배치 1·N 지연(bench)
=====================================================================================================
G5b `g5b_hailo_chip.py`(eval)와 G5 `g5_hailo_bench.py`(bench)의 Runner·측정 순서를 그대로 쓴다. 다른 점은 입력뿐이다.
  - 데이터: --data 아래 dev_zoo224/, dev_zoo256/ 의 <set>_pix.npy (Zoo 전처리 uint8 NHWC 픽셀, s2_g6.py data).
    HEF 입력 크기(H)로 dev_zoo<H>를 고른다.
  - HEF 입력 quant_info가 (scale 1, zp 0)이면 정규화가 HEF 안 → 픽셀 그대로 (eval은 float32 픽셀, HailoRT가 양자화).
    아니면 정규화가 밖 → x/127.5 − 1 (eval float32, bench는 quant_info로 UINT8). 어느 쪽을 썼는지 결과에 남긴다.
[eval]  모델마다 VDevice → 입력·출력 FLOAT32 → dev 전체 → out/emb_<이름>.npz (<set>_emb, 정규화 전)
[bench] 모델마다 VDevice → set_batch_size(B), 입력 UINT8, 출력 HEF 원래 형식 → 워밍업 → n회 (B개 run_async 후 모두 wait, perf_counter)
        입력은 pets_dev 1장. hailortcli monitor 안 켬. 모델 사이 2초.
[사용]  OPENBLAS_NUM_THREADS=1 python3 g6_hailo_chip.py eval  --data .. --out out/eval_X --models X=x.hef
        OPENBLAS_NUM_THREADS=1 python3 g6_hailo_chip.py bench --data .. --out out/bench_b1_s1 --batch 1 --models A=a.hef B=b.hef
"""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

SETS = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Runner:
    """g5b_hailo_chip.Runner(float_io) + g5_hailo_bench.Runner(batch)를 합친 것 (같은 API 호출 순서)."""

    def __init__(self, hef, float_io, batch=1):
        import hailo_platform as hpf
        self.batch = batch
        self.vdevice = hpf.VDevice()
        self.im = self.vdevice.create_infer_model(str(hef))
        self.im.set_batch_size(batch)
        if float_io:
            self.im.input().set_format_type(hpf.FormatType.FLOAT32)
            self.im.output().set_format_type(hpf.FormatType.FLOAT32)
            self.dtype = self.out_dtype = np.float32
        else:
            self.im.input().set_format_type(hpf.FormatType.UINT8)
            self.dtype = np.uint8
            self.out_dtype = {hpf.FormatType.UINT8: np.uint8, hpf.FormatType.UINT16: np.uint16}[self.im.output().format.type]
        self.cfg = self.im.configure()
        self.in_shape = tuple(self.im.input().shape)
        self.out_shape = tuple(self.im.output().shape)
        qi, qo = self.im.input().quant_infos[0], self.im.output().quant_infos[0]
        self.qin = (float(qi.qp_scale), float(qi.qp_zp))
        self.qout = (float(qo.qp_scale), float(qo.qp_zp))
        self.bs, self.obs = [], []
        for _ in range(batch):
            b = self.cfg.create_bindings()
            ob = np.empty(self.out_shape, dtype=self.out_dtype)
            b.output().set_buffer(ob)
            self.bs.append(b)
            self.obs.append(ob)
        self.timeout = max(10000, batch * 300)

    def pixel_domain(self):
        return self.qin == (1.0, 0.0)

    def set_input(self, x):
        self.ib = [np.ascontiguousarray(np.array(x, dtype=self.dtype)) for _ in range(self.batch)]   # 비동기 중 해제 방지

    def run(self):
        jobs = []
        for b, x in zip(self.bs, self.ib):
            b.input().set_buffer(x)
            self.cfg.wait_for_async_ready(self.timeout, 1)
            jobs.append(self.cfg.run_async([b]))
        for j in jobs:
            j.wait(self.timeout)
        return self.obs[0]

    def close(self):
        time.sleep(0.2)
        try:
            self.cfg.shutdown()
        except Exception:
            pass
        for a in ("bs", "obs", "ib", "cfg", "im"):
            try:
                delattr(self, a)
            except Exception:
                pass
        try:
            self.vdevice.release()
        except Exception:
            pass
        del self.vdevice


def info(a):
    v = subprocess.run(["hailortcli", "--version"], capture_output=True, text=True).stdout.strip()
    return {"date": datetime.now().isoformat(timespec="seconds"), "machine": platform.machine(), "platform": platform.platform(),
            "python": platform.python_version(), "hailortcli": v, "cmd": a.cmd}


def cmd_eval(a, out, models):
    res = info(a)
    res["models"] = {}
    for name, hef in models.items():
        t0 = time.time()
        r = Runner(hef, float_io=True)
        H = r.in_shape[0]
        assert r.in_shape == (H, H, 3) and H in (224, 256), r.in_shape
        dd = Path(a.data) / (a.subdir or f"dev_zoo{H}")      # --subdir (B32 HEF test, 기본 꺼짐 = dev_zoo<H>)
        pix = r.pixel_domain()
        embs = {}
        for d in (a.sets.split(",") if a.sets else SETS):     # --sets (G6d, 기본 전부): 데이터셋을 나눠 사이에 냉각
            x = np.load(dd / f"{d}_pix.npy", mmap_mode="r")
            if a.k or a.start:                                # --start (B32 HEF test, 기본 0): 앞 start장 건너뜀 (덩어리 eval)
                x = x[a.start: a.start + a.k if a.k else None]
            assert x.dtype == np.uint8 and x.shape[1:] == (H, H, 3), (d, x.shape)
            e = []
            for i in range(len(x)):
                xi = np.asarray(x[i], np.float32)
                r.set_input(xi if pix else xi / 127.5 - 1.0)
                e.append(r.run().reshape(-1).copy())
            embs[f"{d}_emb"] = np.stack(e).astype(np.float32)
            print(name, d, embs[f"{d}_emb"].shape, f"{time.time() - t0:.0f}s", flush=True)
        res["models"][name] = {"input_domain": "pixel_0_255" if pix else "normalized", "hef": Path(hef).name, "hef_sha256": sha256(hef),
                               "in_shape": r.in_shape, "out_shape": r.out_shape, "quant_in": r.qin, "quant_out": r.qout,
                               "data": str(dd), "k": a.k, **({"start": a.start} if a.start else {}), "sec": round(time.time() - t0)}
        r.close()
        np.savez_compressed(out / f"emb_{name}.npz", **embs)
    (out / "eval_info.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_bench(a, out, models):
    res = info(a)
    res.update({"label": a.label, "batch": a.batch, "warmup": a.warmup, "n": a.n, "runs": []})
    for name, hef in models.items():
        r = Runner(hef, float_io=False, batch=a.batch)
        H = r.in_shape[0]
        x0 = np.load(Path(a.data) / f"dev_zoo{H}" / "pets_dev_pix.npy", mmap_mode="r")[0]
        s, zp = r.qin
        if r.pixel_domain():
            dom, xq = "pixel_0_255", np.asarray(x0, np.uint8)
        else:
            dom, xq = "normalized", np.clip(np.round((np.asarray(x0, np.float32) / 127.5 - 1.0) / s + zp), 0, 255).astype(np.uint8)
        r.set_input(xq)
        for _ in range(a.warmup):
            r.run()
        ts = []
        for _ in range(a.n):
            t0 = time.perf_counter()
            r.run()
            ts.append((time.perf_counter() - t0) * 1000)
        r.close()
        ts = np.array(ts)
        row = {"model": name, "hef_sha256": sha256(hef)[:16], "batch": a.batch, "input_domain": dom, "qin": [s, zp],
               "in_shape": list(r.in_shape), "mean_ms": round(float(ts.mean()), 3), "median_ms": round(float(np.median(ts)), 3),
               "p90_ms": round(float(np.percentile(ts, 90)), 3), "min_ms": round(float(ts.min()), 3), "max_ms": round(float(ts.max()), 3),
               "img_per_s_from_mean": round(1000 * a.batch / float(ts.mean()), 2)}
        res["runs"].append(row)
        print(json.dumps(row), flush=True)
        time.sleep(2)
    (out / "bench.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eval", "bench"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="이름=HEF 경로")
    ap.add_argument("--k", type=int, default=0, help="eval: 데이터셋마다 앞 k장 (0 = 전부)")
    ap.add_argument("--sets", default="", help="(G6d, 기본 전부) eval: 쉼표로 고른 데이터셋만")
    ap.add_argument("--start", type=int, default=0, help="(B32 HEF test, 기본 0) eval: 데이터셋마다 앞 start장 건너뜀 (--k와 함께 덩어리)")
    ap.add_argument("--subdir", default="", help="(B32 HEF test, 기본 꺼짐) eval: --data 아래 읽을 폴더 (예: test_zoo256). 비우면 dev_zoo<H>")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = dict(s.split("=", 1) for s in a.models)
    (cmd_eval if a.cmd == "eval" else cmd_bench)(a, out, models)


if __name__ == "__main__":
    main()
