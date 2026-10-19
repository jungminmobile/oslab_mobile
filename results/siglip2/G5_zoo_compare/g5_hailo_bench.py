"""
g5_hailo_bench.py — Hailo-10H 실칩 지연 (배치 1·배치 N), G5 Zoo 비교용 (10/19)
==========================================================================
`siglip2/scripts/s2_hailo_chip.py`(브랜치 siglip2/g3q-rotate·g3rq-m2rot, AGX ~/ssd/junyong/hailo_repro/) bench를 그대로 옮기고
**기본값이 1인 `--batch`만 더했다.** `--batch 1`이면 측정 순서·타이머·입력이 s2_hailo_chip.py bench와 같다.
  - 모델마다: VDevice → create_infer_model → set_batch_size(B) → 입력 UINT8, 출력 HEF 원래 형식 → configure
  - 워밍업 --warmup번 → --n번, 한 번 = B개를 run_async로 모두 넣고 B개 모두 wait (perf_counter). B=1이면 s2_hailo_chip.py와 같다.
  - hailortcli monitor는 켜지 않는다. 모델마다 VDevice를 새로 만들고 닫는다. 모델 사이 2초 쉼.
  - 입력: dev 사진 pets_dev[0] 1장 (SigLIP 정규화 끝난 float32, mean=std=0.5). HEF 입력 quant_info가
      · 정규화된 값 범위(우리 HEF, scale 1/127·zp 128)이면 그대로 양자화 (s2_hailo_chip.py와 같음)
      · 원본 픽셀 범위(Zoo HEF, scale 1·zp 0, 정규화가 HEF 안)이면 픽셀로 되돌려((x·0.5+0.5)·255 반올림) 넣는다
    (입력 값은 지연에 영향이 없어야 하지만, 각 HEF가 기대하는 형식에 맞춰 넣는다. 어떤 쪽을 썼는지 결과에 남긴다.)
[사용] OPENBLAS_NUM_THREADS=1 python3 g5_hailo_bench.py --data dev200 --out out/bench_b1_s1 --batch 1 --models A=a.hef B=b.hef
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


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class Runner:
    def __init__(self, hef, batch):
        import hailo_platform as hpf
        self.batch = batch
        self.vdevice = hpf.VDevice()
        self.im = self.vdevice.create_infer_model(str(hef))
        self.im.set_batch_size(batch)
        self.im.input().set_format_type(hpf.FormatType.UINT8)
        self.out_dtype = {hpf.FormatType.UINT8: np.uint8, hpf.FormatType.UINT16: np.uint16}[self.im.output().format.type]
        self.cfg = self.im.configure()
        self.in_shape = tuple(self.im.input().shape)
        self.out_shape = tuple(self.im.output().shape)
        qi = self.im.input().quant_infos[0]
        self.qin = (float(qi.qp_scale), float(qi.qp_zp))
        self.bs, self.obs = [], []
        for _ in range(batch):            # 항목마다 따로 bindings·출력 버퍼 (비행 중 재사용 방지, batch_bench.py와 같음)
            b = self.cfg.create_bindings()
            ob = np.empty(self.out_shape, dtype=self.out_dtype)
            b.output().set_buffer(ob)
            self.bs.append(b)
            self.obs.append(ob)
        self.timeout = max(10000, batch * 300)

    def set_input(self, xq):
        self.ib = [np.ascontiguousarray(xq.copy()) for _ in range(self.batch)]   # 비동기 실행 중 해제되지 않게 self에 붙잡아 둔다

    def run(self):
        # s2_hailo_chip.py와 같은 순서: set_buffer → wait_for_async_ready → run_async, 모두 넣은 뒤 wait
        jobs = []
        for b, x in zip(self.bs, self.ib):
            b.input().set_buffer(x)
            self.cfg.wait_for_async_ready(self.timeout, 1)
            jobs.append(self.cfg.run_async([b]))
        for j in jobs:
            j.wait(self.timeout)

    def close(self):
        time.sleep(0.2)   # s2_hailo_chip.py와 같음: 마지막 run 직후 바로 shutdown하면 HAILO_TIMEOUT (측정 구간 밖)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="이름=HEF 경로")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = dict(s.split("=", 1) for s in a.models)
    v = subprocess.run(["hailortcli", "--version"], capture_output=True, text=True).stdout.strip()
    res = {"date": datetime.now().isoformat(timespec="seconds"), "machine": platform.machine(), "platform": platform.platform(),
           "python": platform.python_version(), "hailortcli": v, "label": a.label, "batch": a.batch, "warmup": a.warmup, "n": a.n, "runs": []}
    x0 = np.load(Path(a.data) / "pets_dev.npz")["x"][0].transpose(1, 2, 0)   # 224×224×3 float32, 정규화 끝난 값
    for name, hef in models.items():
        r = Runner(hef, a.batch)
        assert r.in_shape == x0.shape, (name, r.in_shape)
        s, zp = r.qin
        if s == 1.0 and zp == 0.0:
            dom = "pixel_0_255"
            xq = np.clip(np.round((x0 * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
        else:
            dom = "normalized"
            xq = np.clip(np.round(x0 / s + zp), 0, 255).astype(np.uint8)
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
               "in_shape": list(x0.shape), "mean_ms": round(float(ts.mean()), 3), "median_ms": round(float(np.median(ts)), 3),
               "p90_ms": round(float(np.percentile(ts, 90)), 3), "min_ms": round(float(ts.min()), 3), "max_ms": round(float(ts.max()), 3),
               "img_per_s_from_mean": round(1000 * a.batch / float(ts.mean()), 2)}
        res["runs"].append(row)
        print(json.dumps(row), flush=True)
        time.sleep(2)
    (out / "bench.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
