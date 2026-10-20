"""
s2_hailo_chip.py — Hailo-10H 실칩에서 SigLIP2 이미지 HEF 실행: dev 묶음 임베딩 저장(eval) / 배치 1 지연(bench) (G3Q-H, 10/18)
==================================================================================================================
AGX Orin + Hailo-10H(HailoRT 5.3.0)에서 돈다. numpy + hailo_platform만 쓴다 (저장소·데이터셋 불필요).
연구실 측정 코드(~/ssd/junyong/unified_batch/batch_bench.py)와 같은 API: VDevice → create_infer_model → configure → run_async + wait.

[eval]  --data dev200 (portable_eval.py와 같은 <set>.npz: x N×3×224×224 float32, SigLIP2 전처리·정규화 끝난 값)
  HEF 입력은 NHWC다 (s2_dfc_screen.py·s2_dfc_calib.py가 에뮬레이터·보정에 넣은 것과 같은 정규화된 float32 NHWC).
  입력·출력 형식을 FLOAT32로 두어 양자화·역양자화를 HailoRT가 한다 (HEF의 quant_info 그대로).
  출력: --out/emb_<이름>.npz (<set>_emb: N×D float32, 정규화 전), 데스크탑에서 top-1·cos를 계산한다.
[bench] 배치 1 지연. 입력은 dev 사진 1장을 HEF quant_info로 양자화한 UINT8 (batch_bench.py와 같은 UINT8 입력), 출력은 HEF 원래 형식(UINT16).
  모델마다: 워밍업 --warmup회 → --n회 각각 run_async+wait 시간(perf_counter). --rounds번 모델을 번갈아 잰다.
  hailortcli monitor는 켜지 않는다 (팀 규칙: 지연 꼬리 오염). 같은 프로세스에서 VDevice를 모델마다 새로 만들고 닫는다.
[사용]
  OPENBLAS_NUM_THREADS=1 python3 s2_hailo_chip.py eval  --data dev200 --out out_chip --models D=a.hef rothad=b.hef
  OPENBLAS_NUM_THREADS=1 python3 s2_hailo_chip.py bench --data dev200 --out out_chip --models D=a.hef rothad=b.hef --rounds 3
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
    def __init__(self, hef, float_io):
        import hailo_platform as hpf
        self.vdevice = hpf.VDevice()
        self.im = self.vdevice.create_infer_model(str(hef))
        self.im.set_batch_size(1)
        # float_io=False: 입력 UINT8(HEF 입력 그대로), 출력은 HEF 원래 형식 (SigLIP2 D256L1은 마지막 ew_add가 16비트라 UINT16 — UINT8로 두면 configure가 INVALID_ARGUMENT)
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
        self.b = self.cfg.create_bindings()
        self.ob = np.empty(self.out_shape, dtype=self.out_dtype)
        self.b.output().set_buffer(self.ob)

    def run(self, x):
        # 입력 버퍼를 self에 붙잡아 둔다: 연속 배열이 아니면 ascontiguousarray가 임시 복사본을 만들고, 참조가 없으면
        # 비동기 실행 중에 해제되어 장치가 멈춘다 (10/18 bench에서 HAILO_TIMEOUT으로 발견)
        self.ib = np.ascontiguousarray(x, dtype=self.dtype)
        self.b.input().set_buffer(self.ib)
        self.cfg.wait_for_async_ready(10000, 1)
        self.cfg.run_async([self.b]).wait(10000)
        return self.ob

    def close(self):
        # batch_bench.py와 같은 순서: configured.shutdown() → bindings·configured·infer_model 해제 → VDevice release.
        # shutdown 없이 마지막 run 직후 바로 release하면 HailoRT 5.3.0(Hailo-10H, hrpc)에서 VDevice 해제가 HAILO_TIMEOUT으로 멈췄다 (10/18)
        time.sleep(0.2)               # 마지막 run 직후 바로 shutdown하면 HAILO_TIMEOUT (측정 구간 밖)
        try:
            self.cfg.shutdown()
        except Exception:
            pass
        for a in ("b", "ib", "cfg", "im"):
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
    if a.x_npy:      # (기본 꺼짐) dev 캐시 <set>_x.npy를 그대로 (dev 전체, 10/18)
        data = {d: {"x": np.load(Path(a.data) / f"{d}_x.npy", mmap_mode="r")} for d in SETS if (Path(a.data) / f"{d}_x.npy").exists()}
    else:
        data = {d: np.load(Path(a.data) / f"{d}.npz") for d in SETS if (Path(a.data) / f"{d}.npz").exists()}
    for name, hef in models.items():
        t0 = time.time()
        r = Runner(hef, float_io=True)
        assert r.in_shape == (224, 224, 3), r.in_shape
        embs = {}
        for d, z in data.items():
            x = z["x"]
            assert x.dtype == np.float32 and x.shape[1:] == (3, 224, 224)
            e = np.stack([r.run(np.asarray(x[i]).transpose(1, 2, 0)).reshape(-1).copy() for i in range(len(x))]).astype(np.float32)
            embs[f"{d}_emb"] = e
            print(name, d, e.shape, f"{time.time() - t0:.0f}s", flush=True)
        res["models"][name] = {"hef": Path(hef).name, "hef_sha256": sha256(hef), "in_shape": r.in_shape, "out_shape": r.out_shape,
                               "quant_in": r.qin, "quant_out": r.qout, "sec": round(time.time() - t0)}
        r.close()
        np.savez_compressed(out / f"emb_{name}.npz", **embs)
    (out / "eval_info.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def cmd_bench(a, out, models):
    res = info(a)
    res.update({"warmup": a.warmup, "n": a.n, "rounds": a.rounds, "runs": []})
    x0 = np.load(Path(a.data) / "pets_dev.npz")["x"][0].transpose(1, 2, 0)
    for rd in range(1, a.rounds + 1):
        for name, hef in models.items():
            r = Runner(hef, float_io=False)
            s, zp = r.qin
            xq = np.clip(np.round(x0 / s + zp), 0, 255).astype(np.uint8)
            for _ in range(a.warmup):
                r.run(xq)
            ts = []
            for _ in range(a.n):
                t0 = time.perf_counter()
                r.run(xq)
                ts.append((time.perf_counter() - t0) * 1000)
            r.close()
            ts = np.array(ts)
            row = {"round": rd, "model": name, "hef_sha256": sha256(hef)[:16], "mean_ms": round(float(ts.mean()), 3),
                   "median_ms": round(float(np.median(ts)), 3), "p90_ms": round(float(np.percentile(ts, 90)), 3),
                   "min_ms": round(float(ts.min()), 3), "max_ms": round(float(ts.max()), 3), "fps_from_mean": round(1000 / float(ts.mean()), 2)}
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
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--x-npy", action="store_true", help="(기본 꺼짐) eval 입력을 --data/<set>_x.npy에서 (dev 캐시 형식)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models = dict(s.split("=", 1) for s in a.models)
    (cmd_eval if a.cmd == "eval" else cmd_bench)(a, out, models)


if __name__ == "__main__":
    main()
