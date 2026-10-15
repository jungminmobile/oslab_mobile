"""
te_bench.py — 텍스트 인코더 지연·메모리 측정 (스모크 5단계). onnxruntime·numpy만 있으면 돈다 (Jetson AGX Orin용)
=========================================================================================================

[측정 조건] rn50/bench_speed.py와 같은 절차를 텍스트 입력으로 (그 스크립트는 이미지 입력 고정이라 직접 못 씀):
  - ORT CPU, ORT_ENABLE_ALL, intra-op 스레드 고정, inter-op 1 (rn50/common.session 그대로 import)
  - 배치 1 = 질의 문장 1개 (Pets 클래스 프롬프트 토큰을 차례로 돌림, tokens_pets.npy)
  - 모델마다 예열 20회 → 모든 모델을 번갈아 200회씩 × 3바퀴, 스레드 수마다 반복
  - 기록: 회당 평균(ms)·바퀴 간 표준편차·p50·p95, 파일 크기, 세션 생성 후 RSS(모델마다 새 프로세스), CPU 모델·플래그·ORT 버전
  다른 무거운 작업과 동시에 돌리지 말 것.

[사용]  (토큰·ONNX 파일이 있는 폴더를 --dir로)
  python textenc/te_bench.py --dir ~/ssd/junyong/textenc --threads 1 2 4 --out results/textenc/smoke/speed_agx \
      --models fp32=siglip2_text_fp32.onnx A=siglip2_text_int8_dyn.onnx ...
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rn50"))
import common as C   # noqa: E402  rn50/common.py — numpy·json만 import (ORT 세션 규칙)


def cpu_env():
    info = {"machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version()}
    try:
        txt = Path("/proc/cpuinfo").read_text()
        for key in ("model name", "Model", "CPU part"):
            line = next((l for l in txt.splitlines() if l.startswith(key)), None)
            if line:
                info[key] = line.split(":", 1)[1].strip()
        flags = set(next((l for l in txt.splitlines() if l.startswith(("flags", "Features"))), ":").split(":", 1)[1].split())
        info["flags"] = {f: f in flags for f in ("avx2", "avx512f", "avx512_vnni", "avx_vnni", "asimddp", "i8mm", "sve")}
    except OSError:
        try:
            import cpuinfo
            ci = cpuinfo.get_cpu_info()
            info["model name"] = ci.get("brand_raw")
            fl = {f.replace("_", "") for f in ci.get("flags", [])}
            info["flags"] = {f: f.replace("_", "") in fl for f in ("avx2", "avx512f", "avx512_vnni", "avx_vnni")}
        except ImportError:
            pass
    return info


def git_commit():
    try:
        r = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": r("rev-parse", "--short", "HEAD"),
                "dirty": bool(r("status", "--porcelain", "--untracked-files=all", "--", "textenc", "rn50", "siglip2"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def rss_child(path, threads, tok):
    """새 프로세스에서 세션 생성 + 1회 실행 후 RSS(MiB). 모델끼리 메모리가 섞이지 않게."""
    code = ("import sys,numpy as np;sys.path.insert(0,%r);import common as C;import psutil;"
            "s=C.session(%r,%d);s.run(None,{'tokens':np.load(%r)[:1]});"
            "print(psutil.Process().memory_info().rss/2**20)") % (str(REPO / "rn50"), str(path), threads, str(tok))
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
        return round(float(out.stdout.strip().splitlines()[-1]), 1)
    except (ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="ONNX·토큰 파일 폴더")
    ap.add_argument("--models", nargs="+", required=True, help="이름=파일")
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--tokens", default="tokens_pets.npy")
    ap.add_argument("--no-rss", action="store_true")
    ap.add_argument("--allow-nonstandard", action="store_true", help="디버그 전용. 결과에 standard=false로 남는다")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import onnxruntime as ort
    std = (a.warmup, a.n, a.repeats) == (20, 200, 3)
    assert std or a.allow_nonstandard, "PRD 5절 조건은 예열 20 / 200회 / 3바퀴 (디버그용으로만 --allow-nonstandard)"
    d = Path(a.dir).expanduser()
    models = {k: d / v for k, v in (kv.split("=", 1) for kv in a.models)}
    tok = np.load(d / a.tokens)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"date": datetime.now().isoformat(timespec="seconds"), **git_commit(), "onnxruntime": ort.__version__,
           "env": cpu_env(), "cond": vars(a), "standard": std, "results": {}}
    print(json.dumps(res["env"]), "ORT", ort.__version__)
    for th in a.threads:
        sess = {k: C.session(p, th) for k, p in models.items()}
        for s in sess.values():
            for i in range(a.warmup):
                s.run(None, {"tokens": tok[i % len(tok):i % len(tok) + 1]})
        per, lat = {k: [] for k in models}, {k: [] for k in models}
        for r in range(a.repeats):
            for k, s in sess.items():
                t0 = time.perf_counter()
                for i in range(a.n):
                    j = i % len(tok)
                    t1 = time.perf_counter()
                    s.run(None, {"tokens": tok[j:j + 1]})
                    lat[k].append((time.perf_counter() - t1) * 1000)
                per[k].append((time.perf_counter() - t0) * 1000 / a.n)
                print(f"threads {th} repeat {r + 1}/{a.repeats} {k:10s} {per[k][-1]:8.2f} ms", flush=True)
        del sess
        for k, p in models.items():
            L = np.array(lat[k])
            row = {"file": p.name, "MiB": round(p.stat().st_size / 2**20, 1),
                   "ms_mean": round(statistics.mean(per[k]), 2), "ms_std": round(statistics.stdev(per[k]), 2),
                   "lat_p50": round(float(np.percentile(L, 50)), 2), "lat_p95": round(float(np.percentile(L, 95)), 2)}
            res["results"].setdefault(k, {})[f"t{th}"] = row
    if not a.no_rss:
        for k, p in models.items():
            res["results"][k]["rss_MiB_t1"] = rss_child(p, 1, d / a.tokens)
    print(f"\n{'model':10s} {'MiB':>7s} " + " ".join(f"{'t' + str(t) + ' ms':>14s}" for t in a.threads) + f" {'RSS':>7s}")
    for k in models:
        r = res["results"][k]
        print(f"{k:10s} {r['t' + str(a.threads[0])]['MiB']:7.1f} " +
              " ".join(f"{r['t' + str(t)]['ms_mean']:8.2f}±{r['t' + str(t)]['ms_std']:<5.2f}" for t in a.threads) +
              f" {r.get('rss_MiB_t1')}")
    fn = out / f"speed_{datetime.now():%Y%m%d_%H%M%S}.json"
    fn.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("saved", fn)


if __name__ == "__main__":
    main()
