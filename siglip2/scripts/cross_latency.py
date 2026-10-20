"""
cross_latency.py — 제출 후보 3개(RN50·SigLIP2 B/16·B/32)를 한 기기에서 같은 실행으로 재는 독립 스크립트 (G8, 10/20)
==========================================================================================================
[왜] 후보끼리 지연을 비교하려면 같은 기기·같은 전원 모드·같은 실행에서 번갈아 재야 한다 (D-010, PRD 5절).
     저장소·데이터셋 없이 numpy + onnxruntime + psutil만으로 돈다.
[측정 조건 (rn50/bench_speed.py와 같음)]
  - ORT CPU, ORT_ENABLE_ALL(정수 커널), intra-op 스레드 고정, inter-op 1, ORT_SEQUENTIAL, 배치 1
  - 모델마다 워밍업 20장 뒤, 모든 모델을 한 바퀴씩 200장 도는 것을 3번 반복 (번갈아). 이것을 스레드 1·2·4로, 세션 3번
  - 입력: 모델 입력 크기에 맞춘 고정 난수(시드 0, 전처리 뒤 범위 −1~1). 속도는 사진 내용과 무관하다 (rn50 bench 주석과 같음)
  - 기록: 세션마다 장당 평균 ms, 개별 호출 p50·p95 → 3세션 중앙값과 FP32 대비 비율 (같은 모델 계열의 fp32와)
  - peak 메모리: --mem 이면 모델마다 새 프로세스에서 세션 생성 + 추론 50회 중 최대 RSS(MiB)
  - 기기 정보: CPU 이름, 논리 코어, ORT 버전, Windows 전원 구성표(powercfg), 충전 여부·배터리, 세션 전 CPU 부하
[사용]  (전원 모드 = 고성능, 충전기 연결, 다른 프로그램 끄기)
  python cross_latency.py --out out_<기기> --mem \
     --models rn50_fp32=<RN50 FP32 onnx> rn50_b2=<RN50 B2 onnx> \
              b16_fp32=siglip2_image_fp32.onnx b16_m2rot=<B16 M2rot_ort onnx> \
              b32_fp32=<B32 FP32 onnx> b32_m2rot=<B32 후보 onnx>
  이름은 "<계열>_<종류>" 형식. 같은 계열의 "_fp32"를 기준으로 비율을 낸다.
"""
import argparse
import json
import os
import platform
import statistics as st
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort


def session(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def make_input(sess, n):
    i = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in i.shape]
    assert len(shape) == 4 and shape[1] == 3 and shape[2] == shape[3], (i.name, i.shape)
    x = np.random.default_rng(0).uniform(-1, 1, size=(n, 3, shape[2], shape[3])).astype(np.float32)
    return i.name, x


def device_info():
    info = {"date": datetime.now().isoformat(timespec="seconds"), "platform": platform.platform(), "python": platform.python_version(),
            "processor": platform.processor(), "ort": ort.__version__, "numpy": np.__version__, "logical_cores": os.cpu_count()}
    if sys.platform == "win32":
        try:
            info["cpu_name"] = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                                              capture_output=True, text=True).stdout.strip()
            info["power_scheme"] = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, encoding="mbcs").stdout.strip()
            info["battery"] = subprocess.run(["powershell", "-NoProfile", "-Command",
                                              "$b=Get-CimInstance Win32_Battery; if($b){'plugged={0} batt={1}%' -f ($b.BatteryStatus -eq 2),$b.EstimatedChargeRemaining}"],
                                             capture_output=True, text=True).stdout.strip()
        except Exception as e:                 # 기기 정보는 기록용 — 실패해도 측정은 한다
            info["info_error"] = repr(e)
    return info


def cpu_load(sec=5):
    try:
        import psutil
        return psutil.cpu_percent(interval=sec)
    except Exception:
        return None


def run_session(models, threads, n, repeats, warmup):
    sess = {k: session(p, threads) for k, p in models.items()}
    inp = {k: make_input(s, n) for k, s in sess.items()}
    for k, s in sess.items():
        name, x = inp[k]
        for i in range(warmup):
            s.run(None, {name: x[i % n:i % n + 1]})
    per_img = {k: [] for k in models}
    lat = {k: [] for k in models}
    for r in range(repeats):
        for k, s in sess.items():                 # 번갈아
            name, x = inp[k]
            t0 = time.perf_counter()
            for i in range(n):
                t1 = time.perf_counter()
                s.run(None, {name: x[i:i + 1]})
                lat[k].append((time.perf_counter() - t1) * 1000)
            per_img[k].append((time.perf_counter() - t0) * 1000 / n)
    return {k: {"ms_per_img_mean": round(st.mean(per_img[k]), 3), "repeats_ms": [round(v, 3) for v in per_img[k]],
                "lat_p50": round(float(np.percentile(lat[k], 50)), 3), "lat_p95": round(float(np.percentile(lat[k], 95)), 3)} for k in models}


def mem_probe(path, threads, iters):
    import psutil
    p = psutil.Process()
    before = p.memory_info().rss
    s = session(path, threads)
    name, x = make_input(s, 1)
    after_load = p.memory_info().rss
    peak = after_load
    for _ in range(iters):
        s.run(None, {name: x})
        peak = max(peak, p.memory_info().rss)
    mi = 2 ** 20
    # Windows는 peak_wset으로 프로세스 최대 작업 집합도 함께 남긴다
    pw = getattr(p.memory_info(), "peak_wset", None)
    return {"rss_before_MiB": round(before / mi, 1), "after_load_MiB": round(after_load / mi, 1),
            "peak_sampled_MiB": round(peak / mi, 1), "peak_wset_MiB": round(pw / mi, 1) if pw else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="이름=onnx 경로 (이름은 <계열>_fp32 / <계열>_<후보>)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--mem", action="store_true", help="모델마다 새 프로세스로 peak 메모리 (4스레드, 추론 50회)")
    ap.add_argument("--_probe", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._probe:                                   # 메모리 측정용 자식 프로세스
        print(json.dumps(mem_probe(a._probe, 4, 50)))
        return
    models = {s.split("=", 1)[0]: Path(s.split("=", 1)[1]) for s in a.models}
    for k, p in models.items():
        assert p.exists(), f"{k}: {p} 없음"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"device": device_info(), "cond": {k: v for k, v in vars(a).items() if k != "_probe"},
           "files": {k: {"path": str(p), "bytes": p.stat().st_size, "MiB": round(p.stat().st_size / 2 ** 20, 2)} for k, p in models.items()},
           "sessions": []}
    print(json.dumps(res["device"], ensure_ascii=False))
    for s in range(a.sessions):
        for t in a.threads:
            load = cpu_load()
            r = run_session(models, t, a.n, a.repeats, a.warmup)
            res["sessions"].append({"session": s + 1, "threads": t, "cpu_load_before_pct": load, "results": r})
            print(f"session {s + 1} t={t} load={load}% " + "  ".join(f"{k}={v['ms_per_img_mean']:.2f}" for k, v in r.items()), flush=True)
            json.dump(res, open(out / "cross_latency.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    summ = {}
    for t in a.threads:
        rows = [x for x in res["sessions"] if x["threads"] == t]
        for k in models:
            med = st.median([x["results"][k]["ms_per_img_mean"] for x in rows])
            summ.setdefault(k, {})[t] = {"median_ms": round(med, 2), "fps_b1": round(1000 / med, 2),
                                         "p50_ms": round(st.median([x["results"][k]["lat_p50"] for x in rows]), 2),
                                         "p95_ms": round(st.median([x["results"][k]["lat_p95"] for x in rows]), 2)}
    for k in models:
        base = k.rsplit("_", 1)[0] + "_fp32"
        if base in summ and base != k:
            for t in a.threads:
                summ[k][t]["vs_fp32_pct"] = round(100 * (summ[k][t]["median_ms"] / summ[base][t]["median_ms"] - 1), 1)
    res["summary"] = summ
    if a.mem:
        res["peak_mem"] = {}
        for k, p in models.items():
            o = subprocess.run([sys.executable, __file__, "--models", "x=y", "--out", str(out), "--_probe", str(p)], capture_output=True, text=True)
            res["peak_mem"][k] = json.loads(o.stdout.strip().splitlines()[-1]) if o.returncode == 0 else {"error": o.stderr[-300:]}
    json.dump(res, open(out / "cross_latency.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({"summary": summ, "peak_mem": res.get("peak_mem")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
