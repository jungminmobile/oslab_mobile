#!/usr/bin/env python3
"""
peak_mem.py — 모델 하나를 새 프로세스에서 열고 배치 1로 20회 돌린 뒤 메모리(RSS)를 잰다 (리눅스 /proc 기준)
  세션 생성 직후 RSS, 추론 후 RSS, 프로세스 최대 RSS(VmHWM)를 MiB로 출력. 모델마다 따로 실행해야 한다.
사용: for m in models/*.onnx; do python peak_mem.py "$m" --threads 4; done
"""
import argparse, json, platform, sys
import numpy as np
import onnxruntime as ort


def status_mib(key):
    for line in open("/proc/self/status"):
        if line.startswith(key + ":"):
            return round(int(line.split()[1]) / 1024, 1)   # kB → MiB


ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--runs", type=int, default=20)
a = ap.parse_args()

base = status_mib("VmRSS")
so = ort.SessionOptions()
so.intra_op_num_threads, so.inter_op_num_threads = a.threads, 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
s = ort.InferenceSession(a.model, so, providers=["CPUExecutionProvider"])
after_session = status_mib("VmRSS")
hwm_session = status_mib("VmHWM")
x = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
for _ in range(a.runs):
    s.run(None, {"image": x})
print(json.dumps({
    "model": a.model.split("/")[-1], "machine": platform.machine(), "ort": ort.__version__, "threads": a.threads,
    "rss_python_base_MiB": base, "rss_after_session_MiB": after_session, "peak_during_session_create_MiB": hwm_session,
    "rss_after_infer_MiB": status_mib("VmRSS"), "peak_total_MiB": status_mib("VmHWM"),
}, ensure_ascii=False))
