"""mem_c3_child.py — C0 vs C3 메모리 1회분 (mem_child.py와 같은 정의 + 입력 방식 선택)
사용: python mem_c3_child.py <model> <combo> <threads> <n> <interval_ms> <mode>
  mode=held50 : mem_child.py(E14)와 같음 — dev 앞 n장을 float32로 바꿔 프로세스에 들고 있음 (~39 MiB)
  mode=one    : 1장만 float32로 만들어 n번 반복 추론 (입력이 메모리에 거의 안 섞임)
"""
import sys, json, time, threading
from pathlib import Path
import numpy as np, psutil
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "e12")); import session_opts as SO
import onnxruntime  # noqa: F401
model, combo, threads, n, interval, mode = (sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]),
                                            float(sys.argv[5]) / 1000, sys.argv[6])
P = psutil.Process(); MiB = 2 ** 20
pix = np.load(r"C:\s2art_b32\dev256\pets_dev_pix.npy", mmap_mode="r")
k = n if mode == "held50" else 1
x = np.ascontiguousarray((np.asarray(pix[:k], np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2))
del pix
m = P.memory_info(); base, base_peak = m.wset / MiB, m.peak_wset / MiB
t0 = time.perf_counter(); s = SO.make_session(model, combo, threads); t_create = time.perf_counter() - t0
m = P.memory_info(); create_peak, create_wset, create_priv = m.peak_wset / MiB, m.wset / MiB, m.private / MiB
mx = {"wset": 0.0, "priv": 0.0}; stop = threading.Event()
def sample():
    while True:
        mi = P.memory_info(); mx["wset"] = max(mx["wset"], mi.wset / MiB); mx["priv"] = max(mx["priv"], mi.private / MiB)
        if stop.wait(interval): break
th = threading.Thread(target=sample, daemon=True); th.start()
name = s.get_inputs()[0].name; t0 = time.perf_counter()
for i in range(n):
    j = i if mode == "held50" else 0
    s.run(None, {name: x[j:j + 1]})
t_infer = time.perf_counter() - t0
stop.set(); th.join()
m = P.memory_info()
mx["wset"] = max(mx["wset"], m.wset / MiB); mx["priv"] = max(mx["priv"], m.private / MiB)
r = {"base_MiB": base, "base_peak_MiB": base_peak, "create_peak_MiB": create_peak, "create_wset_MiB": create_wset,
     "create_private_MiB": create_priv, "infer_wset_max_MiB": mx["wset"], "infer_private_max_MiB": mx["priv"],
     "peak_MiB": m.peak_wset / MiB, "t_create_s": t_create, "t_infer_s": t_infer}
print(json.dumps({k: round(v, 3 if k.startswith("t_") else 1) for k, v in r.items()}))
