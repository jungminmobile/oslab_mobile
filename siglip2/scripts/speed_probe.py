"""측정 전 속도 확인: B/16 FP32 4스레드 중앙값(ms). 고성능 모드면 약 170, 210 넘으면 이상."""
import sys, time, numpy as np, onnxruntime as ort, psutil
p = sys.argv[1]; t = int(sys.argv[2]) if len(sys.argv) > 2 else 4
so = ort.SessionOptions(); so.intra_op_num_threads = t; so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
i = s.get_inputs()[0]; x = np.random.default_rng(0).uniform(-1, 1, (1, 3, 224, 224)).astype(np.float32)
for _ in range(10): s.run(None, {i.name: x})
ts = []
for _ in range(40):
    a = time.perf_counter(); s.run(None, {i.name: x}); ts.append((time.perf_counter() - a) * 1e3)
f = psutil.cpu_freq()
print(f"probe t={t} median={np.median(ts):.1f} ms  p10={np.percentile(ts,10):.1f}  freq={f.current:.0f}MHz")
