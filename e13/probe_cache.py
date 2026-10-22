"""probe_cache.py — E13-1 0단계 사전 확인: C:\\s2art\\cache 파일 모양·값 범위 (읽기만)."""
import sys, numpy as np
from pathlib import Path
d = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\s2art\cache")
for f in sorted(d.glob("*.npy")):
    a = np.load(f, mmap_mode="r")
    s = f"{f.name:28s} {str(a.shape):22s} {a.dtype}"
    if f.name.endswith("_x.npy"):
        s += f" min {float(a[:4].min()):.3f} max {float(a[:4].max()):.3f}"
    print(s)
for f in sorted(d.glob("*.txt")):
    print(f.name, len(f.read_text().split()))
