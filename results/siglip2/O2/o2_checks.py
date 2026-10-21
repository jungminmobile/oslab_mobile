"""O2 기록용 확인: (1) 보정 사진 동일성 — calib_images(highres,256,0)(timm 224) vs Zoo 224·256 캐시 픽셀 sha256,
(2) 정확도 실행의 기준 임베딩이 O1 저장 임베딩과 같은지. 사용 (siglip2/):  python ../results/siglip2/O2/o2_checks.py
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(Path.cwd()))
import s2_quantize as Q
D = Path(__file__).resolve().parent
P = Path.home() / "siglip2_work" / "g6pre"
W = Path.home() / "siglip2_work" / "g6"
h = lambda x: hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
a = Q.calib_images("highres", 256, 0).astype(np.float32)
b = np.load(P / "cache_zoo224" / "calib256_s0_x.npy").astype(np.float32)
c = np.load(P / "cache_zoo256" / "calib256_s0_x.npy").astype(np.float32)
r = {"timm224_calib_images": {"shape": list(a.shape), "sha256": h(a)}, "zoo224_calib256_s0": {"shape": list(b.shape), "sha256": h(b)},
     "zoo256_calib256_s0": {"shape": list(c.shape), "sha256": h(c)}, "timm224_eq_zoo224": bool(np.array_equal(a, b)),
     "timm224_vs_zoo224_maxabs": float(np.abs(a - b).max())}
r["emb_vs_o1"] = {}
for x, y in (("o2base", "o1b16base"), ("o2oab", "o1b16oab"), ("o2fp", "o1b16fp")):
    e1, e2 = np.load(W / f"g6b_ort_emb_{x}.npz"), np.load(W / f"g6b_ort_emb_{y}.npz")
    r["emb_vs_o1"][f"{x}=={y}"] = sorted(e1.files) == sorted(e2.files) and all(np.array_equal(e1[k], e2[k]) for k in e1.files)
print(json.dumps(r, indent=1))
(D / "regress" / "checks.json").write_text(json.dumps(r, indent=1))
