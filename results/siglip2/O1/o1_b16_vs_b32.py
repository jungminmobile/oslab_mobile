"""O1 기록용 (판정 아님): 같은 dev 사진에서 B16 − B32 고화질 top-1 쌍 비교 구간 (G6b b16_vs_b32_pairs와 같은 방법).
저장 임베딩 ~/siglip2_work/g6/g6b_ort_emb_o1<이름>.npz + HF 분류기. 사용 (siglip2/):  python ../results/siglip2/O1/o1_b16_vs_b32.py
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.cwd()))
from s2_paired_ci import boot_hr  # noqa: E402
W = Path.home() / "siglip2_work" / "g6"
DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]; HR = DATA[1:]
l2n = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
def ok(name, ref):
    e = np.load(W / f"g6b_ort_emb_o1{name}.npz")
    return {d: (np.argmax(l2n(e[d]) @ np.load(W / "ref" / f"{ref}_{d}_clf.npy"), 1) == np.load(W / "ref" / f"{ref}_{d}_y.npy")).astype(float) for d in DATA}
rows = []
for a, b in (("b16base", "b32base"), ("b16oab", "b32oab"), ("b16fp", "b32fp")):
    A, B = ok(a, "b16"), ok(b, "b32")
    diff = {d: A[d] - B[d] for d in HR}
    rows.append({"pair": f"{a}-{b}", "hr_diff": round(float(np.mean([100 * v.mean() for v in diff.values()])), 2), "ci95": boot_hr(diff),
                 "cifar_A_diff": round(100 * float((A["cifar_A"] - B["cifar_A"]).mean()), 2)})
    print(rows[-1])
with open(Path(__file__).resolve().parent / "b16_vs_b32_pairs.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
