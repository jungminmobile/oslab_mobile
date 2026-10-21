"""g6e_check200.py — dev 6×200 판정: 실칩 임베딩 vs 같은 optimized HAR의 에뮬레이터 (README 2단계 판정 규칙)
정상 = 데이터셋 6개 모두 |cos(실칩, FP32) − cos(에뮬레이터, FP32)| ≤ 0.005  (cos는 데이터셋 평균, FP32 = G6 ref, 앞 200장)
--bitref를 주면 그 실칩 임베딩(앞 200장)과 비트 단위로 같은지도 본다 (B16_rb HAR에 같은 설정을 썼을 때 출력이 그대로인지).
사용 (siglip2 venv): python g6e_check200.py --chip <emb.npz> --emu <emu.npz> --ref ~/siglip2_work/g6/ref --tag <tag> [--bitref <emb.npz>] --out check200.jsonl
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
TOL = 0.005


def l2n(x):
    x = x.astype(np.float64)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", required=True); ap.add_argument("--emu", default=""); ap.add_argument("--ref", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--model", default="b16"); ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--bitref", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R, k = Path(a.ref), a.k
    z = np.load(a.chip); e = np.load(a.emu) if a.emu else None; b = np.load(a.bitref) if a.bitref else None
    rows, ok_all, bit_all = [], True, True
    for d in DATA:
        y, W, ref = np.load(R / f"{a.model}_{d}_y.npy")[:k], np.load(R / f"{a.model}_{d}_clf.npy"), l2n(np.load(R / f"{a.model}_{d}_ref_fp32.npy")[:k])
        c = l2n(z[f"{d}_emb"][:k]); assert c.shape == ref.shape, (d, c.shape, ref.shape)
        r = {"set": d, "n": k, "cos_chip_fp32": round(float((c * ref).sum(1).mean()), 4),
             "top1_chip": round(100 * float((np.argmax(c @ W, 1) == y).mean()), 2), "top1_fp32": round(100 * float((np.argmax(ref @ W, 1) == y).mean()), 2)}
        if e is not None:
            m = l2n(e[f"{d}_emb"][:k]); assert m.shape == ref.shape
            ce = float((m * ref).sum(1).mean())
            r.update(cos_emu_fp32=round(ce, 4), d_cos=round(r["cos_chip_fp32"] - ce, 4), cos_emu_chip=round(float((m * c).sum(1).mean()), 4),
                     top1_emu=round(100 * float((np.argmax(m @ W, 1) == y).mean()), 2), within_tol=abs(float((c * ref).sum(1).mean()) - ce) <= TOL)
            ok_all &= r["within_tol"]
        if b is not None:
            raw, rawb = z[f"{d}_emb"][:k], b[f"{d}_emb"][:k]
            r.update(bit_equal_ref=bool(raw.shape == rawb.shape and np.array_equal(raw, rawb)), maxabs_ref=float(np.abs(raw.astype(np.float64) - rawb).max()))
            bit_all &= r["bit_equal_ref"]
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False))
    out = {"date": datetime.now().isoformat(timespec="seconds"), "what": "g6e_check200", "tag": a.tag, "chip": a.chip, "emu": a.emu, "bitref": a.bitref,
           "normal": ok_all if e is not None else None, "bit_equal_all": bit_all if b is not None else None, "tol": TOL, "rows": rows}
    print(f"== {a.tag}: 정상(6개 모두 |Δcos| ≤ {TOL}) = {out['normal']}, 비트 동일 = {out['bit_equal_all']}")
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
