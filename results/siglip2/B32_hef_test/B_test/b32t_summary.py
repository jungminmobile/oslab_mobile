"""b32t_summary.py — README B-5: test 7종 GF·Z 실칩 임베딩 + HF FP32 기준 → top-1(argmax)·Δ·쌍 비교 95% 구간·cos.
쌍: GF − Z, GF − FP32, Z − FP32. 고화질 5종 평균(주 지표)은 s2_paired_ci.boot_hr (층화 짝 부트스트랩, 시드 0, 1,000회 — 준혁 s2_test_summary와 같음).
입력: <out>/ref/<set>_{ref,clf,y}.npy, <chip>/eval_<H>/part_<set>/emb_b32_<H>.npz (<set>_emb). 출력: <log>/test_summary.json, preds_<set>.npz(사진별 예측·cos).
[사용] python b32t_summary.py --out ~/siglip2_work/b32t --chip ~/siglip2_work/b32t/chip_out --log <B_test 폴더>"""
import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "siglip2"))
from s2_paired_ci import boot_hr       # noqa: E402

HR = ["pets", "flowers", "cars", "food", "caltech"]
OTHER = ["cifar_test", "imagenetv2"]
HEFS = ["GF", "Z"]
PAIRS = [("GF", "Z"), ("GF", "fp32"), ("Z", "fp32")]


def l2n(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--chip", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    out, chip, log = Path(a.out), Path(a.chip), Path(a.log)
    C, per = {}, {}
    for s in HR + OTHER:
        ref = l2n(np.load(out / "ref" / f"{s}_ref.npy").astype(np.float64))
        W = np.load(out / "ref" / f"{s}_clf.npy").astype(np.float64)
        y = np.load(out / "ref" / f"{s}_y.npy")
        c = {"fp32": np.argmax(ref @ W, 1) == y}
        r = {"n": int(len(y)), "fp32_top1": round(100 * float(c["fp32"].mean()), 2)}
        npz = {"y": y, "fp32_pred": np.argmax(ref @ W, 1)}
        for h in HEFS:
            e = l2n(np.load(chip / f"eval_{h}" / f"part_{s}" / f"emb_b32_{h}.npz")[f"{s}_emb"].astype(np.float64))
            assert e.shape == ref.shape, (h, s, e.shape, ref.shape)
            p = np.argmax(e @ W, 1)
            c[h] = p == y
            cos = (e * ref).sum(1)
            r[f"{h}_top1"] = round(100 * float(c[h].mean()), 2)
            r[f"{h}_cos_mean"] = round(float(cos.mean()), 4)
            r[f"{h}_cos_p1"] = round(float(np.percentile(cos, 1)), 4)
            npz[f"{h}_pred"], npz[f"{h}_cos"] = p, cos.astype(np.float32)
        for b, q in PAIRS:
            d = c[b].astype(float) - c[q].astype(float)
            r[f"{b}-{q}"] = {"d": round(100 * float(d.mean()), 2), "ci95": boot_hr({s: d})}
        np.savez_compressed(log / f"preds_{s}.npz", **npz)
        C[s], per[s] = c, r
    hr = {"fp32_top1": round(float(np.mean([per[s]["fp32_top1"] for s in HR])), 2)}
    for h in HEFS:
        hr[f"{h}_top1"] = round(float(np.mean([per[s][f"{h}_top1"] for s in HR])), 2)
        hr[f"{h}_cos_mean"] = round(float(np.mean([per[s][f"{h}_cos_mean"] for s in HR])), 4)
    for b, q in PAIRS:
        diffs = {s: C[s][b].astype(float) - C[s][q].astype(float) for s in HR}
        hr[f"{b}-{q}"] = {"d": round(float(np.mean([100 * v.mean() for v in diffs.values()])), 2), "ci95": boot_hr(diffs)}
    git = lambda *x: subprocess.run(["git", "-C", str(REPO), *x], capture_output=True, text=True).stdout.strip()
    res = {"date": datetime.now().isoformat(timespec="seconds"), "commit": git("rev-parse", "--short", "HEAD"),
           "dirty": bool(git("status", "--porcelain", "--untracked-files=no")), "n_boot": 1000, "seed": 0,
           "top1": "argmax (L2 정규화 임베딩 @ HF 분류기)", "hr": hr, "sets": per}
    (log / "test_summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
