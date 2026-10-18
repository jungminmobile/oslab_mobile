"""
s2_hailo_chip_eval.py — s2_hailo_chip.py eval이 저장한 실칩 임베딩으로 top-1·FP32 대비 cos, 짝 비교, 에뮬레이터와의 일치 (G3Q-H, 10/18)
=================================================================================================================
클래스 임베딩(clf)·FP32 기준(ref_fp32)·정답(y)은 dev 캐시 앞 200장 (s2_dfc_screen.py --k 200과 같은 사진·같은 기준).
[사용] (siglip2/ 폴더)
  RN50_RESULTS=../results/siglip2/G3Q_rotate/hailo "$SIGLIP2_PYTHON" hailo/s2_hailo_chip_eval.py --chip <out_chip 폴더> \
      --models D_D256L1 rothad_D256L1 --vs D_D256L1 --emu <pred_<이름>.npz가 있는 폴더>
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import s2common as S                 # noqa: E402
from s2_paired_ci import boot_hr     # noqa: E402
from scripts import env_info         # noqa: E402

RC = S.RC
DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HR = DATA[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--vs", default="")
    ap.add_argument("--emu", default="", help="에뮬레이터 pred_<이름>.npz 폴더 (있으면 사진별 일치 비교)")
    ap.add_argument("--k", type=int, default=200, help="데이터셋마다 앞 k장 (0이면 전체)")
    ap.add_argument("--out-name", default="chip_eval.json")
    a = ap.parse_args()
    out = S.require_results_dir()
    env = env_info.collect()
    res = {"date": datetime.now().isoformat(timespec="seconds"), "commit": env["git"]["commit"], "dirty": env["git"]["dirty"],
           "chip_dir": a.chip, "models": {}, "vs": {}, "emu": {}}
    P = {}
    for m in a.models:
        z = np.load(Path(a.chip) / f"emb_{m}.npz")
        row, P[m] = {}, {}
        for d in DATA:
            sl = slice(None) if a.k == 0 else slice(a.k)
            y, W, ref = (np.load(RC.CACHE_DIR / f"{d}_{s}.npy")[sl] if s != "clf" else np.load(RC.CACHE_DIR / f"{d}_clf.npy")
                         for s in ("y", "clf", "ref_fp32"))
            e = RC.l2n(z[f"{d}_emb"].astype(np.float32))
            assert e.shape == ref.shape, (m, d, e.shape, ref.shape)
            ref = RC.l2n(ref)
            cos = (e * ref).sum(1)
            pred, fp = np.argmax(e @ W, 1), np.argmax(ref @ W, 1)
            P[m][d] = {"pred": pred, "cos": cos, "y": y}
            row[d] = {"n": int(len(y)), "top1": round(100 * float((pred == y).mean()), 2), "fp32_top1": round(100 * float((fp == y).mean()), 2),
                      "cos_mean": round(float(cos.mean()), 4)}
        row["hr_cos"] = round(float(np.mean([row[d]["cos_mean"] for d in HR])), 4)
        row["hr_dtop1"] = round(float(np.mean([row[d]["top1"] - row[d]["fp32_top1"] for d in HR])), 2)
        res["models"][m] = row
        print(m, "chip", {d: (row[d]["top1"], row[d]["cos_mean"]) for d in DATA}, "hr_cos", row["hr_cos"], "hr_dtop1", row["hr_dtop1"])
        if a.emu and (Path(a.emu) / f"pred_{m}.npz").exists():
            q = np.load(Path(a.emu) / f"pred_{m}.npz")
            agree = {d: float(np.mean(q[f"{d}_pred"] == P[m][d]["pred"])) for d in DATA}
            cd = np.concatenate([np.abs(q[f"{d}_cos"] - P[m][d]["cos"]) for d in HR])
            emu_hr = float(np.mean([q[f"{d}_cos"].mean() for d in HR]))
            diffs = {d: (P[m][d]["pred"] == P[m][d]["y"]).astype(float) - (q[f"{d}_pred"] == q[f"{d}_y"]).astype(float) for d in HR}
            res["emu"][m] = {"pred_agree": {d: round(v, 4) for d, v in agree.items()},
                             "pred_agree_hr": round(float(np.mean([agree[d] for d in HR])), 4),
                             "cos_absdiff_mean_hr": float(cd.mean()), "cos_absdiff_max_hr": float(cd.max()),
                             "emu_hr_cos": round(emu_hr, 4), "chip_minus_emu_hr_cos": round(row["hr_cos"] - emu_hr, 4),
                             "chip_minus_emu_top1": round(float(np.mean([100 * v.mean() for v in diffs.values()])), 2),
                             "chip_minus_emu_top1_ci95": boot_hr(diffs)}
            print("  vs emulator", json.dumps(res["emu"][m], ensure_ascii=False))
    if a.vs:
        for m in a.models:
            if m == a.vs:
                continue
            diffs = {d: (P[m][d]["pred"] == P[m][d]["y"]).astype(float) - (P[a.vs][d]["pred"] == P[a.vs][d]["y"]).astype(float) for d in HR}
            r = {"top1_diff": round(float(np.mean([100 * v.mean() for v in diffs.values()])), 2), "top1_diff_ci95": boot_hr(diffs),
                 "hr_cos_diff": round(res["models"][m]["hr_cos"] - res["models"][a.vs]["hr_cos"], 4),
                 "cos_better_frac": round(float(np.mean(np.concatenate([P[m][d]["cos"] > P[a.vs][d]["cos"] for d in HR]))), 3),
                 "pred_agree_hr": round(float(np.mean([np.mean(P[m][d]["pred"] == P[a.vs][d]["pred"]) for d in HR])), 4),
                 "cifar_A_top1_diff": round(res["models"][m]["cifar_A"]["top1"] - res["models"][a.vs]["cifar_A"]["top1"], 2)}
            res["vs"][f"{m}-{a.vs}"] = r
            print(f"{m} − {a.vs}", json.dumps(r, ensure_ascii=False))
    with open(out / a.out_name, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
