"""
g8_reanalysis.py — G8 B1: 모델 실행 없이 이미 있는 사진별 예측으로 다시 계산 (2026-10-22, 독립 검토 REVIEW_20261022 4절 L1~L3)
====================================================================================================================
  L1 test: M2rot_ort − R0 고화질 짝 부트스트랩 (보고서 235행 구간 [+0.32, +0.68]의 raw 근거), 데이터셋별 구간
  L2 dev : M2rot_ort·O-AB·R0의 FP32 대비 고화질 Δ — 부트스트랩 1,000회 시드 0~9의 구간 끝 범위, 10,000회 구간, 데이터셋별 구간, O-AB−M2rot 짝
  L3     : 6×200 스크리닝(정수 커널) Δ·cos와 dev 전체 Δ의 순위 상관 (D-024 약속)
입력 (읽기만): 결과 폴더의 test_<set>.npz (s2_test_eval), pred_<모델>.npz (s2_paired_ci), screen.jsonl, paired_ci.jsonl, dev 캐시 _y/_clf/_ref_fp32
출력: --out 폴더에 reanalysis.json
사용: "$SIGLIP2_PYTHON" scripts/g8_reanalysis.py --test-r0 <G6_test> --test-m2 <G6_m2rot_ort/test> --dev-dirs <dev_full 폴더들> --results-root <results/siglip2> --out <폴더>
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from s2_paired_ci import boot_hr, HR   # noqa: E402  (s2common import → 캐시 경로)
import s2common as S                   # noqa: E402

TEST_HR = ["pets", "flowers", "cars", "food", "caltech"]


def ci(vals_by_set, n, seed):
    return boot_hr(vals_by_set, n_boot=n, seed=seed)


def per_set_ci(v, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    b = [100 * v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [round(float(np.percentile(b, 2.5)), 2), round(float(np.percentile(b, 97.5)), 2)]


def summarize(diffs, label):
    hr = float(np.mean([100 * v.mean() for v in diffs.values()]))
    seeds = [ci(diffs, 1000, s) for s in range(10)]
    return {"what": label, "hr_diff": round(hr, 3),
            "ci95_1000_seed0": seeds[0],
            "ci95_1000_seed0to9_lo_range": [min(s[0] for s in seeds), max(s[0] for s in seeds)],
            "ci95_1000_seed0to9_hi_range": [min(s[1] for s in seeds), max(s[1] for s in seeds)],
            "ci95_10000_seed0": ci(diffs, 10000, 0),
            "per_set": {k: {"diff": round(100 * float(v.mean()), 2), "ci95_10000": per_set_ci(v)} for k, v in diffs.items()}}


def test_part(r0_dir, m2_dir):
    out = []
    z0 = {s: np.load(Path(r0_dir) / f"test_{s}.npz") for s in TEST_HR}
    z1 = {s: np.load(Path(m2_dir) / f"test_{s}.npz") for s in TEST_HR}
    k0 = {s: [k for k in z0[s].files if k.endswith("_pred") and k != "fp32_pred"][0] for s in TEST_HR}
    k1 = {s: [k for k in z1[s].files if k.endswith("_pred") and k != "fp32_pred"][0] for s in TEST_HR}
    for s in TEST_HR:
        assert np.array_equal(z0[s]["y"], z1[s]["y"]), f"test {s} 정답 순서 다름"
        assert np.array_equal(z0[s]["fp32_pred"], z1[s]["fp32_pred"]), f"test {s} FP32 예측 다름"
    c = lambda z, k, s: (z[s][k] == z[s]["y"]).astype(float)
    out.append(summarize({s: c(z1, k1[s], s) - c(z0, k0[s], s) for s in TEST_HR}, "test M2rot_ort − R0"))
    out.append(summarize({s: c(z1, k1[s], s) - c(z1, "fp32_pred", s) for s in TEST_HR}, "test M2rot_ort − FP32"))
    out.append(summarize({s: c(z0, k0[s], s) - c(z0, "fp32_pred", s) for s in TEST_HR}, "test R0 − FP32"))
    out[0]["n"] = {s: int(len(z0[s]["y"])) for s in TEST_HR}
    return out


def dev_part(dirs):
    RC = S.RC
    fp = {}
    for d in HR:
        y, W, ref = (np.load(RC.CACHE_DIR / f"{d}_{k}.npy") for k in ("y", "clf", "ref_fp32"))
        fp[d] = (np.argmax(ref @ W, 1) == y).astype(float)
    models = {}
    for dd in dirs:
        for f in sorted(Path(dd).glob("pred_*.npz")):
            z = np.load(f)
            name = f.stem[5:]
            corr = {d: z[f"{d}_correct"].astype(float) for d in HR}
            if name in models:
                assert all(np.array_equal(models[name][d], corr[d]) for d in HR), f"{name}: 폴더끼리 사진별 결과가 다름"
                continue
            models[name] = corr
    out = [summarize({d: m[d] - fp[d] for d in HR}, f"dev {n} − FP32") for n, m in models.items()]
    names = list(models)
    for a in names:
        for b in names:
            if a < b:
                out.append(summarize({d: models[b][d] - models[a][d] for d in HR}, f"dev {b} − {a}"))
    return out


def screen_part(root):
    full = {}
    for f in glob.glob(os.path.join(root, "**", "paired_ci.jsonl"), recursive=True):
        if "hailo" in f.lower() or "dfc" in f.lower():
            continue
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if "hr_dtop1" in r and r.get("exec", "int") == "int":
                full.setdefault(r["model"], []).append((r["hr_dtop1"], f))
    scr = {}
    for f in glob.glob(os.path.join(root, "**", "screen*.jsonl"), recursive=True):
        if "hailo" in f.lower() or "dfc" in f.lower():
            continue
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("exec") != "int" or not all(d in r for d in HR):
                continue
            d = float(np.mean([r[x]["top1"] - r[x]["fp32_top1"] for x in HR]))
            cs = float(np.mean([r[x]["cos_mean"] for x in HR]))
            scr.setdefault(r["model"], []).append((d, cs, f))
    common = sorted(set(full) & set(scr))
    rows = [{"model": m, "full_hr": full[m][0][0], "screen_hr": scr[m][0][0], "screen_cos": scr[m][0][1],
             "full_file": full[m][0][1].replace("\\", "/").split("results/siglip2/")[-1],
             "screen_file": scr[m][0][2].replace("\\", "/").split("results/siglip2/")[-1]} for m in common]

    def spearman(a, b):
        ra, rb = (np.argsort(np.argsort(v)).astype(float) for v in (a, b))
        return round(float(np.corrcoef(ra, rb)[0, 1]), 3)
    res = {"n_models": len(rows), "rows": rows}
    if len(rows) >= 3:
        f_ = [r["full_hr"] for r in rows]
        res["spearman_screen_hr_vs_full"] = spearman([r["screen_hr"] for r in rows], f_)
        res["spearman_screen_cos_vs_full"] = spearman([r["screen_cos"] for r in rows], f_)
        res["pearson_screen_hr_vs_full"] = round(float(np.corrcoef([r["screen_hr"] for r in rows], f_)[0, 1]), 3)
        ok = [r for r in rows if r["full_hr"] > -5]          # 붕괴 모델을 빼고 (차이가 작은 쪽에서도 맞는지)
        if len(ok) >= 3:
            res["spearman_screen_hr_vs_full_excl_collapse"] = spearman([r["screen_hr"] for r in ok], [r["full_hr"] for r in ok])
            res["spearman_screen_cos_vs_full_excl_collapse"] = spearman([r["screen_cos"] for r in ok], [r["full_hr"] for r in ok])
            res["n_excl_collapse"] = len(ok)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-r0", required=True)
    ap.add_argument("--test-m2", required=True)
    ap.add_argument("--dev-dirs", nargs="+", required=True)
    ap.add_argument("--results-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from scripts import env_info
    env = env_info.collect()
    res = {"git": env["git"], "test": test_part(a.test_r0, a.test_m2), "dev": dev_part(a.dev_dirs),
           "screen_vs_full": screen_part(a.results_root)}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(res, open(Path(a.out) / "reanalysis.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for r in res["test"] + res["dev"]:
        print(r["what"][:110], r["hr_diff"], r["ci95_1000_seed0"], "10k", r["ci95_10000_seed0"],
              "lo", r["ci95_1000_seed0to9_lo_range"], "hi", r["ci95_1000_seed0to9_hi_range"])
    sv = res["screen_vs_full"]
    print("screen vs full: n", sv["n_models"], {k: v for k, v in sv.items() if k.startswith(("spearman", "pearson", "n_excl"))})


if __name__ == "__main__":
    main()
