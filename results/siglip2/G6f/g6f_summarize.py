"""g6f_summarize.py — G6f 정리 (README 2·3절): 6×200 비트 비교·참고 정확도·에뮬레이터 비교, 속도 표, 상한·G 위치, 컨텍스트-지연 표, 감시 로그와 bench 세션 겹침.
정확도 계산은 G6d g6d_summarize.py와 같다 (L2 정규화, 분류기 argmax, 고화질 = 데이터셋 5개 평균, 구간 = s2_paired_ci.boot_hr 시드 0·1000회). 단 데이터셋마다 앞 200장만.
사용: python g6f_summarize.py --chip <AGX hailo_g6f/out 사본> --gfull <G6d G dev 전체 임베딩 npz> --emu <에뮬레이터 npz 폴더> \
        --ref ~/siglip2_work/g6/ref --s2dir ../../../siglip2 --g6d ../G6d --outdir .
"""
import argparse, hashlib, json, re, subprocess, sys
from datetime import datetime
from pathlib import Path
import numpy as np

DATA = ["cifar_A", "pets_dev", "flowers_dev", "cars_dev", "food_dev", "caltech_dev"]
HR = DATA[1:]
K = 200
ORDER = ["b32_Z", "b32_G", "b32_GF", "b32_ALL8", "b32_ALL8F"]
ap = argparse.ArgumentParser()
for x in ["chip", "gfull", "emu", "ref", "s2dir", "g6d", "outdir"]:
    ap.add_argument(f"--{x}", required=True)
a = ap.parse_args()
sys.path.insert(0, a.s2dir)
from s2_paired_ci import boot_hr          # noqa: E402
l2n = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
repo = Path(__file__).resolve().parents[3]
git = lambda *x: subprocess.run(["git", "-C", str(repo), *x], capture_output=True, text=True).stdout.strip()
base = {"commit": git("rev-parse", "--short", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
chip, R, out = Path(a.chip), Path(a.ref), Path(a.outdir)


# ---------- 정확도 (6×200) ----------
def load_chip(h):
    p = chip / f"eval_{h}" / f"emb_{h}.npz"
    return (np.load(p), json.loads((chip / f"eval_{h}" / "eval_info.json").read_text())["models"][h]) if p.exists() else (None, None)


def metrics(z):
    r, P = {}, {}
    for d in DATA:
        y, W, ref = np.load(R / f"b32_{d}_y.npy")[:K], np.load(R / f"b32_{d}_clf.npy"), l2n(np.load(R / f"b32_{d}_ref_fp32.npy")[:K])
        raw = z[f"{d}_emb"].astype(np.float32)[:K]; assert raw.shape == ref.shape, (d, raw.shape, ref.shape)
        e = l2n(raw); pred, fp = np.argmax(e @ W, 1), np.argmax(ref @ W, 1); c = (e * ref).sum(1)
        P[d] = {"ok": (pred == y).astype(float), "fp": (fp == y).astype(float)}
        r[d] = {"top1": round(100 * float((pred == y).mean()), 2), "fp32_top1": round(100 * float((fp == y).mean()), 2),
                "cos_mean": round(float(c.mean()), 4), "cos_min": round(float(c.min()), 4)}
    return {"hr_dtop1": round(float(np.mean([100 * (P[d]["ok"].mean() - P[d]["fp"].mean()) for d in HR])), 2),
            "hr_dtop1_ci95": boot_hr({d: P[d]["ok"] - P[d]["fp"] for d in HR}),
            "hr_top1": round(float(np.mean([100 * P[d]["ok"].mean() for d in HR])), 2),
            "hr_fp32_top1": round(float(np.mean([100 * P[d]["fp"].mean() for d in HR])), 2),
            "hr_cos": round(float(np.mean([r[d]["cos_mean"] for d in HR])), 4), "hr_cos_min": min(r[d]["cos_min"] for d in HR),
            "cifar_A_dtop1": round(r["cifar_A"]["top1"] - r["cifar_A"]["fp32_top1"], 2), "cifar_A_cos": r["cifar_A"]["cos_mean"], "per_set": r}


def bitcmp(za, zb):
    eq = {d: int((za[f"{d}_emb"][:K] == zb[f"{d}_emb"][:K]).all(1).sum()) for d in DATA}
    cos = {d: round(float((l2n(za[f"{d}_emb"][:K].astype(np.float64)) * l2n(zb[f"{d}_emb"][:K].astype(np.float64))).sum(1).mean()), 6) for d in DATA}
    return {"n_equal": sum(eq.values()), "n": K * len(DATA), "bit_equal": sum(eq.values()) == K * len(DATA), "per_set_equal": eq, "cos_mean": cos}


acc = []
gfull = np.load(a.gfull)
emb = {"b32_G(dev 전체 앞 200)": gfull}
for h in ["b32_GF", "b32_ALL8", "b32_ALL8F", "b32_G"]:
    z, info = load_chip(h)
    if z is not None:
        emb[h] = z
        acc.append({"what": "g6f_chip200", "hef": h, **base, "hef_sha256": info["hef_sha256"], "k": info["k"], **metrics(z)})
for t in ["b32_G", "b32_ALL8"]:
    p = Path(a.emu) / f"emu_{t}.npz"
    if p.exists():
        emb[f"emu_{t}"] = np.load(p)
        acc.append({"what": "g6f_emu200", "hef": f"emu_{t}", **base, "npz_sha256_16": hashlib.sha256(p.read_bytes()).hexdigest()[:16], **metrics(emb[f"emu_{t}"])})
for x, y in [("b32_GF", "b32_G(dev 전체 앞 200)"), ("b32_ALL8F", "b32_ALL8"), ("emu_b32_G", "b32_G(dev 전체 앞 200)"), ("emu_b32_ALL8", "b32_ALL8")]:
    if x in emb and y in emb:
        acc.append({"what": "g6f_bitcmp", "a": x, "b": y, **base, **bitcmp(emb[x], emb[y])})
(out / "results_acc200.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in acc))

# ---------- 감시 로그 → 외부 프로세스 시각 ----------
watch = [l.split() for l in (chip / "watch.log").read_text().splitlines() if l.strip()] if (chip / "watch.log").exists() else []
foreign = [w[0] for w in watch if not w[1].endswith("=0")]
log = (chip / "run.log").read_text().splitlines()
T = lambda s: datetime.strptime(s, "%H:%M:%S")
# bench 세션 구간: 첫 HEF '시작'부터 '끝'까지 (run.log)
sess = {}
for l in log:
    m = re.match(r"(\d\d:\d\d:\d\d) other_hailo=(\d+) bench b(\d) s(\d) (\S+) 시작", l)
    if m:
        k = f"b{m[3]}_s{m[4]}"; sess.setdefault(k, {"start": m[1], "other_hailo_at_start": []})["other_hailo_at_start"].append(int(m[2]))
    m = re.match(r"(\d\d:\d\d:\d\d) bench b(\d) s(\d) 끝", l)
    if m:
        sess[f"b{m[2]}_s{m[3]}"]["end"] = m[1]
for k, s in sess.items():
    s["foreign_in_window"] = [f for f in foreign if "end" in s and T(s["start"]) <= T(f) <= T(s["end"])]
    s["watch_lines_in_window"] = sum(1 for w in watch if "end" in s and T(s["start"]) <= T(w[0]) <= T(s["end"]))

# ---------- 속도 ----------
ctx = {}
for p in chip.glob("parsehef_*.txt"):
    m = re.search(r"Number of contexts: (\d+)", p.read_text()); ctx[p.stem[9:]] = int(m[1]) if m else None
speed = {}
for b in (1, 8):
    for s in (1, 2, 3):
        for p in sorted((chip / f"bench_b{b}_s{s}").glob("*/bench.json")):
            r = json.loads(p.read_text())["runs"][0]
            speed.setdefault(r["model"], {}).setdefault(b, []).append(r["mean_ms"])
rows = []
z1 = np.mean(speed["b32_Z"][1]); z8 = np.mean(speed["b32_Z"][8])
for h in [h for h in ORDER if h in speed]:
    r = {"what": "g6f_speed", "hef": h, **base, "contexts_parsehef": ctx.get(h)}
    for b, zz in ((1, z1), (8, z8)):
        v = speed[h][b]; assert len(v) == 3, (h, b, v)
        r.update({f"b{b}_mean_ms_sessions": [round(x, 3) for x in v], f"b{b}_mean_ms": round(float(np.mean(v)), 3),
                  f"b{b}_vs_Z_pct": round(100 * (np.mean(v) / zz - 1), 2), f"b{b}_spread_pct": round(100 * (max(v) - min(v)) / np.mean(v), 2),
                  f"b{b}_img_per_s": round(1000 * b / np.mean(v), 1)})
        rj = chip / f"run2_b{b}_{h}.json"
        if rj.exists():
            t = rj.read_text(); f = re.findall(r'"FPS"\s*:\s*"?([\d.]+)', t); r[f"b{b}_run2_fps"] = float(f[0]) if f else None
    rows.append(r)
S = {r["hef"]: r for r in rows}
derived = {"what": "g6f_derived", **base}
cands = [h for h in ("b32_ALL8", "b32_ALL8F") if h in S]
if cands:
    c = min(cands, key=lambda h: S[h]["b1_mean_ms"]); zc, g, cc = S["b32_Z"]["b1_mean_ms"], S["b32_G"]["b1_mean_ms"], S[c]["b1_mean_ms"]
    derived.update({"ceiling_hef": c, "ceiling_b1_ms": cc, "ceiling_vs_Z_pct": S[c]["b1_vs_Z_pct"],
                    "G_fraction_of_ceiling": round((zc - g) / (zc - cc), 3) if zc - cc > 0 else "정의 안 됨"})
if "b32_GF" in S:
    derived["GF_vs_G_b1_pct"] = round(100 * (S["b32_GF"]["b1_mean_ms"] / S["b32_G"]["b1_mean_ms"] - 1), 2)
    derived["GF_vs_G_b8_pct"] = round(100 * (S["b32_GF"]["b8_mean_ms"] / S["b32_G"]["b8_mean_ms"] - 1), 2)

# ---------- 컨텍스트-지연 ----------
g6d = {r["hef"]: r for r in map(json.loads, (Path(a.g6d) / "results_speed.jsonl").read_text().splitlines()) if r.get("what") == "g6d_speed"}
ctab = [{"hef": h, "source": "G6f", "contexts": S[h]["contexts_parsehef"], "b1_ms": S[h]["b1_mean_ms"], "b8_ms": S[h]["b8_mean_ms"]} for h in S]
ctab += [{"hef": h, "source": "G6d", "contexts": r["contexts"], "b1_ms": r["b1_mean_ms"], "b8_ms": r["b8_mean_ms"]} for h, r in g6d.items() if h.startswith("b32_") and h not in S]   # Z·G는 이번 세션 값만 (중복 제거)


def fit(pts):
    x = np.array([p["contexts"] for p in pts], float); y = np.array([p["b1_ms"] for p in pts], float)
    if len(set(x)) < 2:
        return None
    k, c = np.polyfit(x, y, 1); res = y - (k * x + c)
    return {"n": len(x), "slope_ms_per_ctx": round(float(k), 3), "intercept_ms": round(float(c), 2), "r2": round(float(1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()), 3)}


derived["ctx_fit_G6f"] = fit([p for p in ctab if p["source"] == "G6f"])
derived["ctx_fit_all"] = fit(ctab)
derived["ctx_fit_G6d_only"] = fit([p for p in ctab if p["source"] == "G6d"])
derived["bench_sessions"] = sess
derived["watch"] = {"lines": len(watch), "foreign_lines": len(foreign), "first": watch[0][0] if watch else None, "last": watch[-1][0] if watch else None}
(out / "results_speed.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows + [derived] + [{"what": "g6f_ctx", **p} for p in ctab]))
for r in acc + rows + [derived]:
    print(json.dumps({k: v for k, v in r.items() if k not in ("per_set", "per_set_equal", "bench_sessions")}, ensure_ascii=False))
