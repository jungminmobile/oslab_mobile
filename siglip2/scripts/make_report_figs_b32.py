"""
make_report_figs_b32.py — D-048(B/32 제출) 보고서 그림 추가분 (raw에서 바로 그림, 숫자를 손으로 옮기지 않음)
================================================================================================
실행: D:/mokerton/venvs/figs/Scripts/python.exe siglip2/scripts/make_report_figs_b32.py   (저장소 루트에서, git 필요)
출력: siglip2/docs/figs/F3b_improvement_b32.png, F4b_min_snr_vs_acc.png, F5_hailo_b32.png + figs_data_b32.json
  F3b 제출 모델 P2 FP32 대비 개선율(10/27) + 이전 단계 O-AB 같은 실행·이전 실행 2회: 크기·피크 메모리·지연 1/2/4스레드
  F4b 배포 눈금 층별 SNR 최솟값 vs dev 정확도 손실: 제출 모델 B/32 5구성 (B/16 6구성 값은 figs_data_b32.json에만, 그리지 않음)
  F5  Hailo-10H 실칩 B/32: 배치 1 지연 vs dev 정확도 손실 (Zoo 공개 HEF Z와 우리 구성) — 준용 브랜치 raw를 git show로 읽음
F1(민감도)·F2(2×2)는 make_report_figs.py (10/24부터 B/32만 그림).
"""
import json
import statistics as st
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "siglip2"
G = R / "G8_candidates"
OUT = ROOT / "siglip2" / "docs" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"], "axes.unicode_minus": False,
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#898781", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "savefig.dpi": 300})
B16, B32, INK, MUTED, BAD = "#2a78d6", "#eb6834", "#0b0b0b", "#898781", "#d03b3b"
data = {}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def git_jl(ref_path):
    txt = subprocess.run(["git", "-C", str(ROOT), "show", ref_path], capture_output=True, text=True, encoding="utf-8", check=True).stdout
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def hr(path, needle, exclude=None):
    rows = [r for r in jl(path) if "hr_dtop1" in r and needle in r["model"] and (exclude is None or exclude not in r["model"])]
    assert len(rows) == 1, (path, needle, [r["model"] for r in rows])
    return rows[0]["hr_dtop1"]


# ---- F3b 개선율 (제출 모델 B/32 O-AB, 이 PC 3회)
def sessions(p, recovered=False):
    d = json.load(open(p, encoding="utf-8"))
    if recovered:
        return [(s["threads"], s["ms_per_img_mean"]) for s in d["sessions"]]
    return [(s["threads"], {k: v["ms_per_img_mean"] for k, v in s["results"].items()}) for s in d["sessions"]]


def ratios(rows, a, b, ref_key=None):
    out = {}
    for t in (1, 2, 4):
        rs = [r for tt, r in rows if tt == t]
        if ref_key:   # 사전 이상치 규칙 (b16/O_AB/README 2절): 같은 스레드 중앙값보다 ref가 10% 넘게 느린 세션 제외
            med = st.median([r[ref_key] for r in rs])
            rs = [r for r in rs if r[ref_key] <= 1.10 * med]
        out[t] = round(100 * (st.median([r[a] / r[b] for r in rs]) - 1), 1)
    return out


L = G / "latency_this_pc"
day = sessions(L / "all8_20261022/cross_latency.json") + sessions(L / "all8_20261022_rerun_t24/cross_latency_recovered.json", True)
# 대표값 = 10/27 실행의 제출 모델 P2 (D-052). 이전 단계 O-AB의 같은 실행·이전 실행은 비율이 안정적임을 보이는 재현 범위
runs = {"P2 (제출), 10/27": (ratios(sessions(L / "b32_p2_20261027/cross_latency.json"), "b32_p2", "b32_fp32"), L / "b32_p2_20261027/cross_latency.json", "b32_p2"),
        "O-AB (이전 단계), 10/27 같은 실행": (ratios(sessions(L / "b32_p2_20261027/cross_latency.json"), "b32_oab", "b32_fp32"), L / "b32_p2_20261027/cross_latency.json", "b32_oab"),
        "O-AB, 10/23 새벽": (ratios(sessions(L / "b32_cool_20261023/cross_latency.json"), "b32_oab", "b32_fp32"), L / "b32_cool_20261023/cross_latency.json", "b32_oab"),
        "O-AB, 10/22 낮 (8모델 실행)": (ratios(day, "b32_oab", "b32_fp32", ref_key="b16_fp32"), L / "all8_20261022/cross_latency.json", "b32_oab")}
f3 = {}
for name, (lat, p, key) in runs.items():
    d = json.load(open(p, encoding="utf-8"))
    mem = round(100 * (d["peak_mem"][key]["peak_wset_MiB"] / d["peak_mem"]["b32_fp32"]["peak_wset_MiB"] - 1), 1)
    size = round(100 * (d["files"][key]["bytes"] / d["files"]["b32_fp32"]["bytes"] - 1), 1)
    f3[name] = {"크기": size, "피크 메모리": mem, "지연 1T": lat[1], "지연 2T": lat[2], "지연 4T": lat[4]}
data["F3b"] = {"pct_vs_fp32": f3, "source": [str(p.relative_to(ROOT)) for _, p, _ in runs.values()]}
items = list(next(iter(f3.values())))
fig, ax = plt.subplots(figsize=(6.6, 3.0))
w = 0.2
cols = ["#eb6834", "#2a78d6", "#5b8c5a", "#8e6bbf"]
for i, (name, v) in enumerate(f3.items()):
    xs = [j + (i - 1.5) * w for j in range(len(items))]
    bars = ax.bar(xs, [v[k] for k in items], w, label=name, color=cols[i])
    for x, y in zip(xs, [v[k] for k in items]):
        ax.text(x, y + 1.5, f"{y:.1f}", ha="center", va="bottom", fontsize=5.2, color="white", fontweight="bold", rotation=90)
ax.set_xticks(range(len(items)), items)
ax.set_ylabel("FP32 대비 변화율 (%)")
ax.set_ylim(-82, 0)
ax.set_title("SigLIP2 B/32 제출 모델 P2의 FP32 대비 개선율 — 이 PC(i7-1165G7), 같은 실행 번갈아, 배치 1, 기본 세션", fontsize=8.5)
ax.legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "F3b_improvement_b32.png")
plt.close(fig)

# ---- F4b 배포 눈금 층별 SNR 최솟값 vs dev 정확도 손실
s32 = json.load(open(G / "b32/submit_prep/H5_qdq_snr/why_after_qdq_summary.json", encoding="utf-8"))
s16 = json.load(open(G / "b16/generalization/H5_qdq_snr/why_after_qdq_summary.json", encoding="utf-8"))
P32, P16a, P16o, P16r = (G / "b32/G3RQ/dev_full/paired_ci.jsonl", G / "b16/A1_ablation/dev_full/paired_ci.jsonl",
                         G / "b16/O_AB/dev_full/paired_ci.jsonl", R / "G6_m2rot_ort/dev_full/paired_ci.jsonl")
pts = [("B/32 원래", "b32", s32["b32_orig"], hr(P32, "int8__percentile")),
       ("B/32 회전만", "b32", s32["b32_rot"], hr(P32, "int8_rothad_")),
       ("B/32 M2만", "b32", s32["b32_m2"], hr(P32, "15at6u5-7at10u9_percentile")),
       ("B/32 회전+M2", "b32", s32["b32_m2rot"], hr(P32, "15at6u5-7at10u9_rothad")),
       ("B/32 예전 방식(R0)", "b32", s32["b32_r0"], hr(P32, "fc2L2"))]
pts16 = [("B/16 원래", "b16", s16["b16_orig"], hr(P16a, "int8__percentile")),
       ("B/16 회전만", "b16", s16["b16_rot"], hr(P16a, "int8_rothad_")),
       ("B/16 M2만", "b16", s16["b16_m2"], hr(P16a, "55at5-13at9_percentile")),
       ("B/16 회전+M2", "b16", s16["b16_m2rot_ort"], hr(P16o, "55at5-13at9_rothad_percentile", exclude="o1b")),
       ("B/16 O-AB", "b16", s16["b16_oab"], hr(P16o, "o1b")),
       ("B/16 예전 방식(R0)", "b16", s16["b16_r0"], hr(P16r, "fc2L2"))]
data["F4b"] = [{"name": n, "min_snr_db": s["full"]["min"], "mean_snr_db": s["full"]["mean"], "dev_dtop1": a} for n, _, s, a in pts]
data["F4b_dev_model_not_drawn"] = [{"name": n, "min_snr_db": s["full"]["min"], "mean_snr_db": s["full"]["mean"], "dev_dtop1": a} for n, _, s, a in pts16]
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
for ax, key, lab in ((axs[0], "min", "가장 나쁜 층 SNR (dB)"), (axs[1], "mean", "79개 층 평균 SNR (dB)")):
    for n, m, s, a in pts:
        ax.scatter(s["full"][key], -a, color=B32 if m == "b32" else B16, s=22, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel(lab)
axs[0].invert_yaxis()          # sharey라 한 번만 뒤집는다 (작은 손실이 위)
ticks = [0.5, 1, 2, 5, 10, 20, 50]
axs[0].set_yticks(ticks, [f"{t:g}" for t in ticks])
axs[0].minorticks_off()
axs[1].minorticks_off()
for n, m, s, a in pts:
    lab = n.split(" ", 1)[1]
    axs[1].annotate(lab, (s["full"]["mean"], -a), fontsize=5.8, xytext=(3, 3), textcoords="offset points", color=MUTED,
                    ha="right" if s["full"]["mean"] > 31 else "left")
    if -a > 10:                # 붕괴한 두 점은 왼쪽 패널에서 한 라벨로 단다
        continue
    axs[0].annotate(lab, (s["full"]["min"], -a), fontsize=5.8, xytext=(3, 3), textcoords="offset points", color=MUTED)
axs[0].annotate("원래·회전만 (붕괴)", (1.0, 57), fontsize=5.8, xytext=(8, 0), textcoords="offset points", color=MUTED, va="center")
axs[0].set_ylabel("dev 손실 (%p, 로그, 위가 좋음)")
fig.suptitle("SigLIP2 B/32: 배포 파일 눈금으로 잰 층별 8비트 출력 SNR과 정확도 — 최솟값은 순서가 맞고, 평균은 맞지 않는다", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "F4b_min_snr_vs_acc.png")
plt.close(fig)

# ---- F5 Hailo-10H 실칩 B/32 (준용 G6f: 같은 세션 지연, 실칩 dev 전체 정확도)
spd = {r["hef"]: r for r in git_jl("origin/siglip2/g6f-ceiling:results/siglip2/G6f/results_speed.jsonl") if r.get("what") == "g6f_speed"}
acc = {r["hef"]: r for r in git_jl("origin/siglip2/g6f-ceiling:results/siglip2/G6f/results_devfull.jsonl") if "hr_dtop1" in r}
acc_b = {r.get("hef"): r for r in git_jl("origin/siglip2/g6e-b16-g:results/siglip2/G6b/results_chip.jsonl") if "hr_dtop1" in r}
for k in ("b32_Z", "b32_M"):
    if k in acc_b and k not in acc:
        acc[k] = acc_b[k]
if "b32_GF" not in acc and "b32_G" in acc:      # GF는 G와 출력 1,200장 비트 동일 (G6f) → G 값
    acc["b32_GF"] = dict(acc["b32_G"], note="= G (비트 동일)")
f5 = []
for k in ("b32_Z", "b32_G", "b32_GF", "b32_ALL8F"):
    if k in spd and k in acc:
        f5.append({"hef": k, "b1_ms": spd[k]["b1_mean_ms"], "b1_vs_Z_pct": spd[k]["b1_vs_Z_pct"], "dev_dtop1": acc[k]["hr_dtop1"],
                   "ci95": acc[k].get("hr_dtop1_ci95"), "note": acc[k].get("note", "")})
data["F5"] = {"points": f5, "source": ["origin/siglip2/g6f-ceiling:results/siglip2/G6f/results_speed.jsonl (같은 세션)",
                                       "origin/siglip2/g6f-ceiling:results/siglip2/G6f/results_devfull.jsonl", "origin/siglip2/g6e-b16-g:results/siglip2/G6b/results_chip.jsonl (Z)"]}
fig, ax = plt.subplots(figsize=(4.6, 3.0))
names = {"b32_Z": "Hailo Model Zoo 공개 HEF", "b32_G": "G (우리)", "b32_GF": "GF (우리)", "b32_ALL8F": "ALL8F (우리, 전부 8비트)"}
for p in f5:
    c = MUTED if p["hef"] == "b32_Z" else B32
    ax.errorbar(p["b1_ms"], p["dev_dtop1"], yerr=None if not p["ci95"] else [[p["dev_dtop1"] - p["ci95"][0]], [p["ci95"][1] - p["dev_dtop1"]]],
                fmt="o", color=c, capsize=2, ms=5)
    ax.annotate(names[p["hef"]], (p["b1_ms"], p["dev_dtop1"]), fontsize=6.5, xytext=(4, 3), textcoords="offset points", color=c)
ax.set_xlim(13.0, 18.6)
ax.set_xlabel("배치 1 지연 (ms, 같은 세션)")
ax.set_ylabel("dev 정확도 Δ vs FP32 (%p)")
ax.set_title("Hailo-10H 실칩, SigLIP2 B/32 — 왼쪽 위가 좋음", fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "F5_hailo_b32.png")
plt.close(fig)

json.dump(data, open(OUT / "figs_data_b32.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(data, ensure_ascii=False)[:2500])
