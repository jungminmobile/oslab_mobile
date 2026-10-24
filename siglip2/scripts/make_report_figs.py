"""
make_report_figs.py — 예선 보고서 그림 (raw에서 바로 그림, 숫자를 손으로 옮기지 않음)
=========================================================================================
실행: D:/mokerton/venvs/figs/Scripts/python.exe siglip2/scripts/make_report_figs.py   (저장소 루트에서)
  그림 전용 가상환경(matplotlib)을 쓴다 — 실험 환경(siglip2 venv)은 건드리지 않음
출력: siglip2/docs/figs/*.png (300 dpi) + figs_data.json (그림에 쓴 숫자 전부)
  F1 민감도 지도 (제출 모델 B/32): 하나만 INT8일 때 고화질 cos — 블록 14개 + 연산 유형 6개
  F2 2×2 단독 효과 (제출 모델 B/32): 원래 / 회전만 / M2만 / 회전+M2, dev 전체 Δtop-1과 95% 구간
  (10/24 준용 검토 M1: 본문 그림은 제출 모델만. B/16 값은 figs_data.json에만 남긴다)
  F3 FP32 대비 개선율 (B/16 최종): 크기·피크 메모리·지연(1·2·4스레드), 기기 4개
  F4 처방별 8비트 출력 SNR (원래·회전만·M2만·회전+M2): fc1·fc2 입력, 블록별 (E4b)
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results" / "siglip2"
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


def sens(path):
    """같은 모델이 여러 줄이면 마지막 줄 (재측정 반영)"""
    d = {}
    for r in jl(path):
        k = r["model"].split("_s8_")[-1].removesuffix(".onnx")
        d[k] = r["hr_cos"]
    return d


# ---- F1 민감도 지도
s16 = sens(R / "G3_sensitivity" / "screen.jsonl")
s32 = sens(R / "G8_candidates" / "b32" / "G3_sensitivity" / "screen.jsonl")
blocks = ["only_patch"] + [f"only_b{i}" for i in range(12)] + ["only_head"]
blabels = ["패치"] + [str(i) for i in range(12)] + ["헤드"]
ops = ["only_qkv", "only_attn_qk", "only_attn_av", "only_proj", "only_fc1", "only_fc2"]
olabels = ["QKV", "QKᵀ", "AV", "출력 투영", "MLP fc1", "MLP fc2"]
data["F1"] = {"b16": {k: s16[k] for k in blocks + ops}, "b32": {k: s32[k] for k in blocks + ops},
              "source": ["G3_sensitivity/screen.jsonl", "G8_candidates/b32/G3_sensitivity/screen.jsonl"]}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [14, 6]})
w = 0.62
for ax, keys, labels, title in [(a1, blocks, blabels, "블록 하나만 INT8 (나머지 FP32)"), (a2, ops, olabels, "연산 유형 하나만 INT8")]:
    x = range(len(keys))
    ax.bar(list(x), [s32[k] for k in keys], w, color=B32)
    ax.set_xticks(list(x), labels, rotation=0 if ax is a1 else 30, ha="center" if ax is a1 else "right")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.95, color=MUTED, lw=0.8, ls="--")
    ax.set_title(title, fontsize=9, color=INK, loc="left")
a1.set_ylabel("FP32 대비 임베딩 cos (고화질 5종)")
a1.text(13.6, 0.955, "0.95", color=MUTED, fontsize=7, va="bottom", ha="right")
a1.annotate("블록 0과 5~11이 민감", xy=(6, 0.52), xytext=(7.5, 0.80), fontsize=8, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
fig.suptitle("SigLIP2 B/32 민감도 지도 — 하나만 INT8로 바꿨을 때 FP32 대비 cos (dev 6×200)", fontsize=9, x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig(OUT / "F1_sensitivity.png")
plt.close(fig)

# ---- F2 2×2 단독 효과
def twobytwo(path, tag_m2):
    rows = [r for r in jl(path) if "model" in r]
    out = {}
    for r in rows:
        m = r["model"]
        if "fc2L2" in m:
            continue
        k = ("회전+M2" if "rothad" in m else "M2만") if tag_m2 in m else ("회전만" if "rothad" in m else "원래")
        out[k] = (r["hr_dtop1"], r["hr_dtop1_ci95"])
    return out


t16 = twobytwo(R / "G8_candidates" / "b16" / "A1_ablation" / "dev_full" / "paired_ci.jsonl", "rcmaskbcmlp")
t32 = twobytwo(R / "G8_candidates" / "b32" / "G3RQ" / "dev_full" / "paired_ci.jsonl", "rcmaskbcmlp")
order = ["원래", "회전만", "M2만", "회전+M2"]
data["F2"] = {"b16": t16, "b32": t32, "source": ["G8_candidates/b16/A1_ablation/dev_full/paired_ci.jsonl", "G8_candidates/b32/G3RQ/dev_full/paired_ci.jsonl"]}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1]})
for ax, ylim, title in [(a1, (-70, 2), "전체 범위"), (a2, (-2.4, 0.4), "확대 (M2를 쓴 구성)")]:
    for t, c in [(t32, B32)]:
        xs = list(range(4))
        ys = [t[k][0] for k in order]
        lo = [t[k][0] - t[k][1][0] for k in order]
        hi = [t[k][1][1] - t[k][0] for k in order]
        ax.bar(xs, ys, 0.6, color=c)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
        if ax is a2:
            for xx, yy, k in zip(xs, ys, order):
                if yy > ylim[0]:     # 구간 아래쪽 끝 바로 밑에 값
                    ax.text(xx, t[k][1][0] - 0.05, f"{yy:+.2f}", ha="center", va="top", fontsize=7, color=INK)
                else:                # 축 밖으로 나간 막대는 막대 안 아래쪽에
                    ax.text(xx, ylim[0] + 0.08, f"{yy:+.1f}", ha="center", va="bottom", fontsize=7, color="#ffffff", fontweight="bold")
    ax.set_xticks(range(4), order)
    ax.set_ylim(*ylim)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
a1.set_ylabel("FP32 대비 Δtop-1 (%p, dev 전체)")
fig.suptitle("SigLIP2 B/32: MatMul 전부 INT8에서 기법 하나씩 켜기 (오차 막대 = 95% 구간)", fontsize=9, x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig(OUT / "F2_ablation_2x2.png")
plt.close(fig)

# ---- F3 개선율 (B/16 최종)
m = json.load(open(R / "G8_candidates" / "b16" / "A4_metrics" / "metrics.json", encoding="utf-8"))
lat = m["latency"]
items = [("크기", m["stats"]["m2rot"]["MB"] / m["stats"]["fp32"]["MB"] - 1),
         ("피크 메모리", m["peak_mem_MiB"]["m2rot"] / m["peak_mem_MiB"]["fp32"] - 1)]
for t in ["1", "2", "4"]:
    items.append((f"지연 {t}스레드", lat["m2rot"][t]["median_ms"] / lat["fp32"][t]["median_ms"] - 1))
dev = [("i7-1165G7", -0.499, -0.526), ("i7 (다른 실행)", -0.517, -0.583), ("Ultra 9 285K", -0.557, -0.623), ("Ultra 7 258V", -0.512, -0.627)]
data["F3"] = {"items_pct": {k: round(100 * v, 1) for k, v in items}, "devices_t4_t1": dev,
              "source": ["G8_candidates/b16/A4_metrics/metrics.json", "G6_m2rot_ort/README.md (258V)", "G6b (285K)", "G8_candidates/latency_this_pc"]}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1]})
ys = range(len(items))
a1.barh(list(ys), [100 * v for _, v in items], 0.55, color=B16)
for y, (_, v) in zip(ys, items):
    a1.text(100 * v - 1.5, y, f"{100 * v:.1f}%", va="center", ha="right", fontsize=8, color=INK)
a1.set_yticks(list(ys), [k for k, _ in items])
a1.invert_yaxis()
a1.set_xlim(-90, 0)
a1.set_xlabel("FP32 대비 변화율 (%) — 이 PC, 같은 실행")
a1.set_title("B/16 최종 (M2rot_ort)", fontsize=9, color=INK, loc="left")
yd = range(len(dev))
a2.barh([y - 0.18 for y in yd], [100 * d[1] for d in dev], 0.34, color=B16, label="4스레드")
a2.barh([y + 0.18 for y in yd], [100 * d[2] for d in dev], 0.34, color="#86b6ef", label="1스레드")
a2.set_yticks(list(yd), [d[0] for d in dev])
a2.invert_yaxis()
a2.set_xlim(-70, 0)
a2.set_xlabel("지연 변화율 (%) — 기기마다 FP32와 같은 실행")
a2.set_title("기기가 달라도 개선율은 비슷", fontsize=9, color=INK, loc="left")
a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "F3_improvement.png")
plt.close(fig)

# ---- F4 거대 활성값 처방의 효과: fc1·fc2 입력을 8비트로 했을 때의 출력 SNR (E4b, Percentile 99.999 눈금)
p4 = R / "G8_candidates" / "b16" / "E4_why_after" / "why_after_pct.jsonl"
if p4.exists():
    rows4 = jl(p4)
    graphs = ["원래", "회전만", "M2만", "회전+M2"]
    cols = {"원래": "#898781", "회전만": "#eda100", "M2만": "#1baf7a", "회전+M2": B16}
    snr = {}
    for g in graphs:
        rs = [r for r in rows4 if r["graph"] == g]
        fc1 = [r["out_snr_db"] for r in rs if re.search(r"fc1(_\d+)?$", r["role"])][:12]
        fc2 = [r["out_snr_db"] for r in rs if re.search(r"fc2(_\d+)?$", r["role"])][:12]
        snr[g] = {"fc1": fc1, "fc2": fc2}
    data["F4"] = {"snr_db": snr, "source": "G8_candidates/b16/E4_why_after/why_after_pct.jsonl"}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, key, title in [(a1, "fc1", "MLP fc1 입력 (LayerNorm 뒤, 잔차 흐름에서 옴)"), (a2, "fc2", "MLP fc2 입력 (GELU 뒤)")]:
        for g in graphs:
            dashed = g in ("회전만", "회전+M2")      # 겹치는 쌍(원래=회전만 / M2만=회전+M2)이 둘 다 보이게: 굵은 실선 위에 점선
            ax.plot(range(12), snr[g][key], marker="o" if not dashed else "s", ms=4 if not dashed else 3,
                    lw=3.2 if not dashed else 1.6, ls="--" if dashed else "-", color=cols[g], label=g, zorder=3 if dashed else 2)
        ax.set_xticks(range(12), [str(i) for i in range(12)])
        ax.set_xlabel("블록")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.axvspan(4.5, 11.5, color="#f3f2ee", zorder=0)
    a1.set_ylabel("출력 SNR (dB)")
    a1.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
    a2.text(8, a2.get_ylim()[1] * 0.97, "거대 값이 생기는 블록 5~11", fontsize=7, color=MUTED, ha="center", va="top")
    fig.text(0.01, 0.01, "fc2 입력(MLP 은닉 3072차원)은 회전 대상이 아니라 원래=회전만, M2만=회전+M2 선이 겹친다 (굵은 실선 위의 점선).", fontsize=7, color=MUTED, ha="left", va="bottom")
    fig.suptitle("활성값만 8비트(Percentile 99.999)로 했을 때 층 출력의 SNR — dev 64장", fontsize=9, x=0.01, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "F4_activation_snr.png")
    plt.close(fig)

json.dump(data, open(OUT / "figs_data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", sorted(p.name for p in OUT.glob("*.png")))
