"""보고서 v13 그림 1·2·3: 내부 약어(M2, P2, O-AB) 대신 보고서 본문과 같은 이름으로 다시 그린다.
값은 make_report_figs.py가 남긴 figs_data.json(F1·F2)과 make_report_figs_b32.py가 남긴 figs_data_b32.json(F3b)을 그대로 쓴다 (새 측정 없음).
사용: python siglip2/scripts/make_figs_v13.py  → siglip2/docs/figs/F1_sensitivity_v13.png, F2_ablation_2x2_v13.png, F3b_improvement_b32_v13.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGS = Path(__file__).resolve().parents[1] / "docs" / "figs"
plt.rcParams.update({"font.family": ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#898781", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "savefig.dpi": 300})
B32, INK = "#eb6834", "#0b0b0b"

# ---- 그림 2: 레지스터 캐시·회전 2×2 (dev 전체, FP32 대비 Δtop-1)
t = json.load(open(FIGS / "figs_data.json", encoding="utf-8"))["F2"]["b32"]
keys = ["원래", "회전만", "M2만", "회전+M2"]                        # figs_data.json 안의 키
labels = ["처방 없음", "회전만", "레지스터\n캐시만", "회전 +\n레지스터 캐시"]   # 5.3 SNR 표와 같은 이름
assert set(keys) == set(t), t.keys()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
for ax, ylim, title in [(a1, (-70, 2), "전체 범위"), (a2, (-2.4, 0.4), "확대 (레지스터 캐시를 쓴 구성)")]:
    xs = list(range(4))
    ys = [t[k][0] for k in keys]
    lo = [t[k][0] - t[k][1][0] for k in keys]
    hi = [t[k][1][1] - t[k][0] for k in keys]
    ax.bar(xs, ys, 0.6, color=B32)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    if ax is a2:
        for xx, yy, k in zip(xs, ys, keys):
            if yy > ylim[0]:
                ax.text(xx, t[k][1][0] - 0.05, f"{yy:+.2f}", ha="center", va="top", fontsize=7, color=INK)
            else:
                ax.text(xx, ylim[0] + 0.08, f"{yy:+.1f}", ha="center", va="bottom", fontsize=7, color="#ffffff", fontweight="bold")
    ax.set_xticks(range(4), labels, fontsize=7.5)
    ax.set_ylim(*ylim)
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_title(title, fontsize=9, color=INK, loc="left")
a1.set_ylabel("FP32 대비 Δtop-1 (%p, dev 전체)")
fig.suptitle("SigLIP2 B/32: 행렬곱 전부 INT8에서 처방 하나씩 켜기 (오차 막대 = 95% 구간)", fontsize=9, x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig(FIGS / "F2_ablation_2x2_v13.png")
plt.close(fig)

# ---- 그림 3: 제출 모델의 FP32 대비 개선율 (i7, 같은 실행에서 번갈아)
f3 = json.load(open(FIGS / "figs_data_b32.json", encoding="utf-8"))["F3b"]["pct_vs_fp32"]
names = {"P2 (제출 모델), 10/27 저녁 두 실행 합침": "제출 모델 (대표값)",
         "P2, 10/27 오후 기본 세션": "제출 모델 (메모리 옵션 없는 기본 설정)",
         "O-AB (전 단계 모델), 10/27 저녁 두 실행 합침": "직전 모델(⑥까지, 대표값과 같은 실행)",
         "O-AB, 10/23 새벽 기본 세션": "직전 모델(⑥까지, 다른 날)"}
assert list(f3) == list(names), list(f3)
items = list(next(iter(f3.values())))
fig, ax = plt.subplots(figsize=(6.6, 3.5))
w = 0.2
cols = ["#eb6834", "#2a78d6", "#5b8c5a", "#8e6bbf"]
for i, (name, v) in enumerate(f3.items()):
    xs = [j + (i - 1.5) * w for j in range(len(items))]
    ax.bar(xs, [v[k] for k in items], w, label=names[name], color=cols[i])
    for x, y in zip(xs, [v[k] for k in items]):
        ax.text(x, y + 1.5, f"{y:.1f}", ha="center", va="bottom", fontsize=5.2, color="white", fontweight="bold", rotation=90)
ax.set_xticks(range(len(items)), [k.replace("1T", "1스레드").replace("2T", "2스레드").replace("4T", "4스레드") for k in items])
ax.set_ylabel("FP32 대비 변화율 (%)")
ax.set_ylim(-82, 0)
ax.set_title("SigLIP2 B/32 제출 모델의 FP32 대비 개선율 (i7-1165G7, 같은 실행에서 번갈아, 배치 1)", fontsize=8.5)
ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
fig.tight_layout()
fig.savefig(FIGS / "F3b_improvement_b32_v13.png")
plt.close(fig)

# ---- 그림 1: 민감도 지도 (오른쪽 칸 연산 이름이 겹치던 것을 두 줄로 나눔)
s32 = json.load(open(FIGS / "figs_data.json", encoding="utf-8"))["F1"]["b32"]
MUTED = "#898781"
blocks = ["only_patch"] + [f"only_b{i}" for i in range(12)] + ["only_head"]
blabels = ["패치"] + [str(i) for i in range(12)] + ["헤드"]
ops = ["only_qkv", "only_attn_qk", "only_attn_av", "only_proj", "only_fc1", "only_fc2"]
olabels = ["QKV", "QKᵀ", "AV", "출력\n투영", "MLP\nfc1", "MLP\nfc2"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [14, 6]})
for ax, keys, labels, title in [(a1, blocks, blabels, "블록 하나만 INT8 (나머지 FP32)"), (a2, ops, olabels, "연산 유형 하나만 INT8")]:
    x = list(range(len(keys)))
    ax.bar(x, [s32[k] for k in keys], 0.62, color=B32)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.95, color=MUTED, lw=0.8, ls="--")
    ax.set_title(title, fontsize=9, color=INK, loc="left")
a1.set_ylabel("FP32 대비 코사인 유사도")
a1.text(13.6, 0.955, "0.95", color=MUTED, fontsize=7, va="bottom", ha="right")
a1.annotate("블록 0과 5~11이 민감", xy=(6, 0.52), xytext=(7.5, 0.80), fontsize=8, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
fig.suptitle("SigLIP2 B/32 민감도 지도: 하나만 INT8로 바꿨을 때 FP32 대비 코사인 유사도 (dev 6×200장, 고화질 5종)", fontsize=9, x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig(FIGS / "F1_sensitivity_v13.png")
plt.close(fig)
print("ok")
