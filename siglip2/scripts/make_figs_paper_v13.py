"""보고서 v13 그림 1~4를 논문 그림 형식으로 다시 그린다 (새 측정 없음).

- 값: figs_data.json(F1·F2), figs_data_b32.json(F3b·F5) — make_report_figs*.py가 원자료에서 뽑아 둔 값 그대로
- 형식: 본문 폭(6.25 in)에 100%로 넣는 크기로 그려서 글자가 본문과 비슷한 크기(8 pt)로 보이게 한다.
  그림 안 제목은 없고(캡션이 대신함), 패널은 (a)(b), 눈금은 안쪽.
  색: 우리 방법 = 파랑, 문제·대조 = 주황, 기준(처방 없음·공개 HEF) = 회색. 네 그림에서 같은 뜻에 같은 색을 쓴다.
사용: python siglip2/scripts/make_figs_paper_v13.py → siglip2/docs/figs/F{1,2,3,4}_paper.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGS = Path(__file__).resolve().parents[1] / "docs" / "figs"
W = 6.25  # 본문 폭 (inch)
plt.rcParams.update({
    "font.family": ["Malgun Gothic", "Noto Sans CJK KR", "AppleGothic", "NanumGothic", "DejaVu Sans"],
    "axes.unicode_minus": True, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": True, "axes.spines.right": True, "axes.grid": False,
    "axes.edgecolor": "#222222", "axes.labelcolor": "#222222", "xtick.color": "#222222", "ytick.color": "#222222",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 400, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "legend.frameon": False, "hatch.linewidth": 0.5})
DARK, MID, LIGHT, ACC = "#2b2b2b", "#8c8c8c", "#d4d4d4", "#c8553d"
BLUE, BLUE_L, ORANGE = "#1f5aa6", "#a9c4e6", "#d9822b"


def panel(ax, s, x=-0.10, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")


d1 = json.load(open(FIGS / "figs_data.json", encoding="utf-8"))
d2 = json.load(open(FIGS / "figs_data_b32.json", encoding="utf-8"))

# ---- 그림 1: 민감도 지도
s = d1["F1"]["b32"]
blocks = ["only_patch"] + [f"only_b{i}" for i in range(12)] + ["only_head"]
blab = ["패치"] + [str(i) for i in range(12)] + ["헤드"]
ops = ["only_qkv", "only_attn_qk", "only_attn_av", "only_proj", "only_fc1", "only_fc2"]
olab = ["QKV", "QKᵀ", "AV", "출력\n투영", "MLP\nfc1", "MLP\nfc2"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.9), gridspec_kw={"width_ratios": [14, 7.5], "wspace": 0.12})
for ax, keys, labs in [(a1, blocks, blab), (a2, ops, olab)]:
    v = [s[k] for k in keys]
    cols = [ACC if x < 0.95 else LIGHT for x in v]
    ax.bar(range(len(keys)), v, 0.7, color=cols, edgecolor=DARK, linewidth=0.4)
    ax.axhline(0.95, color=DARK, lw=0.6, ls=(0, (3, 2)))
    ax.set_xticks(range(len(keys)), labs)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.tick_params(axis="x", length=0)
a1.set_ylabel("FP32 대비 코사인 유사도")
a2.set_yticklabels([])
a1.set_xlabel("INT8로 바꾼 블록 (나머지는 FP32)")
a2.set_xlabel("INT8로 바꾼 연산 유형")
a1.text(13.45, 0.965, "0.95", fontsize=7, ha="right", va="bottom")
panel(a1, "(a)", x=-0.075)
panel(a2, "(b)", x=-0.08)
fig.savefig(FIGS / "F1_paper.png")
plt.close(fig)

# ---- 그림 2: 레지스터 캐시·회전 2×2
t = d1["F2"]["b32"]
keys = ["원래", "회전만", "M2만", "회전+M2"]
labs = ["처방 없음", "회전만", "레지스터\n캐시만", "회전 +\n레지스터 캐시"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 1.85), gridspec_kw={"wspace": 0.28})
for ax, ylim in [(a1, (-68, 3)), (a2, (-2.55, 0.2))]:
    y = [t[k][0] for k in keys]
    lo = [t[k][0] - t[k][1][0] for k in keys]
    hi = [t[k][1][1] - t[k][0] for k in keys]
    ax.bar(range(4), y, 0.56, color=[LIGHT, LIGHT, BLUE_L, BLUE], edgecolor=DARK, linewidth=0.4)
    ax.errorbar(range(4), y, yerr=[lo, hi], fmt="none", ecolor=DARK, elinewidth=0.7, capsize=2.5)
    ax.axhline(0, color=DARK, lw=0.6)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.55, 3.75)
    ax.set_xticks(range(4), labs)
    ax.tick_params(axis="x", length=0)
    for i, k in enumerate(keys):
        yy = t[k][0]
        if ax is a1 and abs(yy) < 5:
            continue                      # 작은 값은 (b)에서 읽는다
        if yy < ylim[0]:
            ax.text(i, ylim[0] + 0.06, f"{yy:.1f}\n(축 밖)".replace("-", "−"), ha="center", va="bottom", fontsize=6.5)
        elif ax is a2:
            ax.text(i, t[k][1][0] - 0.06, f"{yy:.2f}".replace("-", "−"), ha="center", va="top", fontsize=7)
        else:
            ax.text(i + 0.31, yy * 0.5, f"{yy:.2f}".replace("-", "−"), ha="left", va="center", fontsize=7)
a1.set_ylabel("FP32 대비 Δtop-1 (%p)")
panel(a1, "(a)", x=-0.13)
panel(a2, "(b)", x=-0.13)
fig.savefig(FIGS / "F2_paper.png")
plt.close(fig)

# ---- 그림 3: FP32 대비 감소율 (가로 막대 = 대표값, 빈 점 = 같은 모델의 다른 실행)
f3 = d2["F3b"]["pct_vs_fp32"]
K = list(f3)
sub, sub2, prev, prev2 = (f3[k] for k in K)   # 제출 대표, 제출 기본 설정, 직전 대표 실행, 직전 다른 날
items = ["크기", "피크 메모리", "지연 1T", "지연 2T", "지연 4T"]
ilab = ["모델 크기", "피크 메모리", "지연 (1스레드)", "지연 (2스레드)", "지연 (4스레드)"]
fig, ax = plt.subplots(figsize=(W * 0.82, 2.2))
h = 0.36
ys = list(range(len(items)))[::-1]
for y, it in zip(ys, items):
    for yy, main, other, col, tc in [(y + h / 2, sub, sub2, BLUE, "#111111"), (y - h / 2, prev, prev2, BLUE_L, "#555555")]:
        ax.barh(yy, -main[it], h, color=col, edgecolor=DARK, linewidth=0.4)
        ax.plot(-other[it], yy, marker="o", ms=3.6, mfc="white", mec=MID, mew=1.0, ls="none")
        ax.text(max(-main[it], -other[it]) + 1.0, yy, f"{-main[it]:.1f}", va="center", fontsize=7, color=tc)
ax.set_yticks(ys, ilab)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 85)
ax.set_xlabel("FP32 대비 감소율 (%, 클수록 좋음)")
from matplotlib.patches import Patch            # noqa: E402
from matplotlib.lines import Line2D             # noqa: E402
ax.legend(handles=[Patch(fc=BLUE, ec=DARK, lw=0.4, label="제출 모델 (대표값)"), Patch(fc=BLUE_L, ec=DARK, lw=0.4, label="직전 모델(⑥까지)"),
                   Line2D([], [], marker="o", ms=3.6, mfc="white", mec=MID, mew=1.0, ls="none", label="같은 모델의 다른 실행")],
          loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, handlelength=1.2, columnspacing=1.2, borderaxespad=0.2)
fig.savefig(FIGS / "F3_paper.png")
plt.close(fig)

# ---- 그림 4: Hailo-10H 실칩 지연 vs dev 정확도 (공개 HEF 기준선으로 나눈 사분면)
pts = {p["hef"]: p for p in d2["F5"]["points"]}
z = pts["b32_Z"]
fig, ax = plt.subplots(figsize=(W * 0.7, 2.45))
ax.axvspan(12.8, z["b1_ms"], ymin=0, ymax=1, color=BLUE, alpha=0.0)
ax.fill_between([12.8, z["b1_ms"]], z["dev_dtop1"], -0.6, color=BLUE, alpha=0.07, lw=0)
ax.axvline(z["b1_ms"], color=MID, lw=0.6, ls=(0, (3, 2)))
ax.axhline(z["dev_dtop1"], color=MID, lw=0.6, ls=(0, (3, 2)))
ax.text(12.9, -0.72, "공개 HEF보다 빠르고 정확한 영역", fontsize=6.8, color=BLUE, va="top")
spec = [("b32_Z", "Hailo 공개 HEF", "s", MID, (0.12, 0.0, "left")),
        ("b32_GF", "우리 HEF (본선용)", "o", BLUE, (0.12, 0.0, "left")),
        ("b32_ALL8F", "잔차·패치 경로까지 8비트", "^", ORANGE, (0.12, 0.0, "left"))]
for k, name, m, c, (dx, dy, ha) in spec:
    p = pts[k]
    lo, hi = p["ci95"]
    ax.errorbar(p["b1_ms"], p["dev_dtop1"], yerr=[[p["dev_dtop1"] - lo], [hi - p["dev_dtop1"]]], fmt=m, ms=5.5,
                mfc=c, mec=DARK, mew=0.6, ecolor=c, elinewidth=0.9, capsize=2.5, zorder=3)
    ax.text(p["b1_ms"] + dx, p["dev_dtop1"] + dy, f"{name}\n{p['b1_ms']:.2f} ms, {p['dev_dtop1']:+.2f}%p".replace('-', '−'), ha=ha, va="center",
            fontsize=6.8, color="#222222", linespacing=1.15)
ax.set_xlim(12.8, 18.4)
ax.set_ylim(-5.2, -0.6)
ax.set_xlabel("배치 1 지연 (ms, 왼쪽이 빠름)")
ax.set_ylabel("FP32 대비 dev Δtop-1 (%p)")
fig.savefig(FIGS / "F4_paper.png")
plt.close(fig)
print("ok")
