"""보고서 v13 후보 그림: FP32 대비 속도 (지연 ms와 처리량 FPS를 나란히). 새 측정 없음.

- 값: 보고서 6.5 "효율·정확도 요약" 표의 세션 중앙값 그대로 (i7-1165G7, FP32와 같은 실행에서 번갈아 측정)
  지연 ms 1/2/4스레드: FP32 126.0 / 96.7 / 68.6, 제출 39.2 / 35.2 / 24.7, 변화율(세션별 비율의 중앙값) −68.0 / −62.9 / −62.9
  처리량 FPS 1/2/4스레드: FP32 7.9 / 10.3 / 14.6, 제출 25.5 / 28.4 / 40.5 (+222 / +175 / +178% = ×3.22 / ×2.75 / ×2.78)
- 원자료 확인: results/siglip2/G8_candidates/latency_this_pc/b32_p2_c3{,b}_20261027/cross_latency.json 6세션에서 다시 계산해 모두 같음
- 형식은 make_figs_paper_v13.py와 같다 (본문 폭 6.25 in, 8 pt, 우리 = 파랑, 기준 = 회색).
사용: python siglip2/scripts/make_fig_speed_v13.py → siglip2/docs/figs/F3s_speed_paper.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

FIGS = Path(__file__).resolve().parents[1] / "docs" / "figs"
W = 6.25
plt.rcParams.update({
    "font.family": ["Malgun Gothic", "Noto Sans CJK KR", "AppleGothic", "NanumGothic", "DejaVu Sans"],
    "axes.unicode_minus": True, "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.edgecolor": "#222222", "axes.labelcolor": "#222222", "xtick.color": "#222222", "ytick.color": "#222222",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "legend.frameon": False})
DARK, LIGHT, BLUE = "#2b2b2b", "#d4d4d4", "#1f5aa6"

TH = ["1스레드", "2스레드", "4스레드"]
LAT_FP32, LAT_SUB, LAT_PCT = [126.0, 96.7, 68.6], [39.2, 35.2, 24.7], [-68.0, -62.9, -62.9]
FPS_FP32, FPS_SUB, FPS_PCT = [7.9, 10.3, 14.6], [25.5, 28.4, 40.5], [222, 175, 178]
# 배수 = 세션 중앙값 ms끼리 나눈 값(원자료에서 계산: 3.216 / 2.748 / 2.779). 표의 +222 / +175 / +178%와 같은 값
MULT = [3.22, 2.75, 2.78]


def panel(ax, s, x=-0.12, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")


fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.15), gridspec_kw={"wspace": 0.26})
x = range(3)
w = 0.36
for ax, f, s, ylim in [(a1, LAT_FP32, LAT_SUB, 158), (a2, FPS_FP32, FPS_SUB, 47)]:
    ax.bar([i - w / 2 for i in x], f, w, color=LIGHT, edgecolor=DARK, linewidth=0.4)
    ax.bar([i + w / 2 for i in x], s, w, color=BLUE, edgecolor=DARK, linewidth=0.4)
    ax.set_xticks(list(x), TH)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, ylim)
    for i in x:
        ax.text(i - w / 2, f[i] + ylim * 0.012, f"{f[i]:.1f}", ha="center", va="bottom", fontsize=6.8, color="#444444")
        ax.text(i + w / 2, s[i] + ylim * 0.012, f"{s[i]:.1f}", ha="center", va="bottom", fontsize=6.8, color="#111111")

# (a) 지연: 막대 위 감소율
for i in x:
    top = LAT_FP32[i] + 158 * 0.075
    a1.text(i, top, f"{LAT_PCT[i]:.1f}%".replace("-", "−"), ha="center", va="bottom", fontsize=7.5,
            color=BLUE, fontweight="bold")
a1.set_ylabel("배치 1 지연 (ms, 낮을수록 좋음)")
# (b) 처리량: 막대 위 배수
for i in x:
    mult = MULT[i]
    a2.text(i + w / 2, FPS_SUB[i] + 47 * 0.075, f"×{mult:.2f}", ha="center", va="bottom", fontsize=7.5,
            color=BLUE, fontweight="bold")
a2.set_ylabel("처리량 (FPS, 높을수록 좋음)")
panel(a1, "(a)", x=-0.14)
panel(a2, "(b)", x=-0.13)
fig.legend(handles=[Patch(fc=LIGHT, ec=DARK, lw=0.4, label="원본 FP32"),
                    Patch(fc=BLUE, ec=DARK, lw=0.4, label="제출 모델 (INT8)")],
           loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=2, handlelength=1.2, columnspacing=1.5)
fig.savefig(FIGS / "F3s_speed_paper.png")
print("ok")
