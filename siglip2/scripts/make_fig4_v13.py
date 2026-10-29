"""보고서 v13 그림 4 (부록 B): Hailo-10H 실칩 B/32 배치 1 지연 vs dev 정확도 손실.
make_report_figs_b32.py가 남긴 figs_data_b32.json의 F5 값을 그대로 쓴다 (새 측정 없음).
v12와 다른 점: 라벨을 부록 B 표 이름과 맞추고, 본선 HEF와 출력이 같은 G 점은 뺐다.
사용: python siglip2/scripts/make_fig4_v13.py  → siglip2/docs/figs/F5_hailo_b32_v13.png
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
OURS, MUTED = "#eb6834", "#898781"
NAMES = {"b32_Z": "Hailo Model Zoo 공개 HEF", "b32_GF": "우리 HEF (본선용)", "b32_ALL8F": "잔차·패치 경로까지 8비트"}

pts = {p["hef"]: p for p in json.load(open(FIGS / "figs_data_b32.json", encoding="utf-8"))["F5"]["points"]}
assert set(NAMES) <= set(pts), pts.keys()
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for k, name in NAMES.items():
    p = pts[k]
    c = MUTED if k == "b32_Z" else OURS
    lo, hi = p["ci95"]
    ax.errorbar(p["b1_ms"], p["dev_dtop1"], yerr=[[p["dev_dtop1"] - lo], [hi - p["dev_dtop1"]]], fmt="o", color=c, capsize=2, ms=5)
    ax.annotate(name, (p["b1_ms"], p["dev_dtop1"]), fontsize=6.5, xytext=(5, 3), textcoords="offset points", color=c)
ax.set_xlim(13.0, 18.6)
ax.set_xlabel("배치 1 지연 (ms, 같은 세션)")
ax.set_ylabel("dev 정확도 차이 (FP32 대비, %p)")
ax.set_title("Hailo-10H 실칩, SigLIP2 B/32 (왼쪽 위가 좋음)", fontsize=8.5)
fig.tight_layout()
fig.savefig(FIGS / "F5_hailo_b32_v13.png")
print("ok", FIGS / "F5_hailo_b32_v13.png")
