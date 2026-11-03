import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in fm.findSystemFonts():
    if "NotoSansCJK" in f: fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Noto Sans CJK KR", "axes.unicode_minus": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "text.color": "#0b0b0b", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "savefig.dpi": 160, "savefig.bbox": "tight"})
d = sys.argv[1]
ws, wt = np.load(f"{d}/w_seasonal.npy"), np.load(f"{d}/w_trend.npy")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.1]})
for ax, W, name in [(axes[0], ws, "Seasonal 가중치"), (axes[1], wt, "Trend 가중치")]:
    v = np.abs(W).max()
    im = ax.imshow(W, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax.set_title(f"{name}  (96 × 336)", loc="left"); ax.set_xlabel("입력 시점 (0=가장 과거, 335=가장 최근)"); ax.set_ylabel("출력 시점 (미래 1~96)")
    fig.colorbar(im, ax=ax, fraction=0.04)
ax = axes[2]
ax.plot(np.arange(336), ws[0], color="#2a78d6", lw=1.4, label="Seasonal, 출력 1번째 행")
ax.plot(np.arange(336), wt[0], color="#eb6834", lw=1.4, label="Trend, 출력 1번째 행")
for k in range(336 - 24, 0, -24):
    ax.axvline(k + 23, color="#c9c8c2", lw=0.6, zorder=0)
ax.set_title("한 줄만 꺼내보기 (회색 세로선 = 24시간 간격)", loc="left"); ax.set_xlabel("입력 시점"); ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", color="#e7e6e1"); [ax.spines[s].set_visible(False) for s in ["top", "right"]]
fig.suptitle("직접 학습한 DLinear 가중치 (ETTh1, L=336 → T=96, 1/L로 초기화)", x=0.01, ha="left", fontsize=13)
fig.tight_layout(); fig.savefig(f"{sys.argv[2]}/15_learned_weights.png")
