import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib import font_manager as fm

for f in fm.findSystemFonts():
    if "NotoSansCJK" in f:
        fm.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": ["Noto Sans CJK KR", "sans-serif"], "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b",
    "axes.grid": True, "grid.color": "#e7e6e1", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
    "savefig.dpi": 160, "savefig.bbox": "tight",
})
BLUE, ORANGE, AQUA, YELLOW, MAG, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
OUT = sys.argv[2]
df = pd.read_csv(sys.argv[1])
ot = df["OT"].values


def box(ax, x, y, w, h, text, fc, ec=None, fs=11, tc=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec or fc, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, c=INK2, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=c, lw=lw))


def blank(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off"); ax.grid(False)
    return fig, ax

# ---------- 01 LTSF 문제 정의 ----------
s, L, T = 2000, 336, 96
fig, ax = plt.subplots(figsize=(11, 3.6))
x = np.arange(L + T)
ax.axvspan(0, L - 1, color=BLUE, alpha=0.07, lw=0)
ax.axvspan(L, L + T - 1, color=ORANGE, alpha=0.09, lw=0)
ax.plot(x[:L], ot[s:s + L], color=BLUE, lw=1.6)
ax.plot(x[L:], ot[s + L:s + L + T], color=ORANGE, lw=1.6)
lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + 6)
ax.text(L / 2, ax.get_ylim()[1] * 0.97, "입력 (look-back window)\nL = 336 시점 = 과거 14일", ha="center", va="top", color=INK2)
ax.text(L + T / 2, ax.get_ylim()[1] * 0.97, "예측 구간 (horizon)\nT = 96 = 다음 4일", ha="center", va="top", color=INK2)
ax.set_xlabel("시점 (1시간 간격)"); ax.set_ylabel("OT (변압기 오일 온도)")
ax.set_title("LTSF: 과거 L개를 보고 미래 T개를 한꺼번에 맞히기 (ETTh1, OT 채널)", loc="left", fontsize=13)
fig.savefig(f"{OUT}/01_ltsf_problem.png"); plt.close(fig)

# ---------- 02 IMS vs DMS ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax in axes:
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
ax = axes[0]
ax.set_title("IMS (반복 예측) — Transformer 계열 원래 방식", loc="left", fontsize=12)
for i in range(6):
    box(ax, 0.3 + i * 0.55, 3.6, 0.48, 0.7, "", BLUE)
for k in range(3):
    xx = 3.65 + k * 1.9
    box(ax, xx, 3.6, 0.7, 0.7, f"y{k+1}", ORANGE, tc="white", bold=True)
    arrow(ax, xx + 0.35, 3.5, xx + 0.35, 2.4, RED)
    ax.text(xx + 0.35, 2.1, "다시 입력으로", ha="center", fontsize=9, color=RED)
    if k < 2:
        arrow(ax, xx + 0.75, 3.95, xx + 1.85, 3.95)
ax.text(0.3, 4.6, "과거 입력", color=INK2, fontsize=10)
ax.text(0.3, 1.0, "한 칸 예측 → 예측값을 다시 넣고 → 또 한 칸 …\n앞에서 틀린 게 뒤로 누적 (error accumulation)", fontsize=10.5, color=INK)
ax = axes[1]
ax.set_title("DMS (직접 예측) — DLinear 방식", loc="left", fontsize=12)
for i in range(6):
    box(ax, 0.3 + i * 0.55, 3.6, 0.48, 0.7, "", BLUE)
box(ax, 3.9, 3.45, 1.5, 1.0, "모델\n한 번", "#e7e6e1", fs=10)
arrow(ax, 3.6, 3.95, 3.85, 3.95); arrow(ax, 5.45, 3.95, 5.75, 3.95)
for k in range(5):
    box(ax, 5.8 + k * 0.8, 3.6, 0.7, 0.7, f"y{k+1}", ORANGE, tc="white", bold=True, fs=10)
ax.text(0.3, 4.6, "과거 입력", color=INK2, fontsize=10)
ax.text(0.3, 1.0, "T개를 한 번에 출력 → 누적 오차 없음\n긴 horizon(T=720)일수록 유리", fontsize=10.5, color=INK)
fig.savefig(f"{OUT}/02_ims_vs_dms.png"); plt.close(fig)

# ---------- 03 시계열 분해 (실제 ETTh1) ----------
seg = ot[s:s + L].astype(float)
k = 25; p = (k - 1) // 2
pad = np.concatenate([np.repeat(seg[:1], p), seg, np.repeat(seg[-1:], p)])
trend = np.convolve(pad, np.ones(k) / k, mode="valid")
seas = seg - trend
fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
axes[0].plot(seg, color=INK2, lw=1.3); axes[0].plot(trend, color=BLUE, lw=2)
axes[0].set_title("① 원본 x (회색)  +  이동평균 trend (파랑)", loc="left")
axes[1].plot(trend, color=BLUE, lw=2); axes[1].set_title("② Trend = MovingAvg(x, kernel=25)  → 느린 흐름", loc="left")
axes[2].plot(seas, color=ORANGE, lw=1.3); axes[2].axhline(0, color=MUTED, lw=1)
axes[2].set_title("③ Seasonal(나머지) = x − Trend  → 하루 주기(24h) 같은 빠른 진동", loc="left")
axes[2].set_xlabel("입력 window 안의 시점 (0~335)")
fig.suptitle("DLinear의 핵심: 입력을 Trend + Seasonal 두 조각으로 나눈다 (ETTh1 OT 실제 데이터)", x=0.01, ha="left", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/03_decomposition.png"); plt.close(fig)

# ---------- 04 이동평균 + replicate padding ----------
fig, ax = blank((12, 4.3)); ax.set_xlim(-0.5, 12.5); ax.set_ylim(0, 5)
vals = [3, 5, 4, 6, 8, 7]
ax.text(-0.4, 4.55, "예: kernel_size = 5 → 양쪽에 (5−1)/2 = 2칸씩 패딩. 패딩값 = 맨 앞/맨 뒤 값 복사", fontsize=11, color=INK)
cells = [3, 3] + vals + [7, 7]
for i, v in enumerate(cells):
    padc = i < 2 or i >= 8
    box(ax, i * 1.05, 3.0, 0.9, 0.8, str(v), "#e7e6e1" if padc else BLUE, tc=INK if padc else "white", bold=True)
ax.text(0.95, 2.75, "front pad", ha="center", fontsize=9, color=INK2)
ax.text(9.35, 2.75, "end pad", ha="center", fontsize=9, color=INK2)
ax.add_patch(Rectangle((-0.08, 2.92), 5.25, 0.96, fill=False, ec=ORANGE, lw=2.2, ls="--"))
ax.text(2.55, 2.45, "첫 창: (3+3+3+5+4)/5 = 3.6", ha="center", fontsize=10, color=ORANGE)
pad = np.array(cells, float)
tr = [pad[i:i + 5].mean() for i in range(6)]
for i, v in enumerate(tr):
    box(ax, (i + 2) * 1.05, 1.1, 0.9, 0.8, f"{v:.1f}", AQUA, tc="white", bold=True)
ax.text(-0.4, 1.4, "trend\n(길이 그대로 6)", fontsize=10, color=INK2)
ax.text(-0.4, 3.25, "x", fontsize=12, color=INK2)
ax.text(-0.4, 0.3, "AvgPool1d(kernel=5, stride=1, padding=0) 을 패딩된 길이 10에 적용 → 출력 길이 10−5+1 = 6 = 원래 길이.  seasonal = x − trend",
        fontsize=10.5, color=INK)
fig.savefig(f"{OUT}/04_moving_avg_padding.png"); plt.close(fig)

# ---------- 05 DLinear 아키텍처 + shape ----------
fig, ax = blank((13, 5.4)); ax.set_xlim(0, 13); ax.set_ylim(0, 6)
box(ax, 0.2, 2.5, 1.9, 1.1, "입력 x\n[B, L, C]\n[32, 336, 7]", "#e7e6e1", fs=10)
box(ax, 2.6, 2.4, 2.0, 1.3, "series_decomp\nMovingAvg(k=25)", VIOLET, tc="white", fs=10.5, bold=True)
arrow(ax, 2.1, 3.05, 2.55, 3.05)
box(ax, 5.2, 4.1, 2.0, 1.0, "Trend\n[B, L, C]", BLUE, tc="white", fs=10)
box(ax, 5.2, 1.0, 2.0, 1.0, "Seasonal = x − Trend\n[B, L, C]", ORANGE, tc="white", fs=10)
arrow(ax, 4.6, 3.3, 5.15, 4.5); arrow(ax, 4.6, 2.8, 5.15, 1.5)
box(ax, 7.6, 4.1, 2.1, 1.0, "permute → [B, C, L]\nLinear_Trend\nL → T", "#d6e6f8", fs=9.5)
box(ax, 7.6, 1.0, 2.1, 1.0, "permute → [B, C, L]\nLinear_Seasonal\nL → T", "#fbe0d4", fs=9.5)
arrow(ax, 7.2, 4.6, 7.55, 4.6); arrow(ax, 7.2, 1.5, 7.55, 1.5)
ax.add_patch(plt.Circle((10.4, 3.05), 0.35, color=GREEN)); ax.text(10.4, 3.05, "+", ha="center", va="center", color="white", fontsize=18, fontweight="bold")
arrow(ax, 9.7, 4.5, 10.2, 3.35); arrow(ax, 9.7, 1.6, 10.2, 2.75)
box(ax, 11.0, 2.45, 1.9, 1.2, "예측 출력\npermute → [B, T, C]\n[32, 96, 7]", "#e7e6e1", fs=9.5)
arrow(ax, 10.75, 3.05, 10.95, 3.05)
ax.text(0.2, 0.25, "학습되는 건 Linear 2개뿐: 파라미터 = 2 × (L×T + T) = 2 × (336×96 + 96) = 64,704개 (individual=False일 때, 채널 7개가 같은 가중치를 공유)",
        fontsize=10.5, color=INK)
ax.text(0.2, 5.6, "DLinear forward — 텐서 shape 따라가기 (B=batch, L=seq_len, C=채널, T=pred_len)", fontsize=13)
fig.savefig(f"{OUT}/05_dlinear_architecture.png"); plt.close(fig)

# ---------- 06 Linear 층 = 가중합 ----------
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.axis("off"); ax.grid(False); ax.set_xlim(0, 10); ax.set_ylim(0, 8)
ax.set_title("출력 한 칸 = 입력 L칸 전체의 가중합", loc="left", fontsize=12)
xs = [0.6 + i * 0.9 for i in range(6)]
for i, xx in enumerate(xs):
    box(ax, xx - 0.35, 6.3, 0.7, 0.7, f"x{i+1}" if i < 5 else "x_L", BLUE, tc="white", fs=9)
box(ax, 4.2, 1.2, 1.2, 0.8, "출력 t", ORANGE, tc="white", bold=True)
for i, xx in enumerate(xs):
    ax.plot([xx, 4.8], [6.25, 2.05], color=MUTED, lw=0.6 + 1.6 * (i / 5))
ax.text(5.6, 4.0, "선 굵기 = 가중치 크기\n(예: 최근 시점일수록 크다)", fontsize=10, color=INK2)
ax.text(0.2, 0.2, "모든 미래 시점 t마다 이런 줄이 L개씩 → W는 T×L 행렬", fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{OUT}/06_linear_weighted_sum.png"); plt.close(fig)

# ---------- 07 채널 독립 / individual ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for ax, ind in zip(axes, [False, True]):
    ax.axis("off"); ax.grid(False); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    names = ["HUFL", "HULL", "MUFL", "…", "OT"]
    cols = [BLUE, AQUA, VIOLET, MUTED, ORANGE]
    for i, (n, c) in enumerate(zip(names, cols)):
        y = 4.8 - i * 1.05
        box(ax, 0.3, y, 1.8, 0.75, n, c, tc="white" if n != "…" else INK, fs=10)
        if n == "…":
            continue
        if ind:
            box(ax, 3.8, y, 2.6, 0.75, f"W_{n}  (L→T)", "#e7e6e1", fs=9.5)
            arrow(ax, 2.15, y + 0.37, 3.75, y + 0.37)
            arrow(ax, 6.45, y + 0.37, 7.6, y + 0.37)
        else:
            arrow(ax, 2.15, y + 0.37, 3.75, 2.9)
        box(ax, 7.65, y, 1.9, 0.75, f"pred {n}", c, tc="white", fs=10)
    if not ind:
        box(ax, 3.8, 2.3, 2.6, 1.3, "공유 W 하나\n(L→T)", "#e7e6e1", fs=11, bold=True)
        for i in [0, 1, 2, 4]:
            arrow(ax, 6.45, 2.95, 7.6, 4.8 - i * 1.05 + 0.37)
        ax.set_title("기본값 (individual=False): 모든 채널이 같은 W", loc="left", fontsize=12)
    else:
        ax.set_title("--individual: 채널마다 자기 W", loc="left", fontsize=12)
fig.text(0.01, -0.02, "두 경우 모두 채널끼리 섞지 않는다 (HUFL 예측에 OT 값은 쓰이지 않음) → 'channel independence'", fontsize=11)
fig.savefig(f"{OUT}/07_channel_independence.png"); plt.close(fig)

# ---------- 08 Linear / NLinear / DLinear ----------
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
specs = [
    ("Linear", ["x", "Linear(L→T)", "예측"], [BLUE, "#e7e6e1", ORANGE], "그냥 선형층 하나"),
    ("NLinear", ["x − x_L", "Linear(L→T)", "+ x_L → 예측"], [BLUE, "#e7e6e1", ORANGE], "마지막 값을 빼고 예측 후 다시 더함\n→ 분포 이동(distribution shift)에 강함"),
    ("DLinear", ["Trend / Seasonal", "Linear ×2", "합 → 예측"], [VIOLET, "#e7e6e1", ORANGE], "분해 후 각각 선형층\n→ 뚜렷한 추세가 있을 때 강함"),
]
for ax, (name, parts, cs, desc) in zip(axes, specs):
    ax.axis("off"); ax.grid(False); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.set_title(name, loc="left", fontsize=14, fontweight="bold")
    for i, (pt, c) in enumerate(zip(parts, cs)):
        box(ax, 0.2 + i * 3.35, 3.2, 2.9, 1.1, pt, c, tc="white" if c != "#e7e6e1" else INK, fs=10)
        if i < 2:
            arrow(ax, 3.12 + i * 3.35, 3.75, 3.52 + i * 3.35, 3.75)
    ax.text(0.2, 1.2, desc, fontsize=10.5, color=INK2)
fig.suptitle("LTSF-Linear 가족", x=0.01, ha="left", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/08_linear_family.png"); plt.close(fig)

# ---------- 09 데이터 분할 ----------
fig, axes = plt.subplots(2, 1, figsize=(12, 4.0))
for ax, (title, parts) in zip(axes, [
    ("ETTh1/ETTh2/ETTm1/ETTm2 : 앞 20개월만 사용 — 12 : 4 : 4 개월 (뒤 4개월치는 버림)", [("train 12개월", 12, BLUE), ("val 4개월", 4, AQUA), ("test 4개월", 4, ORANGE), ("미사용", 4, "#e7e6e1")]),
    ("custom (Exchange, Weather, ECL, Traffic, ILI) : 전체를 7 : 1 : 2 비율", [("train 70%", 7, BLUE), ("val 10%", 1, AQUA), ("test 20%", 2, ORANGE)]),
]):
    ax.axis("off"); ax.grid(False)
    tot = sum(p[1] for p in parts); x0 = 0
    for lab, w, c in parts:
        ww = w / tot * 10
        ax.add_patch(Rectangle((x0, 0.2), ww - 0.05, 0.55, color=c))
        ax.text(x0 + ww / 2, 0.47, lab, ha="center", va="center", color="white" if c != "#e7e6e1" else INK2, fontsize=10)
        x0 += ww
    ax.set_xlim(0, 10); ax.set_ylim(0, 1.1); ax.set_title(title, loc="left", fontsize=11)
fig.text(0.01, -0.04, "val/test 시작점은 seq_len만큼 앞으로 당겨진다 (border1 = 경계 − seq_len) → 첫 test 샘플도 입력 336개를 온전히 가짐.\nStandardScaler는 train 구간으로만 fit → 정보 누설 없음. 보고되는 MSE/MAE는 이 정규화된 스케일 기준!", fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{OUT}/09_data_split.png"); plt.close(fig)

# ---------- 10 슬라이딩 윈도우 ----------
fig, ax = blank((12, 4.0)); ax.set_xlim(0, 12); ax.set_ylim(0, 5)
ax.add_patch(Rectangle((0.3, 3.7), 11.4, 0.5, color="#e7e6e1"))
ax.text(0.3, 4.4, "data_x (정규화된 연속 시계열)", fontsize=10.5, color=INK2)
for k, yy in enumerate([2.6, 1.6]):
    off = 0.3 + k * 0.6
    ax.add_patch(Rectangle((off, yy), 5.6, 0.5, color=BLUE))
    ax.text(off + 2.8, yy + 0.25, f"seq_x  (index={k}부터 336개)", ha="center", va="center", color="white", fontsize=10)
    ax.add_patch(Rectangle((off + 5.6, yy), 1.6, 0.5, color=ORANGE))
    ax.text(off + 6.4, yy + 0.25, "정답 96", ha="center", va="center", color="white", fontsize=9.5)
    ax.add_patch(Rectangle((off + 4.8, yy - 0.12), 0.8, 0.12, color=MUTED))
ax.text(8.5, 2.05, "회색 = label_len(48) 구간: Transformer 디코더용.\n   Linear 계열은 forward(batch_x)만 쓰고\n   batch_y[:, -pred_len:] 만 정답으로 씀", fontsize=10, color=INK2)
ax.text(0.3, 0.5, "한 칸씩 밀면서 샘플을 만든다 → 샘플 개수 = len(data) − seq_len − pred_len + 1   (ETTh1 train: 8640 − 336 − 96 + 1 = 8209)", fontsize=10.5)
fig.savefig(f"{OUT}/10_sliding_window.png"); plt.close(fig)

# ---------- 11 코드 실행 흐름 ----------
fig, ax = blank((13, 5.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 7)
steps = [
    (0.2, 5.6, "scripts/…/etth1.sh", "하이퍼파라미터 지정\npython run_longExp.py …"),
    (3.4, 5.6, "run_longExp.py", "argparse, seed=2021\nitr번 반복 실행"),
    (6.6, 5.6, "exp/exp_main.py", "Exp_Main\n_build_model / train / test"),
    (9.8, 5.6, "models/DLinear.py", "Model(configs)\nforward(x)"),
    (6.6, 2.9, "data_provider/", "data_factory → Dataset_ETT_hour\n분할·정규화·윈도우"),
    (3.4, 2.9, "utils/tools.py", "EarlyStopping(patience=3)\nadjust_learning_rate(type1)"),
    (0.2, 2.9, "utils/metrics.py", "MSE / MAE / RSE / CORR"),
    (9.8, 2.9, "출력물", "checkpoints/…/checkpoint.pth\nresult.txt · logs/*.log\ntest_results/*.pdf"),
]
for x0, y0, t1, t2 in steps:
    box(ax, x0, y0, 3.0, 1.2, "", "#eef3fb", ec=BLUE)
    ax.text(x0 + 1.5, y0 + 0.88, t1, ha="center", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.text(x0 + 1.5, y0 + 0.38, t2, ha="center", va="center", fontsize=9, color=INK)
arrow(ax, 3.2, 6.2, 3.38, 6.2); arrow(ax, 6.4, 6.2, 6.58, 6.2); arrow(ax, 9.6, 6.2, 9.78, 6.2)
arrow(ax, 8.1, 5.55, 8.1, 4.15); arrow(ax, 7.2, 5.55, 4.9, 4.15); arrow(ax, 7.0, 5.55, 1.7, 4.15); arrow(ax, 9.0, 5.55, 11.3, 4.15)
ax.text(0.2, 1.4, "train(): 매 epoch마다 train → vali loss, test loss 출력 → vali loss로 early stopping → best 체크포인트 재로딩\n"
        "test(): test 셋 전체 예측 → mse, mae 출력 + result.txt에 한 줄 추가", fontsize=10.5)
ax.text(0.2, 6.85, "코드 실행 흐름", fontsize=13)
fig.savefig(f"{OUT}/11_code_flow.png"); plt.close(fig)

# ---------- 12 learning rate 스케줄 ----------
fig, ax = plt.subplots(figsize=(8, 3.4))
ep = np.arange(1, 11); lr0 = 0.005
lr = lr0 * 0.5 ** (ep - 1)
ax.plot(ep, lr, color=BLUE, lw=2, marker="o", ms=8, mec=SURF, mew=2)
ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("learning rate (log)")
ax.set_title("lradj='type1': 매 epoch마다 학습률 절반 (ETTh1, 시작 0.005)", loc="left", fontsize=12)
ax.annotate("epoch 1: 0.005", (1, lr[0]), xytext=(2.2, lr[0]), color=INK2, fontsize=10)
ax.annotate("epoch 10: ≈ 0.0000098", (10, lr[-1]), xytext=(6.2, lr[-1] * 0.9), color=INK2, fontsize=10)
fig.savefig(f"{OUT}/12_lr_schedule.png"); plt.close(fig)

# ---------- 13 논문 결과 비교 (Table 2, T=96/720 MSE) ----------
ds = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Exchange", "Weather", "Electricity", "Traffic"]
dl96 = [0.375, 0.289, 0.299, 0.167, 0.081, 0.176, 0.140, 0.410]
fe96 = [0.376, 0.346, 0.379, 0.203, 0.148, 0.217, 0.193, 0.587]
au96 = [0.449, 0.358, 0.505, 0.255, 0.197, 0.266, 0.201, 0.613]
dl720 = [0.472, 0.605, 0.425, 0.397, 0.643, 0.323, 0.203, 0.466]
fe720 = [0.506, 0.463, 0.543, 0.421, 1.195, 0.403, 0.246, 0.626]
au720 = [0.514, 0.515, 0.671, 0.433, 1.447, 0.419, 0.254, 0.660]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=False)
for ax, (h, a, b, c) in zip(axes, [(96, dl96, fe96, au96), (720, dl720, fe720, au720)]):
    xi = np.arange(len(ds)); w = 0.26
    for j, (vals, name, col) in enumerate([(a, "DLinear", BLUE), (b, "FEDformer", ORANGE), (c, "Autoformer", AQUA)]):
        ax.bar(xi + (j - 1) * w, vals, w - 0.03, color=col, label=name)
    ax.set_xticks(xi); ax.set_xticklabels(ds, rotation=25)
    ax.set_title(f"MSE (낮을수록 좋음), horizon T = {h}", loc="left", fontsize=12)
    ax.grid(axis="x", visible=False)
axes[0].legend(frameon=False, loc="upper left")
fig.suptitle("논문 Table 2: DLinear vs Transformer 계열 (다변량)", x=0.01, ha="left", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/13_paper_results.png"); plt.close(fig)

# ---------- 14 Transformer 순서 불변성 ----------
fig, ax = blank((12, 3.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 4.5)
a = [3, 5, 4, 8]; bshuf = [8, 3, 4, 5]
for i, v in enumerate(a):
    box(ax, 0.3 + i * 0.9, 3.0, 0.8, 0.7, str(v), BLUE, tc="white", bold=True)
for i, v in enumerate(bshuf):
    box(ax, 0.3 + i * 0.9, 1.4, 0.8, 0.7, str(v), VIOLET, tc="white", bold=True)
ax.text(4.1, 3.3, "원래 순서", fontsize=10, color=INK2); ax.text(4.1, 1.7, "섞은 순서", fontsize=10, color=INK2)
box(ax, 5.6, 1.9, 2.4, 1.4, "Self-Attention\n(위치 정보 X)", "#e7e6e1", fs=10.5)
arrow(ax, 5.0, 3.3, 5.55, 2.9); arrow(ax, 5.0, 1.75, 5.55, 2.3)
arrow(ax, 8.05, 2.6, 8.6, 2.6)
ax.text(8.7, 2.25, "같은 값들의 '집합'으로 봄\n→ 순서를 섞어도 결과가 거의 같다\n(positional encoding으로 일부만 보완)", fontsize=10.5)
ax.text(0.3, 0.3, "논문의 실험: 입력 순서를 섞으면 Linear는 크게 나빠지는데 Transformer는 거의 그대로 → '시간 순서를 제대로 안 쓰고 있다'는 증거", fontsize=10.5)
fig.savefig(f"{OUT}/14_permutation_invariance.png"); plt.close(fig)
print("done")
