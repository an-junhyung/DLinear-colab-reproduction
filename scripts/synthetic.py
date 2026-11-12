"""Synthetic experiments that illustrate two mechanisms seen in the reproduction.

A. Level shift: the test period sits at a different level than the training period.
   Linear (1/L init, as in the June 2022 code) vs NLinear, trained with Adam;
   test MSE is logged every epoch, and the closed-form least-squares fit is the end point.
B. Look-back: a series with a weekday/weekend pattern (period 168 h).
   Closed-form Linear for several look-back lengths L, and its weight matrix at L = 336.

Data handling follows the repository's custom loader: 70/10/20 split in time,
standardised with the training mean and std, windows of L inputs and T outputs.
Outputs go to results/synthetic/.
"""
import os
import numpy as np
import pandas as pd
import torch

OUT = os.path.join(os.path.dirname(__file__), '..', 'results', 'synthetic')
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(1)


def ar1(n, phi, sigma, rng):
    e = np.zeros(n)
    z = rng.normal(0, sigma, n)
    for t in range(1, n):
        e[t] = phi * e[t - 1] + z[t]
    return e


def split(x, L):
    n = len(x)
    a, b = int(n * 0.7), int(n * 0.8)
    mu, sd = x[:a].mean(), x[:a].std()
    z = (x - mu) / sd
    return z[:a], z[a - L:b], z[b - L:], (mu, sd)


def windows(z, L, T):
    idx = np.arange(len(z) - L - T + 1)
    X = np.stack([z[i:i + L] for i in idx])
    Y = np.stack([z[i + L:i + L + T] for i in idx])
    return X, Y


def ols(X, Y, lam=1e-3):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    B = np.linalg.solve(A, Xb.T @ Y)
    return B[:-1], B[-1]  # W (L x T), b (T)


def mse(P, Y):
    return float(((P - Y) ** 2).mean())


# ---------------------------------------------------------------- A. level shift
def make_shift(seed, n, d):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # level: mean-reverting around +d/2 during training/validation, jumps to -d/2 for the test period
    lvl = np.zeros(n)
    target = np.where(t < int(n * 0.8), d / 2, -d / 2)
    lvl[0] = d / 2
    for i in range(1, n):
        lvl[i] = lvl[i - 1] + 0.01 * (target[i] - lvl[i - 1]) + rng.normal(0, 0.05)
    x = lvl + np.sin(2 * np.pi * t / 24) + ar1(n, 0.9, 0.3, rng)
    return t, x, lvl


def closed_pair(x, L, T):
    tr, va, te = [windows(z, L, T) for z in split(x, L)[:3]]
    W, b = ols(*tr)
    lin = mse(te[0] @ W + b, te[1])
    W, b = ols(tr[0] - tr[0][:, -1:], tr[1] - tr[0][:, -1:])
    nl = mse((te[0] - te[0][:, -1:]) @ W + b + te[0][:, -1:], te[1])
    return lin, nl


def shift_scan(n=17420, L=336, T=96, ds=(0, 1, 2, 3, 4, 6)):
    rows = []
    for d in ds:
        for seed in range(3):
            t, x, lvl = make_shift(seed, n, d)
            lin, nl = closed_pair(x, L, T)
            rows.append(dict(jump=d, seed=seed, linear_mse=lin, nlinear_mse=nl))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'level_shift_scan.csv'), index=False)
    return df.groupby('jump')[['linear_mse', 'nlinear_mse']].mean()


def level_shift(seed=0, n=17420, L=336, T=96, epochs=30, lr=0.005, bs=32, d=3.0):
    t, x, lvl = make_shift(seed, n, d)
    tr, va, te = [windows(z, L, T) for z in split(x, L)[:3]]

    rows = []
    # closed form end points
    for name, sub in [('Linear', False), ('NLinear', True)]:
        Xtr, Ytr = tr
        Xte, Yte = te
        if sub:
            W, b = ols(Xtr - Xtr[:, -1:], Ytr - Xtr[:, -1:])
            P = (Xte - Xte[:, -1:]) @ W + b + Xte[:, -1:]
        else:
            W, b = ols(Xtr, Ytr)
            P = Xte @ W + b
        rows.append(dict(model=name, epoch=0, kind='closed_form', vali_mse=np.nan, test_mse=mse(P, Yte)))

    ex = {}
    for name, sub in [('Linear', False), ('NLinear', True)]:
        torch.manual_seed(2021)
        lin = torch.nn.Linear(L, T)
        with torch.no_grad():
            lin.weight.copy_(torch.full((T, L), 1.0 / L))  # 1/L init as in the June 2022 code
        opt = torch.optim.Adam(lin.parameters(), lr=lr)

        def fwd(X):
            X = torch.as_tensor(X, dtype=torch.float32)
            if sub:
                last = X[:, -1:]
                return lin(X - last) + last
            return lin(X)

        def ev(d):
            with torch.no_grad():
                return mse(fwd(d[0]).numpy(), d[1])

        rows.append(dict(model=name, epoch=0, kind='trained', vali_mse=ev(va), test_mse=ev(te)))
        g = torch.Generator().manual_seed(2021)
        Xtr = torch.as_tensor(tr[0], dtype=torch.float32)
        Ytr = torch.as_tensor(tr[1], dtype=torch.float32)
        for ep in range(1, epochs + 1):
            for gp in opt.param_groups:  # type1 schedule: halve every epoch after the first
                gp['lr'] = lr * (0.5 ** (ep - 1))
            perm = torch.randperm(len(Xtr), generator=g)
            for i in range(0, len(perm), bs):
                j = perm[i:i + bs]
                loss = ((fwd(Xtr[j]) - Ytr[j]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            rows.append(dict(model=name, epoch=ep, kind='trained', vali_mse=ev(va), test_mse=ev(te)))
        with torch.no_grad():  # one test window, kept for a figure
            k = len(te[0]) // 2
            ex[name] = fwd(te[0][k:k + 1]).numpy()[0]
            ex['input'], ex['truth'] = te[0][k], te[1][k]
    df = pd.DataFrame(rows)
    pd.DataFrame({k: pd.Series(v) for k, v in ex.items()}).to_csv(os.path.join(OUT, 'level_shift_example.csv'), index=False)
    df.to_csv(os.path.join(OUT, 'level_shift_epochs.csv'), index=False)
    lv = pd.DataFrame({'t': t, 'x': x, 'level': lvl})
    lv.iloc[::4].to_csv(os.path.join(OUT, 'level_shift_series.csv'), index=False)
    return df


# ---------------------------------------------------------------- B. look-back
def lookback(seed=0, n=17420, T=96, Ls=(24, 48, 96, 144, 168, 192, 336, 504, 720)):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    day = (t // 24) % 7
    weekday = day < 5
    x = np.where(weekday, 1.0, 0.3) * np.sin(2 * np.pi * t / 24) + np.where(weekday, 0.8, 0.0) + ar1(n, 0.9, 0.3, rng)
    rows = []
    W336 = None
    for L in Ls:
        tr, va, te = [windows(z, L, T) for z in split(x, L)[:3]]
        W, b = ols(*tr)
        rows.append(dict(seq_len=L, pred_len=T, test_mse=mse(te[0] @ W + b, te[1])))
        if L == 336:
            W336 = W
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'lookback_weekly.csv'), index=False)
    np.save(os.path.join(OUT, 'lookback_weekly_W336.npy'), W336.T)  # (T, L)
    pd.DataFrame({'t': t[:24 * 21], 'x': x[:24 * 21]}).to_csv(os.path.join(OUT, 'lookback_weekly_series.csv'), index=False)
    return df


if __name__ == '__main__':
    print(lookback().to_string(index=False))
    print(shift_scan().to_string())
    df = level_shift()
    print(df.to_string(index=False))
