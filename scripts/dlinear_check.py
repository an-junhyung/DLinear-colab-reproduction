"""공식 LTSF-Linear 설정을 그대로 따라 한 독립 DLinear 검증 스크립트 (ETTh1, M, L=336)."""
import sys, random, numpy as np, pandas as pd, torch, torch.nn as nn

csv, pred_len = sys.argv[1], int(sys.argv[2])
seq_len, bs, lr, epochs, patience = 336, 32, 0.005, 10, 3
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 2021
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

df = pd.read_csv(csv).iloc[:, 1:].values.astype(np.float32)
b1 = [0, 8640 - seq_len, 8640 + 2880 - seq_len]
b2 = [8640, 8640 + 2880, 8640 + 5760]
mean, std = df[:8640].mean(0), df[:8640].std(0)
data = (df - mean) / std

class DS(torch.utils.data.Dataset):
    def __init__(self, k):
        self.x = data[b1[k]:b2[k]]
    def __len__(self):
        return len(self.x) - seq_len - pred_len + 1
    def __getitem__(self, i):
        return self.x[i:i + seq_len], self.x[i + seq_len:i + seq_len + pred_len]

class DLinear(nn.Module):
    def __init__(self, k=25):
        super().__init__()
        self.k = k
        self.avg = nn.AvgPool1d(k, stride=1)
        self.ls = nn.Linear(seq_len, pred_len)
        self.lt = nn.Linear(seq_len, pred_len)
    def forward(self, x):                       # x: [B, L, C]
        p = (self.k - 1) // 2
        xp = torch.cat([x[:, :1].repeat(1, p, 1), x, x[:, -1:].repeat(1, p, 1)], 1)
        trend = self.avg(xp.permute(0, 2, 1)).permute(0, 2, 1)
        seas = x - trend
        out = self.ls(seas.permute(0, 2, 1)) + self.lt(trend.permute(0, 2, 1))
        return out.permute(0, 2, 1)             # [B, T, C]

L = lambda k, sh, dl: torch.utils.data.DataLoader(DS(k), batch_size=bs, shuffle=sh, drop_last=dl)
tr, va, te = L(0, True, True), L(1, False, False), L(2, False, False)
m = DLinear()
VIZ = len(sys.argv) > 4
if VIZ:
    with torch.no_grad():
        m.ls.weight.fill_(1 / seq_len); m.lt.weight.fill_(1 / seq_len)
opt = torch.optim.Adam(m.parameters(), lr=lr); crit = nn.MSELoss()

def evaluate(loader):
    m.eval(); P, T = [], []
    with torch.no_grad():
        for x, y in loader:
            P.append(m(x).numpy()); T.append(y.numpy())
    m.train(); P, T = np.concatenate(P), np.concatenate(T)
    return ((P - T) ** 2).mean(), np.abs(P - T).mean()

best, cnt, state = np.inf, 0, None
for ep in range(epochs):
    for x, y in tr:
        opt.zero_grad(); loss = crit(m(x), y); loss.backward(); opt.step()
    v, _ = evaluate(va)
    print(f"epoch {ep+1} vali {v:.4f}", flush=True)
    if v < best:
        best, cnt, state = v, 0, {k: t.clone() for k, t in m.state_dict().items()}
    else:
        cnt += 1
        if cnt >= patience:
            break
    for g in opt.param_groups:
        g['lr'] = lr * 0.5 ** ep          # type1: epoch(1-based)+1 → lr * 0.5^(epoch)
m.load_state_dict(state)
mse, mae = evaluate(te)
if VIZ:
    np.save("w_seasonal.npy", m.ls.weight.detach().numpy()); np.save("w_trend.npy", m.lt.weight.detach().numpy())
print(f"RESULT ETTh1 pred_len={pred_len} mse={mse:.4f} mae={mae:.4f}")
