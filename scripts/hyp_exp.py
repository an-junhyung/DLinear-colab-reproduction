"""One run of the improvement-hypothesis experiments, on an unmodified LTSF-Linear checkout.

Usage:
  python hyp_exp.py <repo_dir> key=value ... -- <the official run_longExp.py arguments>

Keys
  mode     train | ens | ridge
  seed     replaces the repository's hard-coded fix_seed = 2021           (train)
  variant  base | sublast     sublast = subtract the last input value before the model and
                              add it back after (H1 for DLinear; for ridge, NLinear-style)
  keep     file to copy the best checkpoint to                            (train)
  ckpts    comma-separated checkpoints whose weights are averaged         (ens, H3)

What is changed and what is not
  - run_longExp.py is executed as it is, except for the seed line and, for ens, --is_training 0.
  - Training, early stopping, learning-rate schedule and data loading are the repository's.
  - The test step is replaced by one that adds up the errors batch by batch in float64. It gives
    the repository's numbers (same loader, same drop_last) without holding all predictions in memory.
  - ens: for a model that is linear in its weights, averaging weights equals averaging predictions.
  - ridge: closed-form ridge regression with an intercept, one weight shared by all variables,
    on the same scaled splits. The penalty is lam * (number of training windows) * ||W||^2, the
    intercept is not penalised, and lam is chosen on the validation split. Test error is reported
    on the same windows the repository's test loader keeps (drop_last), and on all windows.

Prints
  train/ens:  the repository's log, ending with   mse:<mse>, mae:<mae>
  ridge:      RIDGE,<lam>,<val_mse>,<test_mse>,<test_mae>,<test_mse_all>   per lam
              BEST,<lam>,<val_mse>,<test_mse>,<test_mae>,<test_mse_all>
"""
import glob
import os
import shutil
import sys

repo = sys.argv[1]
cut = sys.argv.index('--')
kv = dict(a.split('=', 1) for a in sys.argv[2:cut])
OFF = sys.argv[cut + 1:]
MODE = kv.get('mode', 'train')
SEED = int(kv.get('seed', 2021))
VAR = kv.get('variant', 'base')
CKPTS = [c for c in kv.get('ckpts', '').split(',') if c]
LAMS = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]

os.chdir(repo)
sys.path.insert(0, repo)

import numpy as np
import torch

if not hasattr(np, 'Inf'):  # NumPy 2 removed np.Inf, which the repository still uses
    np.Inf = np.inf

if VAR == 'sublast' and MODE != 'ridge':
    import models.DLinear as D
    _forward = D.Model.forward

    def forward(self, x):
        last = x[:, -1:, :].detach()
        return _forward(self, x - last) + last
    D.Model.forward = forward

import exp.exp_main as em


def test(self, setting, test=0):
    _, loader = self._get_data(flag='test')
    if CKPTS:
        states = [torch.load(c, map_location=self.device) for c in CKPTS]
        self.model.load_state_dict({k: sum(s[k] for s in states) / len(states) for k in states[0]})
    elif test:
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    P = self.args.pred_len
    f = -1 if self.args.features == 'MS' else 0
    se = ae = n = 0
    self.model.eval()
    with torch.no_grad():
        for bx, by, _, _ in loader:
            out = self.model(bx.float().to(self.device))
            d = (out[:, -P:, f:] - by.float().to(self.device)[:, -P:, f:]).double()
            se += float((d ** 2).sum())
            ae += float(d.abs().sum())
            n += d.numel()
    print('mse:{}, mae:{}'.format(se / n, ae / n))


em.Exp_Main.test = test

src = open('run_longExp.py').read()
assert 'fix_seed = 2021' in src
src = src.replace('fix_seed = 2021', 'fix_seed = %d' % SEED)

if MODE in ('train', 'ens'):
    if MODE == 'ens':
        OFF[OFF.index('--is_training') + 1] = '0'
    sys.argv = ['run_longExp.py'] + OFF
    exec(compile(src, 'run_longExp.py', 'exec'), {'__name__': '__main__'})
    mid = OFF[OFF.index('--model_id') + 1]
    for d in glob.glob('checkpoints/%s_*' % mid):
        if MODE == 'train' and 'keep' in kv:
            os.makedirs(os.path.dirname(kv['keep']), exist_ok=True)
            shutil.copy(os.path.join(d, 'checkpoint.pth'), kv['keep'])
        shutil.rmtree(d)
    sys.exit(0)

# ---------------------------------------------------------------- ridge (H2)
sys.argv = ['run_longExp.py'] + OFF
ns = {'__name__': 'head'}
exec(compile(src.split('Exp = Exp_Main')[0], 'run_longExp.py', 'exec'), ns)
args = ns['args']
from data_provider.data_factory import data_provider

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
L, T = args.seq_len, args.pred_len
SUB = VAR == 'sublast'


def windows(ds):
    X = torch.tensor(ds.data_x, dtype=torch.float64, device=dev)
    N = len(ds)
    for c in range(X.shape[1]):
        s = X[:, c]
        xw, yw = s.unfold(0, L, 1)[:N], s.unfold(0, T, 1)[L:L + N]
        last = xw[:, -1:] if SUB else 0
        A = torch.cat([xw - last, torch.ones(N, 1, dtype=torch.float64, device=dev)], 1)
        yield A, yw, last


def errors(ds, W, keep=None):
    se = ae = n = 0
    for A, yw, last in windows(ds):
        d = (A @ W + last - yw)[:keep]
        se += float((d ** 2).sum())
        ae += float(d.abs().sum())
        n += d.numel()
    return se / n, ae / n


tr, va, te = (data_provider(args, f)[0] for f in ('train', 'val', 'test'))
G = torch.zeros(L + 1, L + 1, dtype=torch.float64, device=dev)
B = torch.zeros(L + 1, T, dtype=torch.float64, device=dev)
n = 0
for A, yw, last in windows(tr):
    G += A.T @ A
    B += A.T @ (yw - last)
    n += A.shape[0]
pen = torch.eye(L + 1, dtype=torch.float64, device=dev)
pen[-1, -1] = 0
keep = len(te) // args.batch_size * args.batch_size
best = None
for lam in LAMS:
    W = torch.linalg.pinv(G, hermitian=True) @ B if lam == 0 else torch.linalg.solve(G + lam * n * pen, B)
    row = (lam, errors(va, W)[0]) + errors(te, W, keep) + (errors(te, W)[0],)
    print('RIDGE,' + ','.join(str(x) for x in row))
    if best is None or row[1] < best[1]:
        best = row
print('BEST,' + ','.join(str(x) for x in best))
