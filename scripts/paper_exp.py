"""Runs one cell of the paper's analysis experiments on the official LTSF-Linear repository.

Usage:
  python paper_exp.py <repo_dir> <data_dir> key=value ...

Keys
  exp      t2 | emb | lookback | closefar | short | t4 | eff      (label only, except closefar/short/eff)
  model    FEDformer | Autoformer | Informer | Transformer | Linear | NLinear | DLinear
           | InfAttLin | InfEmbLin          (the two Table 4 variants, built here; see below)
  data     exchange | electricity | traffic | weather | ili | ETTh1 | ETTh2 | ETTm1 | ETTm2
  pred_len, seq_len (default 96; ili 36), embed_type (default 0), seed (default 2021)
  variant  closefar: close | far        short: ori | short
  epochs   overrides the official epoch count (official: exchange 1, traffic 3, weather 2, else 10)

Prints one line:
  RESULT,<exp>,<model>,<data>,<seq_len>,<pred_len>,<embed_type>,<variant>,<seed>,<mse>,<mae>,<mse_droplast>,<mae_droplast>,<epochs_run>,<seconds>
  (eff prints RESULT,eff,<model>,<params>,<ms_per_forward>,<peak_MiB>)

What is and is not official
  - Transformer / Informer / Autoformer / FEDformer and their settings come from the repository
    (scripts/EXP-LongForecasting/Formers_Long.sh, FEDformer/scripts/LongForecasting.sh).
  - embed_type 0..4 is the repository's own switch (scripts/EXP-Embedding).
  - closefar (Table 3), short (Table 7) and the two Table 4 variants have no public code.
    They are implemented here from the paper's text, so they are an interpretation:
      closefar: every sample is [far input 0..95][close input 96..191][target 192..192+T).
                'close' feeds 96..191, 'far' feeds 0..95. The decoder start tokens are the last
                label_len steps of whichever input is fed.
      short:    the training split is cut to its last 8,760 steps. The scaler is unchanged.
      InfAttLin: Informer with every self-attention (encoder and decoder) replaced by a
                 linear layer on the token features. Cross-attention is kept.
      InfEmbLin: Informer's input embedding, a linear map d_model -> channels, then a
                 linear map over time (seq_len -> pred_len). Nothing else.
"""
import os
import sys
import time

repo, data_dir = sys.argv[1], sys.argv[2]
kv = dict(a.split('=', 1) for a in sys.argv[3:])
EXP = kv.get('exp', 't2')
MODEL = kv['model']
DATA = kv.get('data', 'exchange')
P = int(kv.get('pred_len', 96))
SEED = int(kv.get('seed', 2021))
EMB = int(kv.get('embed_type', 0))
VAR = kv.get('variant', '-')
GAP = 96            # closefar: distance between the far input and the target
SHORT_N = int(kv.get('short_n', 8760))

CFG = {  # data_path, --data, channels, official epochs for the Formers
    'exchange': ('exchange_rate.csv', 'custom', 8, 1),
    'electricity': ('electricity.csv', 'custom', 321, 10),
    'traffic': ('traffic.csv', 'custom', 862, 3),
    'weather': ('weather.csv', 'custom', 21, 2),
    'ili': ('national_illness.csv', 'custom', 7, 10),
    'ETTh1': ('ETTh1.csv', 'ETTh1', 7, 10), 'ETTh2': ('ETTh2.csv', 'ETTh2', 7, 10),
    'ETTm1': ('ETTm1.csv', 'ETTm1', 7, 10), 'ETTm2': ('ETTm2.csv', 'ETTm2', 7, 10),
}
path, dname, C, off_ep = CFG[DATA]
L = int(kv.get('seq_len', 36 if DATA == 'ili' else 96))
LABEL = 18 if DATA == 'ili' else 48
linear = MODEL in ('Linear', 'NLinear', 'DLinear')
EPOCHS = int(kv.get('epochs', 10 if linear else off_ep))

fed = MODEL == 'FEDformer'
root = os.path.join(repo, 'FEDformer') if fed else repo
os.chdir(root)
sys.path.insert(0, root)

import numpy as np
import torch
import torch.nn as nn

if not hasattr(np, 'Inf'):  # NumPy 2 removed np.Inf, which the repository still uses
    np.Inf = np.inf

import data_provider.data_loader as dl
import exp.exp_main as em

# ---------------------------------------------------------------- data variants
if EXP == 'closefar':
    for cls in (dl.Dataset_Custom,):
        def _len(self):
            return len(self.data_x) - self.seq_len - GAP - self.pred_len + 1

        def _get(self, index):
            ib = index + (GAP if VAR == 'close' else 0)
            ie = ib + self.seq_len
            tb = index + GAP + self.seq_len
            te = tb + self.pred_len
            seq_x = self.data_x[ib:ie]
            seq_y = np.concatenate([self.data_x[ie - self.label_len:ie], self.data_y[tb:te]], 0)
            x_mark = self.data_stamp[ib:ie]
            y_mark = np.concatenate([self.data_stamp[ie - self.label_len:ie], self.data_stamp[tb:te]], 0)
            return seq_x, seq_y, x_mark, y_mark
        cls.__len__ = _len
        cls.__getitem__ = _get

if EXP == 'short' and VAR == 'short':
    _read = dl.Dataset_Custom.__read_data__

    def _read_short(self):
        _read(self)
        if self.set_type == 0:
            self.data_x = self.data_x[-SHORT_N:]
            self.data_y = self.data_y[-SHORT_N:]
            self.data_stamp = self.data_stamp[-SHORT_N:]
    dl.Dataset_Custom.__read_data__ = _read_short

# ---------------------------------------------------------------- Table 4 variants
if MODEL in ('InfAttLin', 'InfEmbLin'):
    from models import Informer
    from layers.Embed import DataEmbedding

    class _LinAttn(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lin = nn.Linear(d, d)

        def forward(self, q, k, v, attn_mask=None):
            return self.lin(q), None

    class _AttLin(nn.Module):
        def __init__(self, a):
            super().__init__()
            self.m = Informer.Model(a)
            for layer in self.m.encoder.attn_layers:
                layer.attention = _LinAttn(a.d_model)
            for layer in self.m.decoder.layers:
                layer.self_attention = _LinAttn(a.d_model)

        def forward(self, *x, **k):
            return self.m(*x, **k)

    class _EmbLin(nn.Module):
        def __init__(self, a):
            super().__init__()
            self.emb = DataEmbedding(a.enc_in, a.d_model, a.embed, a.freq, a.dropout)
            self.proj = nn.Linear(a.d_model, a.c_out)
            self.time = nn.Linear(a.seq_len, a.pred_len)

        def forward(self, x_enc, x_mark_enc, *a, **k):
            o = self.proj(self.emb(x_enc, x_mark_enc))
            return self.time(o.permute(0, 2, 1)).permute(0, 2, 1)

    _build = em.Exp_Main._build_model

    def _build_custom(self):
        if self.args.model == 'InfAttLin':
            return _AttLin(self.args).float()
        if self.args.model == 'InfEmbLin':
            return _EmbLin(self.args).float()
        return _build(self)
    em.Exp_Main._build_model = _build_custom

# ---------------------------------------------------------------- scoring
T0 = time.time()
STATE = {'epochs': 0}
_vali = em.Exp_Main.vali
_calls = {'n': 0}


def _vali_count(self, *a, **k):  # vali is called twice per epoch (val, test) in the repository
    _calls['n'] += 1
    return _vali(self, *a, **k)


em.Exp_Main.vali = _vali_count


def _forward(self, bx, by, bxm, bym):
    a = self.args
    if 'Linear' in a.model:
        return self.model(bx)
    dec = torch.cat([by[:, :a.label_len, :], torch.zeros_like(by[:, -a.pred_len:, :])], dim=1).float()
    out = self.model(bx, bxm, dec, bym)
    return out[0] if isinstance(out, (tuple, list)) else out


def test(self, setting, test=0):
    _, loader = self._get_data(flag='test')
    self.model.eval()
    a = self.args
    se, ae = [], []
    with torch.no_grad():
        for bx, by, bxm, bym in loader:
            bx = bx.float().to(self.device); by = by.float().to(self.device)
            bxm = bxm.float().to(self.device); bym = bym.float().to(self.device)
            out = _forward(self, bx, by, bxm, bym)
            d = (out[:, -a.pred_len:, :] - by[:, -a.pred_len:, :]).double()
            se.append((d ** 2).mean(dim=(1, 2)).cpu()); ae.append(d.abs().mean(dim=(1, 2)).cpu())
    se = torch.cat(se).numpy(); ae = torch.cat(ae).numpy()
    m = (len(se) // a.batch_size) * a.batch_size  # what a drop_last=True test loader would score
    print(f'RESULT,{EXP},{MODEL},{DATA},{L},{P},{EMB},{VAR},{SEED},{se.mean():.6f},{ae.mean():.6f},'
          f'{se[:m].mean():.6f},{ae[:m].mean():.6f},{_calls["n"] // 2},{time.time() - T0:.0f}', flush=True)


em.Exp_Main.test = test

if EXP == 'eff':
    def measure(self, setting):
        a = self.args
        dev = self.device
        self.model.eval()
        n = sum(p.numel() for p in self.model.parameters())
        bx = torch.randn(1, a.seq_len, a.enc_in, device=dev); bxm = torch.randn(1, a.seq_len, 4, device=dev)
        by = torch.randn(1, a.label_len + a.pred_len, a.enc_in, device=dev)
        bym = torch.randn(1, a.label_len + a.pred_len, 4, device=dev)
        cuda = dev.type == 'cuda'
        with torch.no_grad():
            for _ in range(3):
                _forward(self, bx, by, bxm, bym)
            if cuda:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            ts = []
            for _ in range(5):
                t = time.time(); _forward(self, bx, by, bxm, bym)
                if cuda:
                    torch.cuda.synchronize()
                ts.append(time.time() - t)
        peak = torch.cuda.max_memory_allocated() / 2 ** 20 if cuda else float('nan')
        print(f'RESULT,eff,{MODEL},{n},{1000 * sum(ts) / 5:.2f},{peak:.0f}', flush=True)
        return self.model
    em.Exp_Main.train = measure
    em.Exp_Main.test = lambda self, setting, test=0: None

# ---------------------------------------------------------------- run the repository's own entry script
tag = f'px_{EXP}_{MODEL}_{DATA}_{L}_{P}_{EMB}_{VAR}_{SEED}'
argv = ['--is_training', '1', '--root_path', data_dir, '--data_path', path, '--data', dname,
        '--features', 'M', '--seq_len', str(L), '--label_len', str(LABEL), '--pred_len', str(P),
        '--enc_in', str(C), '--dec_in', str(C), '--c_out', str(C), '--des', 'Exp', '--itr', '1',
        '--num_workers', '0', '--train_epochs', str(EPOCHS), '--model', MODEL,
        '--e_layers', '2', '--d_layers', '1', '--factor', '3', '--embed_type', str(EMB)]
if linear:
    argv += ['--learning_rate', kv.get('lr', '0.005'), '--batch_size', kv.get('bs', '32')]
argv += ['--task_id', tag] if fed else ['--model_id', tag]
sys.argv = ['run.py' if fed else 'run_longExp.py'] + argv
src = open(sys.argv[0]).read().replace('fix_seed = 2021', f'fix_seed = {SEED}')
exec(compile(src, sys.argv[0], 'exec'), {'__name__': '__main__'})
