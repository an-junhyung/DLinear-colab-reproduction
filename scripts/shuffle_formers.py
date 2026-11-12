"""Table 5 (order shuffling) for the Transformer models, test-time perturbation only.

Usage (from anywhere):
  python shuffle_formers.py <repo_dir> <model> <pred_len> <seed> <dataset_dir>
  model: Autoformer | Informer | Transformer | FEDformer

The model is trained once on the normal order with the repository's exchange-rate
settings (seq_len 96, label_len 48, 1 epoch for the Formers), then the test set is scored
five ways:
  none      : as is
  shuf      : encoder input shuffled in time, decoder start tokens left as they are
  half      : encoder input halves swapped, decoder start tokens left as they are
  shuf_all  : as shuf, and the decoder start tokens are taken from the shuffled input
  half_all  : as half, and the decoder start tokens are taken from the swapped input
Prints lines 'RESULT,<model>,<pred_len>,<seed>,<mode>,<mse>,<mae>'.
"""
import os
import sys

repo, model, P, seed, data_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
fed = model == 'FEDformer'
root = os.path.join(repo, 'FEDformer') if fed else repo
os.chdir(root)
sys.path.insert(0, root)

import numpy as np
import torch
import exp.exp_main as em

if not hasattr(np, 'Inf'):  # NumPy 2 removed np.Inf, which the repository still uses
    np.Inf = np.inf

MODES = ['none', 'shuf', 'half', 'shuf_all', 'half_all']


def perturb(x, mode, g):
    if mode.startswith('shuf'):
        idx = torch.randperm(x.shape[1], generator=g).to(x.device)
        return x[:, idx, :]
    if mode.startswith('half'):
        h = x.shape[1] // 2
        return torch.cat([x[:, h:, :], x[:, :h, :]], dim=1)
    return x


def test(self, setting, test=0):
    _, loader = self._get_data(flag='test')
    if test:
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
    self.model.eval()
    a = self.args
    for mode in MODES:
        g = torch.Generator().manual_seed(seed)
        se = ae = 0.0
        cnt = 0
        with torch.no_grad():
            for bx, by, bxm, bym in loader:
                bx = bx.float().to(self.device); by = by.float().to(self.device)
                bxm = bxm.float().to(self.device); bym = bym.float().to(self.device)
                px = perturb(bx, mode, g)
                if 'Linear' in a.model:
                    out = self.model(px)
                else:
                    start = px[:, -a.label_len:, :] if mode.endswith('_all') else by[:, :a.label_len, :]
                    dec = torch.cat([start, torch.zeros_like(by[:, -a.pred_len:, :])], dim=1).float()
                    out = self.model(px, bxm, dec, bym)
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                f = -1 if a.features == 'MS' else 0
                d = (out[:, -a.pred_len:, f:] - by[:, -a.pred_len:, f:]).double()
                se += float((d ** 2).sum()); ae += float(d.abs().sum()); cnt += d.numel()
        print(f'RESULT,{model},{P},{seed},{mode},{se / cnt:.6f},{ae / cnt:.6f}', flush=True)


em.Exp_Main.test = test

argv = ['--is_training', '1', '--root_path', data_dir, '--data_path', 'exchange_rate.csv',
        '--data', 'custom', '--features', 'M', '--seq_len', '96', '--pred_len', str(P),
        '--enc_in', '8', '--des', 'Exp', '--itr', '1', '--num_workers', '0']
argv += ['--label_len', '48', '--e_layers', '2', '--d_layers', '1', '--factor', '3',
         '--dec_in', '8', '--c_out', '8', '--train_epochs', '1', '--model', model]
argv += ['--task_id', f'sh_{P}_{seed}'] if fed else ['--model_id', f'sh_{P}_{seed}']
sys.argv = ['run.py' if fed else 'run_longExp.py'] + argv
src = open(sys.argv[0]).read()
src = src.replace('fix_seed = 2021', f'fix_seed = {seed}')
exec(compile(src, sys.argv[0], 'exec'), {'__name__': '__main__'})
