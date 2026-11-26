"""Stage 1 of the improvement hypotheses (H1, H2, H3): builds the job list and runs it.

Usage:
  python hyp_run.py <june_repo> <aug_repo> <out_dir> <datasets, comma-separated> [workers] [num_workers]

  june_repo   LTSF-Linear at 04ffd8d (the code that gives the paper's DLinear numbers)
  aug_repo    LTSF-Linear at 7f7a25b (the first code with NLinear)
  datasets    ETTh1,ETTh2,ETTm1,ETTm2,exchange_rate,ili,weather,electricity,traffic

Per dataset and forecast length:
  Ridge, Ridge+sub          closed-form solution, plain and with the last value subtracted   (H2)
  DLinear                   baseline, seeds 2021-2025
  DLinear+sub               DLinear with the last value subtracted, same settings            (H1)
  NLinear                   baseline, seeds 2021-2025 (only where its settings were reproduced)
  seed = ens                the five seeds' weights averaged                                 (H3)

Settings come from the repositories' own scripts, with the changes that reproduce the paper:
  DLinear  ILI learning rate 0.01, ETTm2 720 learning rate 0.01, Traffic up to 30 epochs
  NLinear  ETTh2 learning rate 0.005, ETTm2 720 learning rate 0.01

Finished jobs are in <out_dir>/results.csv and are skipped when the script is run again.
"""
import csv
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lab

JUNE, AUG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DATASETS = sys.argv[4].split(',')
WORKERS = int(sys.argv[5]) if len(sys.argv) > 5 else 1
NUM_WORKERS = sys.argv[6] if len(sys.argv) > 6 else '2'
SEEDS = [2021, 2022, 2023, 2024, 2025]
NLINEAR_DATA = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate']  # cells reproduced with the Aug code
COLS = ['model', 'data', 'pred_len', 'seed', 'lr', 'batch', 'lam', 'mse', 'mae', 'mse_all', 'val',
        'epochs_run', 'best_epoch', 'seconds']
RES = os.path.join(OUT, 'results.csv')
os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'ckpt'), exist_ok=True)
lab._SHIM, lab._WORK = os.path.join(OUT, '_shim'), os.path.join(OUT, '_collect')
lock = threading.Lock()


def done():
    if not os.path.exists(RES):
        return set()
    return {(r['model'], r['data'], r['pred_len'], r['seed']) for r in csv.DictReader(open(RES))}


def save(**row):
    with lock:
        new = not os.path.exists(RES)
        with open(RES, 'a', newline='') as f:
            w = csv.DictWriter(f, COLS)
            if new:
                w.writeheader()
            w.writerow({c: row.get(c, '') for c in COLS})


def official(repo, sub, name, model):
    lab.ROOT = repo
    edits = [('model_name=DLinear', 'model_name=' + model)] if model != 'DLinear' else []
    cmds = lab.commands('scripts/EXP-LongForecasting/%s/%s.sh' % (sub, name.lower()), edits)
    assert len(cmds) == 4, (name, len(cmds))
    for c in cmds:
        p = lab.arg(c, '--pred_len')
        lab.set_arg(c, '--num_workers', NUM_WORKERS)
        if model == 'DLinear':
            if name == 'ili' or (name == 'ETTm2' and p == '720'):
                lab.set_arg(c, '--learning_rate', '0.01')
            if name == 'traffic':
                lab.set_arg(c, '--train_epochs', '30')
        else:
            if name == 'ETTh2':
                lab.set_arg(c, '--learning_rate', '0.005')
            if name == 'ETTm2' and p == '720':
                lab.set_arg(c, '--learning_rate', '0.01')
    return cmds


def run(repo, c, model, data, seed, kv):
    p = lab.arg(c, '--pred_len')
    key = (model, data, p, str(seed))
    if key in DONE:
        return
    tag = '%s_%s_%s_%s' % (model.replace('+', ''), data, p, seed)
    argv = c['argv'][c['argv'].index('run_longExp.py') + 1:]
    argv[argv.index('--model_id') + 1] = 'hyp_' + tag
    log = os.path.join(OUT, 'logs', tag + '.log')
    t = time.time()
    with open(log, 'w') as f:
        r = subprocess.run([sys.executable, '-u', os.path.join(HERE, 'hyp_exp.py'), repo]
                           + ['%s=%s' % x for x in kv.items()] + ['--'] + argv,
                           stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS='1'))
    s = open(log, errors='ignore').read()
    row = dict(model=model, data=data, pred_len=p, seed=seed, lr=lab.arg(c, '--learning_rate'),
               batch=lab.arg(c, '--batch_size'), seconds=round(time.time() - t))
    if kv.get('mode') == 'ridge':
        m = re.search(r'^BEST,(.*)$', s, re.M)
        if m:
            lam, val, mse, mae, mse_all = m.group(1).split(',')
            row.update(lam=lam, val=val, mse=mse, mae=mae, mse_all=mse_all, lr='', seed='-')
    else:
        m = re.search(r'^mse:([-\d.eE+]+), mae:([-\d.eE+]+)', s, re.M)
        if m:
            vali = [float(v) for v in re.findall(r'^Epoch: \d+, Steps.*Vali Loss: ([-\d.eE+]+)', s, re.M)]
            row.update(mse=m.group(1), mae=m.group(2))
            if vali:
                row.update(epochs_run=len(vali), best_epoch=vali.index(min(vali)) + 1, val=min(vali))
    if r.returncode != 0 or not m:
        print('FAILED %s\n%s' % (tag, s[-500:]), flush=True)
        return
    save(**row)
    print('%-34s mse %.4f  (%ds)' % (tag, float(row['mse']), row['seconds']), flush=True)


def ckpt(model, data, p, seed):
    return os.path.join(OUT, 'ckpt', '%s_%s_%s_%s.pth' % (model.replace('+', ''), data, p, seed))


for data in DATASETS:
    DONE = done()
    dl = official(JUNE, 'DLinear', data, 'DLinear')
    nl = official(AUG, 'Linear', data, 'NLinear') if data in NLINEAR_DATA else []
    jobs = []
    for c in dl:
        p = lab.arg(c, '--pred_len')
        jobs.append((JUNE, c, 'Ridge', data, '-', dict(mode='ridge')))
        jobs.append((JUNE, c, 'Ridge+sub', data, '-', dict(mode='ridge', variant='sublast')))
        for s in SEEDS:
            jobs.append((JUNE, c, 'DLinear', data, s, dict(mode='train', seed=s, keep=ckpt('DLinear', data, p, s))))
            jobs.append((JUNE, c, 'DLinear+sub', data, s,
                         dict(mode='train', seed=s, variant='sublast', keep=ckpt('DLinear+sub', data, p, s))))
    for c in nl:
        p = lab.arg(c, '--pred_len')
        for s in SEEDS:
            jobs.append((AUG, c, 'NLinear', data, s, dict(mode='train', seed=s, keep=ckpt('NLinear', data, p, s))))
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(lambda j: run(*j), jobs))
    DONE = done()
    ens = []
    for repo, cmds, models in [(JUNE, dl, ['DLinear', 'DLinear+sub']), (AUG, nl, ['NLinear'])]:
        for c in cmds:
            p = lab.arg(c, '--pred_len')
            for m in models:
                files = [ckpt(m, data, p, s) for s in SEEDS]
                if all(os.path.exists(f) for f in files):
                    kv = dict(mode='ens', ckpts=','.join(files))
                    if m == 'DLinear+sub':
                        kv['variant'] = 'sublast'
                    ens.append((repo, c, m, data, 'ens', kv))
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(lambda j: run(*j), ens))
    print('== %s done' % data, flush=True)
