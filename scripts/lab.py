"""DLinear 추가 실험 도우미.

공식 스크립트(scripts/EXP-*/...sh)를 '가짜 python'으로 한 번 돌려서 실제 실행될 명령 목록만 뽑고,
그 명령을 하나씩 실행한다. 이렇게 하면
  - 하이퍼파라미터를 손으로 옮겨 적지 않고 공식 스크립트 값을 그대로 쓰고
  - 실험마다 진행 상황이 한 줄씩 찍히고
  - 중간에 끊겨도 다시 실행하면 끝난 실험은 건너뛴다.
"""
import json, os, re, shutil, subprocess, sys, time

import pandas as pd

ROOT = '/content/LTSF-Linear'
_SHIM = '/content/_shim'
_WORK = '/content/_collect'

DATASETS = {'etth1.csv': 'ETTh1', 'etth2.csv': 'ETTh2', 'ettm1.csv': 'ETTm1', 'ettm2.csv': 'ETTm2',
            'electricity.csv': 'Electricity', 'exchange_rate.csv': 'Exchange', 'traffic.csv': 'Traffic',
            'weather.csv': 'Weather', 'national_illness.csv': 'ILI'}


def commands(script, edits=()):
    """공식 스크립트에서 실행될 python 명령 목록을 뽑는다. edits = [(찾을 문자열, 바꿀 문자열), ...]"""
    os.makedirs(_SHIM, exist_ok=True)
    with open(f'{_SHIM}/python', 'w') as f:
        f.write('#!/usr/bin/env python3\n'
                'import json, os, sys\n'
                'try:\n    log = os.readlink("/proc/self/fd/1")\nexcept OSError:\n    log = ""\n'
                'with open(os.environ["CMD_FILE"], "a") as f:\n'
                '    f.write(json.dumps({"argv": sys.argv[1:], "log": log}) + "\\n")\n')
    os.chmod(f'{_SHIM}/python', 0o755)

    shutil.rmtree(_WORK, ignore_errors=True)
    for d in ['logs/LongForecasting/univariate', 'logs/LookBackWindow']:  # 스크립트가 안 만드는 폴더도 미리
        os.makedirs(f'{_WORK}/{d}')
    src = open(os.path.join(ROOT, script)).read()
    for a, b in edits:
        assert a in src, f'{a!r} 가 {script} 에 없음'
        src = src.replace(a, b)
    open(f'{_WORK}/run.sh', 'w').write(src)
    env = dict(os.environ, PATH=f'{_SHIM}:{os.environ["PATH"]}', CMD_FILE=f'{_WORK}/cmds.jsonl')
    subprocess.run(['bash', 'run.sh'], cwd=_WORK, env=env, check=True, stdout=subprocess.DEVNULL)
    out = [json.loads(l) for l in open(f'{_WORK}/cmds.jsonl')]
    for c in out:
        c['log'] = os.path.basename(c['log']) if c['log'].startswith(_WORK) else ''
    return out


def arg(c, *names, default=None):
    a = c['argv']
    for i, x in enumerate(a[:-1]):
        if x in names:
            return a[i + 1]
    return default


def set_arg(c, name, value):
    a = c['argv']
    if name in a:
        a[a.index(name) + 1] = str(value)
    else:
        a += [name, str(value)]
    return c


def info(c):
    return dict(model=arg(c, '--model'), dataset=DATASETS[arg(c, '--data_path').lower()],
                features=arg(c, '--features', '--feature', default='M'),
                seq_len=int(arg(c, '--seq_len')), pred_len=int(arg(c, '--pred_len')),
                individual='--individual' in c['argv'])


def parse_log(path):
    """로그에서 (testing 설정, mse, mae)를 순서대로 뽑는다. --itr N이면 N개."""
    if not os.path.exists(path):
        return []
    text = open(path, errors='ignore').read()
    out = []
    for m in re.finditer(r'>>>>>>>testing : (\S+?)<<<.*?mse:([-\d.eE+]+), mae:([-\d.eE+]+)', text, re.S):
        out.append((m.group(1), float(m.group(2)), float(m.group(3))))
    return out


def run(cmds, tag):
    """명령을 차례로 실행. 로그는 logs/<tag>/ 에 저장, 이미 끝난 실험은 건너뜀."""
    logdir = os.path.join(ROOT, 'logs', tag)
    os.makedirs(logdir, exist_ok=True)
    t0 = time.time()
    for i, c in enumerate(cmds, 1):
        c['logpath'] = os.path.join(logdir, c.get('name') or c['log'] or f'run{i}.log')
        d = info(c)
        label = (f"{d['dataset']} {d['features']} L={d['seq_len']} T={d['pred_len']}"
                 + (' individual' if d['individual'] else ''))
        need = int(arg(c, '--itr', default=1))
        if len(parse_log(c['logpath'])) >= need:
            print(f'[{i}/{len(cmds)}] {label}: 이미 완료, 건너뜀', flush=True)
            continue
        t = time.time()
        with open(c['logpath'], 'w') as f:
            r = subprocess.run([sys.executable] + c['argv'], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        subprocess.run('rm -f results/*/pred.npy', shell=True, cwd=ROOT)  # 예측 배열은 안 쓰므로 디스크 절약
        res = parse_log(c['logpath'])
        if r.returncode != 0 or len(res) < need:
            tail = open(c['logpath'], errors='ignore').read()[-600:]
            print(f'[{i}/{len(cmds)}] {label}: 실패 (exit {r.returncode})\n{tail}', flush=True)
            continue
        mses = ', '.join(f'{m:.4f}' for _, m, _ in res)
        print(f'[{i}/{len(cmds)}] {label}: mse {mses}  ({time.time() - t:.0f}s, 누적 {(time.time() - t0) / 60:.0f}분)',
              flush=True)


def collect(cmds):
    rows = []
    for c in cmds:
        for k, (setting, mse, mae) in enumerate(parse_log(c.get('logpath', ''))):
            rows.append(dict(info(c), run=k, mse=mse, mae=mae))
    return pd.DataFrame(rows)
