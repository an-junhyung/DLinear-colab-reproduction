# 논문 Table 8 DLinear 행: Electricity, L=96, T=720 (변수 321개)
import os, subprocess, sys, time, types, statistics
if not os.path.isdir('/content/LTSF-Linear'):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/cure-lab/LTSF-Linear.git', '/content/LTSF-Linear'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ptflops'], check=True)
sys.path.insert(0, '/content/LTSF-Linear')
import torch
from models import DLinear
from ptflops import get_model_complexity_info

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
args = types.SimpleNamespace(seq_len=96, pred_len=720, individual=False, enc_in=321)
model = DLinear.Model(args).to(dev).eval()

n_params = sum(p.numel() for p in model.parameters())
macs, _ = get_model_complexity_info(model, (96, 321), as_strings=False,
                                    print_per_layer_stat=False, verbose=False)
print(f'device: {torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}')
print(f'params: {n_params:,} ({n_params/1e3:.1f}K)   MACs (batch 1): {macs/1e9:.3f}G')

def sync():
    if dev == 'cuda':
        torch.cuda.synchronize()

for bs in [1, 16, 32]:
    x = torch.randn(bs, 96, 321, device=dev)
    if dev == 'cuda':
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(50):          # warm-up
            model(x)
        sync()
        runs = []
        for _ in range(5):           # 논문: 5회 평균
            t0 = time.perf_counter()
            for _ in range(200):
                model(x)
            sync()
            runs.append((time.perf_counter() - t0) / 200 * 1e3)
    line = f'batch {bs:>2}: {statistics.mean(runs):.3f} ms/step (5 runs: ' + ', '.join(f'{r:.3f}' for r in runs) + ')'
    if dev == 'cuda':
        line += f'   peak tensor memory: {torch.cuda.max_memory_allocated()/2**20:.1f} MiB'
    print(line)

if dev == 'cuda':
    used = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip()
    print(f'nvidia-smi memory.used (CUDA context 포함): {used}')
