"""
Compara 4 configs: antigo, 3 variantes
MNIST com 2000 amostras, 8 gerações.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dnaty_saas"))

import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset

from dnaty.evolution.evolver import DnatyEvolver

N_SAMPLES = 2000
N_GENS    = 8
SEED      = 42

def make_loaders():
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(root="/tmp/data", train=True,  download=True, transform=t)
    test_ds  = datasets.MNIST(root="/tmp/data", train=False, download=True, transform=t)
    train_ds = Subset(train_ds, list(range(N_SAMPLES)))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True,  num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0)
    return train_loader, test_loader

def run_config(label, n_pop, t_local):
    torch.manual_seed(SEED)
    train_loader, test_loader = make_loaders()
    evolver = DnatyEvolver(
        n_pop=n_pop, n_generations=N_GENS, t_local=t_local,
        lr=1e-3, lambda1=1e-4, lambda2=1e-3,
        device="cpu", input_size=784, n_classes=10,
        init_hidden=[128, 64], batch_size=512, verbose=False,
    )
    t0 = time.time()
    best, history = evolver.run(train_loader, test_loader)
    elapsed = time.time() - t0
    total_evals = n_pop * N_GENS * t_local
    return elapsed, best.acc * 100, total_evals

configs = [
    ("ANTIGO       n_pop=10 t_local=2", 10, 2),
    ("TLOCAL1      n_pop=10 t_local=1", 10, 1),
    ("EQUILIBRADO  n_pop=8  t_local=1",  8, 1),
    ("NOVO         n_pop=6  t_local=1",  6, 1),
]

results = {}
print(f"MNIST {N_SAMPLES} amostras, {N_GENS} gerações\n{'='*65}")
for label, n_pop, t_local in configs:
    print(f"\n  {label} ...")
    elapsed, acc, evals = run_config(label, n_pop, t_local)
    results[label] = (elapsed, acc, evals)
    print(f"    Tempo: {elapsed:.1f}s | Acurácia: {acc:.2f}% | {evals} runs")

print(f"\n{'='*65}")
t_ref, a_ref = results[configs[0][0]][0], results[configs[0][0]][1]
for label, (elapsed, acc, evals) in results.items():
    speedup = t_ref / elapsed
    delta   = acc - a_ref
    print(f"  {label[:12]}  {elapsed:6.1f}s  {speedup:.2f}x  acc={acc:.2f}%  delta={delta:+.2f}pp")
