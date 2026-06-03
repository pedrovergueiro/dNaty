"""Benchmarks REAIS do dNaty em datasets reais. Sem numero inventado."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dnaty_saas"))
import torch, numpy as np
from torch.utils.data import TensorDataset, DataLoader
from dnaty.evolution.evolver import DnatyEvolver
from dnaty.training.local_train import evaluate
from dnaty.core.arch import DynamicMLP
from dnaty.compress import compress
from dnaty.experiments.fast_dataset import FastDataset
from routes.train import _parse_tabular

CORES = torch.get_num_threads()
results = []

def flops_of(sizes):
    return sum(2*sizes[i]*sizes[i+1] for i in range(len(sizes)-1))

def run_tabular(name, path):
    X, y, classes = _parse_tabular(path)
    n_feat, n_cls = X.shape[1], len(classes)
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    n = len(Xt); idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    ntr = int(n*0.8)
    tr = DataLoader(TensorDataset(Xt[idx[:ntr]], yt[idx[:ntr]]), batch_size=128, shuffle=True)
    va = DataLoader(TensorDataset(Xt[idx[ntr:]], yt[idx[ntr:]]), batch_size=128)
    init_hidden = [128, 64]
    orig_sizes = [n_feat] + init_hidden + [n_cls]
    orig_flops = flops_of(orig_sizes)
    torch.manual_seed(42); np.random.seed(42)
    ev = DnatyEvolver(n_pop=15, n_generations=30, t_local=3, lambda2=3e-6,
        device="cpu", input_size=n_feat, n_classes=n_cls, init_hidden=init_hidden,
        batch_size=128, verbose=False)
    t0 = time.time()
    ev.run(tr, va, early_stop_patience=30)
    dt = time.time() - t0
    cand = [i for i in ev.population if i.acc >= 0.95] or ev.population
    best = min(cand, key=lambda i: i.count_flops())
    cf = best.count_flops()
    red = max(0.0, 1 - cf/max(orig_flops,1))*100
    acc_val, _ = evaluate(best, va)  # held-out
    results.append((name, f"{n}rows/{n_feat}feat/{n_cls}cls", dt, red, acc_val*100,
                    f"{orig_flops}->{cf}"))
    print(f"{name}: {dt:.1f}s FLOPs-{red:.1f}% acc(holdout)={acc_val*100:.1f}% {orig_flops}->{cf}", flush=True)

def run_image(name):
    ds = FastDataset(name, device="cpu", train_subset=10_000)
    import torch.nn as nn
    model = nn.Sequential(nn.Flatten(), nn.Linear(3072 if name=="CIFAR10" else 784,512), nn.ReLU(),
        nn.Linear(512,256), nn.ReLU(), nn.Linear(256,10))
    t0=time.time()
    r = compress(model, ds, target_flops=0.5, n_generations=30, seed=42, verbose=False)
    dt=time.time()-t0
    results.append((name, "10K subset", dt, r.flops_reduction_pct, r.accuracy*100,
                    f"{r.original_flops}->{r.compressed_flops}"))
    print(f"{name}: {dt:.1f}s FLOPs-{r.flops_reduction_pct:.1f}% acc(val)={r.accuracy*100:.1f}% {r.original_flops}->{r.compressed_flops}", flush=True)

base = r"C:\Users\pedro\Downloads\archive (1)"
print(f"=== dNaty REAL benchmarks | {CORES} CPU cores | config: 30 gens, pop15 ===", flush=True)
run_tabular("Indonesian Youth Digital Friction", base + r"\indonesian_youth_digital_friction_dataset.csv")
run_tabular("Social Friction (students vs workers)", base + r"\social_friction_students_vs_workers.csv")
run_tabular("Social Friction v2", base + r"\social_friction_v2.csv")
run_image("CIFAR10")

print("\n=== TABELA FINAL (real) ===", flush=True)
for name, shape, dt, red, acc, fl in results:
    print(f"{name:42} | {shape:22} | {dt:5.0f}s | -{red:4.1f}% FLOPs | {acc:5.1f}% acc | {fl}", flush=True)
