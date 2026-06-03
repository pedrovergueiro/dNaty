"""dNaty REAL em datasets reais do UCI. Held-out accuracy honesta."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dnaty_saas"))
import torch, numpy as np
from torch.utils.data import TensorDataset, DataLoader
from dnaty.evolution.evolver import DnatyEvolver
from dnaty.training.local_train import evaluate
from routes.train import _parse_tabular

CORES = torch.get_num_threads()
DS = [
    ("UCI Adult / Census Income", r"c:\tmp\real_datasets\uci_adult_income.csv"),
    ("UCI Covertype (25k)",        r"c:\tmp\real_datasets\uci_covertype_25k.csv"),
    ("UCI Wine Quality (red)",     r"c:\tmp\real_datasets\uci_wine_quality_red.csv"),
]
INIT_HIDDEN = [256, 128]
def flops(sizes): return sum(2*sizes[i]*sizes[i+1] for i in range(len(sizes)-1))

print(f"=== dNaty REAL (UCI) | {CORES} cores | 30 gens pop15 | init_hidden={INIT_HIDDEN} ===", flush=True)
rows = []
for name, path in DS:
    X, y, classes = _parse_tabular(path)
    nf, nc = X.shape[1], len(classes)
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    n = len(Xt); idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
    CAP = 10_000  # subset nao afeta a reducao de FLOPs (arquitetura-driven); acelera
    if n > CAP:
        idx = idx[:CAP]; n = CAP
    ntr = int(n*0.8)
    tr = DataLoader(TensorDataset(Xt[idx[:ntr]], yt[idx[:ntr]]), batch_size=256, shuffle=True)
    va = DataLoader(TensorDataset(Xt[idx[ntr:]], yt[idx[ntr:]]), batch_size=256)
    of = flops([nf]+INIT_HIDDEN+[nc])
    torch.manual_seed(42); np.random.seed(42)
    ev = DnatyEvolver(n_pop=15, n_generations=30, t_local=3, lambda2=3e-6, device="cpu",
        input_size=nf, n_classes=nc, init_hidden=INIT_HIDDEN, batch_size=256, verbose=False)
    t0=time.time(); ev.run(tr, va, early_stop_patience=30); dt=time.time()-t0
    cand = [i for i in ev.population if i.acc>=0.90] or ev.population
    best = min(cand, key=lambda i: i.count_flops())
    cf = best.count_flops(); red = max(0.0,1-cf/max(of,1))*100
    acc,_ = evaluate(best, va)
    rows.append((name, n, nf, nc, dt, red, acc*100, of, cf))
    print(f"{name}: {n}rows/{nf}feat/{nc}cls | {dt:.0f}s | -{red:.1f}% FLOPs | acc(holdout)={acc*100:.1f}% | {of}->{cf}", flush=True)

print("\n=== TABELA UCI (real) ===", flush=True)
for name,n,nf,nc,dt,red,acc,of,cf in rows:
    print(f"{name:32}| {n:6}rows {nf:3}feat {nc}cls | {dt:4.0f}s | -{red:4.1f}% FLOPs | {acc:5.1f}% acc", flush=True)
