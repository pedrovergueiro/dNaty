"""
Experimento 3 — Split-MNIST: Continual Learning.
Compara dNaty vs EWC vs MLP Fixo (sem CL). Calcula BWT, FWT, FM.
"""
from __future__ import annotations
import os, json, time
import numpy as np
import torch

from dnaty.experiments.data_utils import get_split_mnist, get_mnist
from dnaty.experiments.baselines import train_ewc_cl, FixedMLP, train_fixed_mlp
from dnaty.evolution.evolver import DnatyEvolver
from dnaty.training.local_train import evaluate, micro_adapt
from dnaty.analysis.cl_metrics import compute_cl_metrics
from dnaty.analysis.stats import summary_stats
from dnaty.core.arch import DynamicMLP
from dnaty.core.individual import Individual

SEEDS = [0, 1, 2, 3, 4]
N_TASKS = 5
N_EPOCHS_CL = 5
TRAIN_SUBSET_CL = 800
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_dnaty_cl_seed(seed: int, device: str) -> dict:
    """dNaty em Split-MNIST sequencial com micro-adaptação."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    task_loaders = [get_split_mnist(t, train_subset=TRAIN_SUBSET_CL) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))

    # Inicializar modelo base com dNaty (evolver leve na tarefa 0)
    evolver = DnatyEvolver(
        n_pop=4, n_generations=8, t_local=2, device=device, verbose=False,
    )
    train0, val0 = task_loaders[0]
    best_ind, _ = evolver.run(train0, val0)
    current_ind = best_ind

    import torch.optim as optim
    import torch.nn as nn

    for t in range(N_TASKS):
        train_l, val_l = task_loaders[t]

        if t > 0:
            # Micro-adaptação: fine-tuning com taxa muito baixa (anti-forgetting)
            opt = optim.Adam(current_ind.model.parameters(), lr=5e-5)
            criterion = nn.CrossEntropyLoss()
            current_ind.model.train()
            for epoch in range(N_EPOCHS_CL):
                for xb, yb in train_l:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    criterion(current_ind.model(xb), yb).backward()
                    opt.step()

        # Avaliar em todas as tarefas até t
        for j in range(t + 1):
            acc, _ = evaluate(current_ind, task_loaders[j][1], device)
            R[t, j] = acc

    # Baselines single-task para FWT
    baselines = np.zeros(N_TASKS)
    for t in range(N_TASKS):
        train_l, val_l = task_loaders[t]
        acc, _ = train_fixed_mlp(train_l, val_l, n_epochs=8, device=device)
        baselines[t] = acc

    metrics = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics, "baselines": baselines.tolist()}


def run_ewc_cl_seed(seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    task_loaders = [get_split_mnist(t, train_subset=TRAIN_SUBSET_CL) for t in range(N_TASKS)]
    R = train_ewc_cl(task_loaders, n_epochs=N_EPOCHS_CL, device=device)
    baselines = np.zeros(N_TASKS)
    for t in range(N_TASKS):
        train_l, val_l = task_loaders[t]
        acc, _ = train_fixed_mlp(train_l, val_l, n_epochs=8, device=device)
        baselines[t] = acc
    metrics = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics}


def run_mlp_cl_seed(seed: int, device: str) -> dict:
    """MLP Fixo sem CL — treina sequencialmente sem proteção."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    import torch.nn as nn
    import torch.optim as optim

    task_loaders = [get_split_mnist(t, train_subset=TRAIN_SUBSET_CL) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))
    model = FixedMLP(hidden=[128, 64]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    def eval_task(loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        return correct / max(total, 1)

    for t, (train_l, _) in enumerate(task_loaders):
        model.train()
        for _ in range(N_EPOCHS_CL):
            for xb, yb in train_l:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()
        for j in range(t + 1):
            R[t, j] = eval_task(task_loaders[j][1])

    baselines = np.zeros(N_TASKS)
    for t in range(N_TASKS):
        train_l, val_l = task_loaders[t]
        acc, _ = train_fixed_mlp(train_l, val_l, n_epochs=10, device=device)
        baselines[t] = acc

    metrics = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"\n{'='*60}")
    print("Experimento 3 — Split-MNIST Continual Learning")
    print(f"{'='*60}")

    dnaty_results, ewc_results, mlp_results = [], [], []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        print("  [dNaty CL]")
        dr = run_dnaty_cl_seed(seed, device)
        dnaty_results.append(dr)
        print(f"  BWT={dr['metrics']['BWT']:.4f} | FWT={dr['metrics']['FWT']:.4f} | FM={dr['metrics']['FM']:.4f}")

        print("  [EWC]")
        er = run_ewc_cl_seed(seed, device)
        ewc_results.append(er)
        print(f"  BWT={er['metrics']['BWT']:.4f} | FWT={er['metrics']['FWT']:.4f} | FM={er['metrics']['FM']:.4f}")

        print("  [MLP Fixo (sem CL)]")
        mr = run_mlp_cl_seed(seed, device)
        mlp_results.append(mr)
        print(f"  BWT={mr['metrics']['BWT']:.4f} | FWT={mr['metrics']['FWT']:.4f} | FM={mr['metrics']['FM']:.4f}")

    def mean_metric(results, key):
        return round(float(np.mean([r["metrics"][key] for r in results])), 4)

    def std_metric(results, key):
        return round(float(np.std([r["metrics"][key] for r in results])), 4)

    print(f"\n{'─'*50}")
    print("RESULTADOS FINAIS — Split-MNIST CL")
    print(f"{'─'*50}")
    for name, results in [("dNaty", dnaty_results), ("EWC", ewc_results), ("MLP Fixo", mlp_results)]:
        bwt = mean_metric(results, "BWT")
        fwt = mean_metric(results, "FWT")
        fm = mean_metric(results, "FM")
        bwt_std = std_metric(results, "BWT")
        print(f"  {name:12s} | BWT={bwt:.4f}±{bwt_std:.4f} | FWT={fwt:.4f} | FM={fm:.4f}")

    dnaty_bwt = [r["metrics"]["BWT"] for r in dnaty_results]
    ewc_bwt = [r["metrics"]["BWT"] for r in ewc_results]
    from dnaty.analysis.stats import paired_ttest
    t, p, d = paired_ttest(dnaty_bwt, ewc_bwt)
    print(f"\n  dNaty vs EWC (BWT): p={p:.4f}, d={d:.3f} {'*' if p < 0.05 else ''}")

    all_results = {
        "dnaty": dnaty_results,
        "ewc": ewc_results,
        "mlp_no_cl": mlp_results,
        "summary": {
            "dnaty_bwt": {"mean": mean_metric(dnaty_results, "BWT"), "std": std_metric(dnaty_results, "BWT")},
            "ewc_bwt": {"mean": mean_metric(ewc_results, "BWT"), "std": std_metric(ewc_results, "BWT")},
            "mlp_bwt": {"mean": mean_metric(mlp_results, "BWT"), "std": std_metric(mlp_results, "BWT")},
            "dnaty_fwt": mean_metric(dnaty_results, "FWT"),
            "dnaty_fm": mean_metric(dnaty_results, "FM"),
            "ttest_dnaty_vs_ewc_bwt": {"t": t, "p": p, "d": d},
        },
    }

    out_path = os.path.join(RESULTS_DIR, "exp3_cl_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        import json
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {out_path}")
    return all_results


if __name__ == "__main__":
    main()
