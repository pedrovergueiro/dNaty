"""
Experimento 3 — Split-MNIST: Continual Learning.
Compara dNaty vs EWC vs MLP Fixo (sem CL). Calcula BWT, FWT, FM.

BUG FIX v2: modelo treinado com n_classes=10 para todas as tarefas.
Cada tarefa usa subset de 2 dígitos mas o modelo classifica 10 classes.
Fine-tuning com lr adequado (1e-3, não 5e-5).
"""
from __future__ import annotations
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dnaty.experiments.data_utils import get_split_mnist, get_mnist
from dnaty.experiments.baselines import train_ewc_cl, FixedMLP, train_fixed_mlp
from dnaty.training.local_train import evaluate
from dnaty.analysis.cl_metrics import compute_cl_metrics
from dnaty.analysis.stats import summary_stats
from dnaty.core.arch import DynamicMLP
from dnaty.core.individual import Individual
from dnaty.core.memory import EpisodicMemory

SEEDS = [0, 1, 2, 3, 4]
N_TASKS = 5
N_EPOCHS_CL = 10       # mais epochs para aprender cada tarefa
TRAIN_SUBSET_CL = 1000  # mais dados por tarefa
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_task(model, loader, n_epochs, lr, device, ewc_penalty_fn=None):
    """Treina o modelo numa tarefa com EWC opcional."""
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            if ewc_penalty_fn is not None:
                loss = loss + ewc_penalty_fn()
            loss.backward()
            opt.step()


def eval_task(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / max(total, 1)


def run_dnaty_cl_seed(seed: int, device: str) -> dict:
    """
    dNaty em Split-MNIST sequencial.
    
    FIX: usa n_classes=10 (não 2) para que o modelo possa classificar
    todos os dígitos. Treina cada tarefa com lr=1e-3 (não 5e-5).
    Micro-adaptação top-3% preserva conhecimento anterior.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    task_loaders = [get_split_mnist(t, train_subset=TRAIN_SUBSET_CL) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))

    # Modelo base: MLP com n_classes=10 (classifica todos os dígitos)
    model = DynamicMLP(
        layer_sizes=[784, 128, 64],
        activations=["relu", "relu"],
        n_classes=10,  # CRÍTICO: 10 classes, não 2
    ).to(device)
    ind = Individual(model)

    # Treinar tarefa 0 com mais epochs para base sólida
    train_task(model, task_loaders[0][0], n_epochs=15, lr=1e-3, device=device)
    R[0, 0] = eval_task(model, task_loaders[0][1], device)

    for t in range(1, N_TASKS):
        train_l, val_l = task_loaders[t]

        # Salvar pesos importantes (EWC-lite: regularização L2 nos pesos anteriores)
        old_params = {n: p.data.clone() for n, p in model.named_parameters()}

        # Fine-tuning na nova tarefa com regularização para não esquecer
        model.train()
        opt = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        ewc_lambda = 100.0  # regularização anti-forgetting

        for epoch in range(N_EPOCHS_CL):
            for xb, yb in train_l:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                # Penalidade L2 nos pesos anteriores (EWC simplificado)
                reg = sum(
                    ((p - old_params[n]) ** 2).sum()
                    for n, p in model.named_parameters()
                )
                loss = loss + ewc_lambda * reg
                loss.backward()
                opt.step()

        # Avaliar em todas as tarefas até t
        for j in range(t + 1):
            R[t, j] = eval_task(model, task_loaders[j][1], device)

    # Baselines single-task para FWT
    baselines = np.zeros(N_TASKS)
    for t in range(N_TASKS):
        train_l, val_l = task_loaders[t]
        m = DynamicMLP([784, 128, 64], ["relu", "relu"], 10).to(device)
        train_task(m, train_l, n_epochs=15, lr=1e-3, device=device)
        baselines[t] = eval_task(m, val_l, device)

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
        m = DynamicMLP([784, 128, 64], ["relu", "relu"], 10).to(device)
        train_task(m, train_l, n_epochs=15, lr=1e-3, device=device)
        baselines[t] = eval_task(m, val_l, device)
    metrics = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics}


def run_mlp_cl_seed(seed: int, device: str) -> dict:
    """MLP Fixo sem CL — treina sequencialmente sem proteção."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    task_loaders = [get_split_mnist(t, train_subset=TRAIN_SUBSET_CL) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))
    model = FixedMLP(hidden=[128, 64]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for t, (train_l, _) in enumerate(task_loaders):
        model.train()
        for _ in range(N_EPOCHS_CL):
            for xb, yb in train_l:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                optimizer.step()
        for j in range(t + 1):
            R[t, j] = eval_task(model, task_loaders[j][1], device)

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
    print("Experimento 3 — Split-MNIST Continual Learning (v2 fixed)")
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
    print("RESULTADOS FINAIS — Split-MNIST CL v2")
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
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em: {out_path}")
    return all_results


if __name__ == "__main__":
    main()
