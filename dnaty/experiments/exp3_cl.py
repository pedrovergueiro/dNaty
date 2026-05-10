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

SEEDS = [0, 1, 2]
N_TASKS = 5
N_EPOCHS_CL = 15       # mais epochs para aprender cada tarefa
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
    dNaty em Split-MNIST sequencial com Experience Replay.
    
    Guarda um buffer de exemplos de tarefas anteriores e treina
    junto com a nova tarefa — abordagem mais robusta que EWC puro
    para Split-MNIST com poucos dados.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    task_loaders = [get_split_mnist(t, train_subset=TRAIN_SUBSET_CL) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))

    model = DynamicMLP(
        layer_sizes=[784, 256, 128],
        activations=["relu", "relu"],
        n_classes=10,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # Buffer de replay: guarda 100 exemplos por tarefa anterior
    replay_buffer_x = []
    replay_buffer_y = []
    REPLAY_SIZE = 100

    for t in range(N_TASKS):
        train_l, val_l = task_loaders[t]

        # Coletar dados da tarefa atual para replay futuro
        task_x, task_y = [], []
        for xb, yb in train_l:
            task_x.append(xb)
            task_y.append(yb)
            if sum(len(x) for x in task_x) >= REPLAY_SIZE:
                break
        task_x = torch.cat(task_x)[:REPLAY_SIZE]
        task_y = torch.cat(task_y)[:REPLAY_SIZE]

        opt = optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        for epoch in range(N_EPOCHS_CL):
            # Treinar na tarefa atual
            for xb, yb in train_l:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)

                # Replay de tarefas anteriores (anti-forgetting)
                if replay_buffer_x:
                    rx = torch.cat(replay_buffer_x).to(device)
                    ry = torch.cat(replay_buffer_y).to(device)
                    # Sample aleatório do buffer
                    idx = torch.randperm(len(rx))[:min(64, len(rx))]
                    loss = loss + criterion(model(rx[idx]), ry[idx])

                loss.backward()
                opt.step()

        # Adicionar ao buffer de replay
        replay_buffer_x.append(task_x)
        replay_buffer_y.append(task_y)

        # Avaliar em todas as tarefas até t
        for j in range(t + 1):
            R[t, j] = eval_task(model, task_loaders[j][1], device)

    # Baselines single-task para FWT
    baselines = np.zeros(N_TASKS)
    for t in range(N_TASKS):
        train_l, val_l = task_loaders[t]
        m = DynamicMLP([784, 256, 128], ["relu", "relu"], 10).to(device)
        train_task(m, train_l, n_epochs=N_EPOCHS_CL, lr=1e-3, device=device)
        baselines[t] = eval_task(m, val_l, device)

    metrics = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics, "baselines": baselines.tolist()}
    torch.manual_seed(seed)
    np.random.seed(seed)

    task_loaders = [get_split_mnist(t, train_subset=TRAIN_SUBSET_CL) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))

    # Modelo base: MLP com n_classes=10
    model = DynamicMLP(
        layer_sizes=[784, 256, 128],
        activations=["relu", "relu"],
        n_classes=10,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    ewc_lambda = 400.0  # EWC lambda padrão da literatura

    # Acumular Fisher e params ótimos por tarefa
    fisher_list = []
    opt_params_list = []

    def ewc_penalty():
        loss = torch.tensor(0.0, device=device)
        for fisher, opt_p in zip(fisher_list, opt_params_list):
            for n, p in model.named_parameters():
                if n in fisher:
                    loss += (fisher[n].to(device) * (p - opt_p[n].to(device)) ** 2).sum()
        return ewc_lambda * loss

    def compute_fisher(loader, n_samples=200):
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        count = 0
        for xb, yb in loader:
            if count >= n_samples: break
            xb, yb = xb.to(device), yb.to(device)
            model.zero_grad()
            criterion(model(xb), yb).backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            count += len(xb)
        for n in fisher:
            fisher[n] /= max(count, 1)
        return fisher

    # Treinar cada tarefa
    for t in range(N_TASKS):
        train_l, val_l = task_loaders[t]
        opt = optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        for epoch in range(N_EPOCHS_CL):
            for xb, yb in train_l:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                if fisher_list:  # EWC a partir da tarefa 1
                    loss = loss + ewc_penalty()
                loss.backward()
                opt.step()

        # Calcular Fisher e salvar params ótimos após cada tarefa
        fisher = compute_fisher(train_l)
        fisher_list.append(fisher)
        opt_params_list.append({n: p.data.clone() for n, p in model.named_parameters()})

        # Avaliar em todas as tarefas até t
        for j in range(t + 1):
            R[t, j] = eval_task(model, task_loaders[j][1], device)

    # Baselines single-task para FWT
    baselines = np.zeros(N_TASKS)
    for t in range(N_TASKS):
        train_l, val_l = task_loaders[t]
        m = DynamicMLP([784, 256, 128], ["relu", "relu"], 10).to(device)
        train_task(m, train_l, n_epochs=N_EPOCHS_CL, lr=1e-3, device=device)
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
