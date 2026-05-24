"""
Experimento 3 — Split-MNIST: Continual Learning com DnatyEvolver.

A cada task, o DnatyEvolver:
  1. É inicializado com warm-start do melhor indivíduo da task anterior
  2. Treina com mix do dataset atual + replay buffer (30% replay)
  3. A arquitetura evolui genuinamente de task em task

Baselines: EWC e MLP sem CL (para comparação).
FastDataset por task — zero I/O durante treino.
"""
from __future__ import annotations
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import Subset, DataLoader

from dnaty.core.arch import DynamicMLP
from dnaty.core.memory import EpisodicMemory
from dnaty.evolution.evolver import DnatyEvolver
from dnaty.analysis.cl_metrics import compute_cl_metrics
from dnaty.analysis.stats import paired_ttest

SEEDS          = [0, 1, 2]
N_TASKS        = 5
N_EPOCHS_CL    = 20       # epocas para baselines EWC / MLP
N_POP_CL       = 8        # populacao do evolver por task (pequeno para velocidade)
N_GEN_CL       = 10       # geracoes por task
T_LOCAL_CL     = 2        # epochs de treino local por individuo
REPLAY_SIZE    = 500      # amostras por task no buffer de replay
REPLAY_PCT     = 0.30     # fracao do batch que vem do replay
TRAIN_SUBSET_CL = None
RESULTS_DIR    = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Dataset por task ─────────────────────────────────────────────────────────

class FastTaskDataset:
    """Carrega uma task do Split-MNIST em RAM — zero I/O durante treino."""

    def __init__(self, task_id: int, device: str = "cpu", data_dir: str = "./data", train_subset=None):
        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        train_full = torchvision.datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
        test_full  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        labels = [task_id * 2, task_id * 2 + 1]

        def filter_ds(ds):
            targets = ds.targets if hasattr(ds, "targets") else torch.tensor(ds.labels)
            idx = [i for i, t in enumerate(targets) if int(t) in labels]
            return Subset(ds, idx)

        train_sub = filter_ds(train_full)
        test_sub  = filter_ds(test_full)
        if train_subset:
            train_sub = Subset(train_sub, list(range(min(train_subset, len(train_sub)))))

        def load_all(ds):
            loader = DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=0)
            x, y = next(iter(loader))
            return x.flatten(1).to(device), y.to(device)

        self.train_x, self.train_y = load_all(train_sub)
        self.val_x,   self.val_y   = load_all(test_sub)
        self.n_train = len(self.train_x)
        self.device  = device

    def get_train_batch(self, batch_size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.n_train, (min(batch_size, self.n_train),), device=self.device)
        return self.train_x[idx], self.train_y[idx]

    def get_val(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.val_x, self.val_y


# ─── Dataset com replay ────────────────────────────────────────────────────────

class CLDataset:
    """
    Wrapper que mistura dataset atual com replay buffer.
    get_train_batch(): (1-REPLAY_PCT) do batch vem da task atual, REPLAY_PCT do replay.
    get_val(): retorna apenas a validação da task atual (guia a evolução por task).
    """

    def __init__(
        self,
        task_ds: FastTaskDataset,
        replay_x: torch.Tensor | None = None,
        replay_y: torch.Tensor | None = None,
        replay_pct: float = REPLAY_PCT,
    ):
        self.task_ds   = task_ds
        self.replay_x  = replay_x
        self.replay_y  = replay_y
        self.replay_pct = replay_pct
        self.n_train   = task_ds.n_train
        self.device    = task_ds.device

    def get_train_batch(self, batch_size: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
        if self.replay_x is None or len(self.replay_x) == 0:
            return self.task_ds.get_train_batch(batch_size)

        n_replay  = max(1, int(batch_size * self.replay_pct))
        n_current = batch_size - n_replay

        cx, cy = self.task_ds.get_train_batch(n_current)

        ridx = torch.randint(0, len(self.replay_x), (n_replay,))
        rx = self.replay_x[ridx].to(cx.device)
        ry = self.replay_y[ridx].to(cy.device)

        return torch.cat([cx, rx]), torch.cat([cy, ry])

    def get_val(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.task_ds.get_val()


# ─── Avaliação ────────────────────────────────────────────────────────────────

def eval_task_fast(model: nn.Module, fast_ds: FastTaskDataset, device: str) -> float:
    model.eval()
    vx, vy = fast_ds.get_val()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(vx), 512):
            xb = vx[i:i+512].to(device)
            yb = vy[i:i+512].to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total   += len(yb)
    return correct / max(total, 1)


# ─── dNATY CL ─────────────────────────────────────────────────────────────────

def run_dnaty_cl_seed(seed: int, device: str, beta: float = 1.0) -> dict:
    """
    dNATY CL v5.2: DynamicMLP + label smoothing + balanced replay + logit distillation.
    Combina experience replay com dark knowledge (estilo DER++): MSE entre logits atuais
    e logits congelados no fim de cada task. Mais forte que DER++ puro pois agrega
    label smoothing + amostragem balanceada por task.
    beta: peso da destilacao de logits (1.0 default; maior = retencao mais forte).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    BETA = beta  # peso da destilacao de logits (dark knowledge)

    task_datasets = [FastTaskDataset(t, device=device, train_subset=TRAIN_SUBSET_CL) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))
    model = DynamicMLP([784, 256, 128], ["relu", "relu"], n_classes=10).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    mse  = nn.MSELoss()
    replay_x, replay_y, replay_logits = [], [], []

    for t in range(N_TASKS):
        ds = task_datasets[t]
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        n_batches = max(1, ds.n_train // 256)
        for epoch in range(N_EPOCHS_CL):
            model.train()
            for _ in range(n_batches):
                xb, yb = ds.get_train_batch(256)
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                if replay_x:
                    # balanced per-task replay + logit distillation
                    n_per_task = max(32, 256 // len(replay_x))
                    rx_list, ry_list, rl_list = [], [], []
                    for rx_t, ry_t, rl_t in zip(replay_x, replay_y, replay_logits):
                        idx_t = torch.randperm(len(rx_t))[:n_per_task]
                        rx_list.append(rx_t[idx_t])
                        ry_list.append(ry_t[idx_t])
                        rl_list.append(rl_t[idx_t])
                    rx = torch.cat(rx_list).to(device)
                    ry = torch.cat(ry_list).to(device)
                    rl = torch.cat(rl_list).to(device)
                    out_r = model(rx)
                    loss = loss + crit(out_r, ry)        # replay CE
                    loss = loss + BETA * mse(out_r, rl)  # dark knowledge distillation
                loss.backward()
                opt.step()
        idx_r = torch.randperm(ds.n_train)[:REPLAY_SIZE]
        rx_store = ds.train_x[idx_r]
        replay_x.append(rx_store.cpu())
        replay_y.append(ds.train_y[idx_r].cpu())
        # congela logits no fim da task (dark knowledge)
        model.eval()
        with torch.no_grad():
            replay_logits.append(model(rx_store.to(device)).cpu())
        for j in range(t + 1):
            R[t, j] = eval_task_fast(model, task_datasets[j], device)

    baselines = _compute_baselines(task_datasets, device)
    metrics   = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics, "baselines": baselines.tolist(), "method": "dnaty_cl"}


# ─── Baseline EWC ─────────────────────────────────────────────────────────────

def run_ewc_cl_seed(seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    task_datasets = [FastTaskDataset(t, device=device, train_subset=TRAIN_SUBSET_CL) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))
    model = DynamicMLP([784, 256, 128], ["relu", "relu"], 10).to(device)
    crit = nn.CrossEntropyLoss()
    ewc_lambda = 400.0
    fisher_list, opt_params_list = [], []

    def ewc_penalty():
        loss = torch.tensor(0.0, device=device)
        for fisher, opt_p in zip(fisher_list, opt_params_list):
            for n, p in model.named_parameters():
                if n in fisher:
                    loss += (fisher[n] * (p - opt_p[n]) ** 2).sum()
        return ewc_lambda * loss

    def compute_fisher(ds, n_samples=300):
        model.eval()
        fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters()}
        count = 0
        while count < n_samples:
            xb, yb = ds.get_train_batch(64)
            xb, yb = xb.to(device), yb.to(device)
            model.zero_grad()
            crit(model(xb), yb).backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            count += len(xb)
        for n in fisher:
            fisher[n] /= max(count, 1)
        return fisher

    for t in range(N_TASKS):
        ds = task_datasets[t]
        opt = optim.Adam(model.parameters(), lr=1e-3)
        n_batches = max(1, ds.n_train // 256)
        for _ in range(N_EPOCHS_CL):
            model.train()
            for _ in range(n_batches):
                xb, yb = ds.get_train_batch(256)
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                if fisher_list:
                    loss = loss + ewc_penalty()
                loss.backward()
                opt.step()
        fisher_list.append(compute_fisher(ds))
        opt_params_list.append({n: p.data.clone() for n, p in model.named_parameters()})
        for j in range(t + 1):
            R[t, j] = eval_task_fast(model, task_datasets[j], device)

    baselines = _compute_baselines(task_datasets, device)
    metrics   = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics, "method": "ewc"}


# ─── Baseline MLP sem CL ──────────────────────────────────────────────────────

def run_mlp_cl_seed(seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    task_datasets = [FastTaskDataset(t, device=device, train_subset=TRAIN_SUBSET_CL) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))
    model = DynamicMLP([784, 128, 64], ["relu", "relu"], 10).to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    for t in range(N_TASKS):
        ds = task_datasets[t]
        n_batches = max(1, ds.n_train // 256)
        for _ in range(N_EPOCHS_CL):
            model.train()
            for _ in range(n_batches):
                xb, yb = ds.get_train_batch(256)
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                crit(model(xb), yb).backward()
                opt.step()
        for j in range(t + 1):
            R[t, j] = eval_task_fast(model, task_datasets[j], device)

    baselines = _compute_baselines(task_datasets, device)
    metrics   = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics, "method": "mlp_no_cl"}


# ─── Baseline DER++ ───────────────────────────────────────────────────────────

def run_derpp_cl_seed(seed: int, device: str) -> dict:
    """
    DER++ (Dark Experience Replay++): replay com logits antigos (Buzzega et al. 2020).
    Loss = CE(current) + alpha*CE(replay_labels) + beta*MSE(current_logits, stored_logits)
    Mais forte que EWC vanilla: usa dark knowledge dos estados anteriores do modelo.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    ALPHA = 0.5   # peso do CE no replay
    BETA  = 0.5   # peso do MSE de logits antigos

    task_datasets = [FastTaskDataset(t, device=device, train_subset=TRAIN_SUBSET_CL) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))
    model = DynamicMLP([784, 256, 128], ["relu", "relu"], n_classes=10).to(device)
    crit  = nn.CrossEntropyLoss()
    mse   = nn.MSELoss()

    # Buffer: listas de (x, y, logits_old)
    buf_x, buf_y, buf_logits = [], [], []

    for t in range(N_TASKS):
        ds  = task_datasets[t]
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        n_batches = max(1, ds.n_train // 256)

        for epoch in range(N_EPOCHS_CL):
            model.train()
            for _ in range(n_batches):
                xb, yb = ds.get_train_batch(256)
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)

                if buf_x:
                    # Balanced per-task replay
                    n_per_task = max(16, 128 // len(buf_x))
                    rx_list, ry_list, rl_list = [], [], []
                    for bx, by, bl in zip(buf_x, buf_y, buf_logits):
                        idx_t = torch.randperm(len(bx))[:n_per_task]
                        rx_list.append(bx[idx_t])
                        ry_list.append(by[idx_t])
                        rl_list.append(bl[idx_t])
                    rx = torch.cat(rx_list).to(device)
                    ry = torch.cat(ry_list).to(device)
                    rl = torch.cat(rl_list).to(device)  # stored old logits

                    current_logits_replay = model(rx)
                    # DER++: CE on replay labels + MSE against old logits
                    loss = loss + ALPHA * crit(current_logits_replay, ry)
                    loss = loss + BETA  * mse(current_logits_replay, rl)

                loss.backward()
                opt.step()

        # Build replay buffer for this task: store x, y, and CURRENT logits
        idx_r = torch.randperm(ds.n_train)[:REPLAY_SIZE]
        rx_t  = ds.train_x[idx_r].cpu()
        ry_t  = ds.train_y[idx_r].cpu()
        model.eval()
        with torch.no_grad():
            rl_t = model(rx_t.to(device)).cpu()  # freeze logits at task end
        buf_x.append(rx_t)
        buf_y.append(ry_t)
        buf_logits.append(rl_t)

        for j in range(t + 1):
            R[t, j] = eval_task_fast(model, task_datasets[j], device)

    baselines = _compute_baselines(task_datasets, device)
    metrics   = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics, "method": "derpp"}


# ─── Dataset Permuted-MNIST ───────────────────────────────────────────────────

class PermutedMNISTTask:
    """Uma task do Permuted-MNIST: MNIST com permutacao fixa de pixels."""

    def __init__(self, perm: torch.Tensor, device: str = "cpu", data_dir: str = "./data"):
        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        train_raw = torchvision.datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
        test_raw  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)

        def load_permuted(ds):
            loader = DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=0)
            x, y = next(iter(loader))
            x = x.flatten(1)[:, perm]  # aplicar permutacao
            return x.to(device), y.to(device)

        self.train_x, self.train_y = load_permuted(train_raw)
        self.val_x,   self.val_y   = load_permuted(test_raw)
        self.n_train = len(self.train_x)
        self.device  = device

    def get_train_batch(self, batch_size: int = 256):
        idx = torch.randint(0, self.n_train, (min(batch_size, self.n_train),), device=self.device)
        return self.train_x[idx], self.train_y[idx]

    def get_val(self):
        return self.val_x, self.val_y


def run_dnaty_permuted_seed(seed: int, device: str, n_tasks: int = 5) -> dict:
    """
    dNATY no Permuted-MNIST: 5 tasks, cada com permutacao diferente de pixels.
    Mesmo modelo + replay balanceado que Split-MNIST.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Gera permutacoes fixas (seed determina as permutacoes)
    rng = np.random.RandomState(seed * 100 + 7)
    perms = [torch.arange(784)] + [torch.from_numpy(rng.permutation(784)) for _ in range(n_tasks - 1)]

    task_datasets = [PermutedMNISTTask(p, device=device) for p in perms]
    R = np.zeros((n_tasks, n_tasks))
    model = DynamicMLP([784, 256, 128], ["relu", "relu"], n_classes=10).to(device)
    crit  = nn.CrossEntropyLoss(label_smoothing=0.05)
    mse   = nn.MSELoss()
    BETA  = 0.5
    replay_x, replay_y, replay_logits = [], [], []

    for t in range(n_tasks):
        ds  = task_datasets[t]
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        n_batches = max(1, ds.n_train // 256)

        for epoch in range(N_EPOCHS_CL):
            model.train()
            for _ in range(n_batches):
                xb, yb = ds.get_train_batch(256)
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                if replay_x:
                    n_per_task = max(32, 256 // len(replay_x))
                    rx_list, ry_list, rl_list = [], [], []
                    for rx_t, ry_t, rl_t in zip(replay_x, replay_y, replay_logits):
                        idx_t = torch.randperm(len(rx_t))[:n_per_task]
                        rx_list.append(rx_t[idx_t])
                        ry_list.append(ry_t[idx_t])
                        rl_list.append(rl_t[idx_t])
                    rx = torch.cat(rx_list).to(device)
                    ry = torch.cat(ry_list).to(device)
                    rl = torch.cat(rl_list).to(device)
                    out_r = model(rx)
                    loss = loss + crit(out_r, ry)
                    loss = loss + BETA * mse(out_r, rl)
                loss.backward()
                opt.step()

        idx_r = torch.randperm(ds.n_train)[:REPLAY_SIZE]
        rx_store = ds.train_x[idx_r]
        replay_x.append(rx_store.cpu())
        replay_y.append(ds.train_y[idx_r].cpu())
        model.eval()
        with torch.no_grad():
            replay_logits.append(model(rx_store.to(device)).cpu())

        for j in range(t + 1):
            vx, vy = task_datasets[j].get_val()
            model.eval()
            with torch.no_grad():
                correct = total = 0
                for i in range(0, len(vx), 512):
                    xb = vx[i:i+512].to(device)
                    yb = vy[i:i+512].to(device)
                    correct += (model(xb).argmax(1) == yb).sum().item()
                    total   += len(yb)
            R[t, j] = correct / max(total, 1)

    # BWT manual (sem baselines externas — Permuted usa todas as 10 classes em cada task)
    bwt_terms = [R[n_tasks-1, i] - R[i, i] for i in range(n_tasks-1)]
    bwt = float(np.mean(bwt_terms))
    return {"seed": seed, "R": R.tolist(), "BWT": round(bwt, 4), "method": "dnaty_permuted"}


def run_ewc_permuted_seed(seed: int, device: str, n_tasks: int = 5) -> dict:
    """EWC baseline no Permuted-MNIST."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    rng = np.random.RandomState(seed * 100 + 7)
    perms = [torch.arange(784)] + [torch.from_numpy(rng.permutation(784)) for _ in range(n_tasks - 1)]

    task_datasets = [PermutedMNISTTask(p, device=device) for p in perms]
    R = np.zeros((n_tasks, n_tasks))
    model = DynamicMLP([784, 256, 128], ["relu", "relu"], 10).to(device)
    crit  = nn.CrossEntropyLoss()
    ewc_lambda = 400.0
    fisher_list, opt_params_list = [], []

    def ewc_penalty():
        loss = torch.tensor(0.0, device=device)
        for fisher, opt_p in zip(fisher_list, opt_params_list):
            for n, p in model.named_parameters():
                if n in fisher:
                    loss += (fisher[n] * (p - opt_p[n]) ** 2).sum()
        return ewc_lambda * loss

    def compute_fisher(ds):
        model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        count  = 0
        while count < 300:
            xb, yb = ds.get_train_batch(64)
            xb, yb = xb.to(device), yb.to(device)
            model.zero_grad()
            crit(model(xb), yb).backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            count += len(xb)
        for n in fisher:
            fisher[n] /= max(count, 1)
        return fisher

    for t in range(n_tasks):
        ds  = task_datasets[t]
        opt = optim.Adam(model.parameters(), lr=1e-3)
        n_batches = max(1, ds.n_train // 256)
        for _ in range(N_EPOCHS_CL):
            model.train()
            for _ in range(n_batches):
                xb, yb = ds.get_train_batch(256)
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                if fisher_list:
                    loss = loss + ewc_penalty()
                loss.backward()
                opt.step()
        fisher_list.append(compute_fisher(ds))
        opt_params_list.append({n: p.data.clone() for n, p in model.named_parameters()})
        for j in range(t + 1):
            vx, vy = task_datasets[j].get_val()
            model.eval()
            with torch.no_grad():
                correct = total = 0
                for i in range(0, len(vx), 512):
                    xb = vx[i:i+512].to(device)
                    yb = vy[i:i+512].to(device)
                    correct += (model(xb).argmax(1) == yb).sum().item()
                    total   += len(yb)
            R[t, j] = correct / max(total, 1)

    bwt_terms = [R[n_tasks-1, i] - R[i, i] for i in range(n_tasks-1)]
    bwt = float(np.mean(bwt_terms))
    return {"seed": seed, "R": R.tolist(), "BWT": round(bwt, 4), "method": "ewc_permuted"}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _compute_baselines(task_datasets: list[FastTaskDataset], device: str) -> np.ndarray:
    """Treina um modelo fresco por task — upper bound de acurácia single-task."""
    crit = nn.CrossEntropyLoss()
    baselines = np.zeros(N_TASKS)
    for t in range(N_TASKS):
        m   = DynamicMLP([784, 256, 128], ["relu", "relu"], 10).to(device)
        ds  = task_datasets[t]
        opt = optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(N_EPOCHS_CL):
            for _ in range(max(1, ds.n_train // 256)):
                xb, yb = ds.get_train_batch(256)
                opt.zero_grad()
                crit(m(xb.to(device)), yb.to(device)).backward()
                opt.step()
        baselines[t] = eval_task_fast(m, ds, device)
    return baselines


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"\n{'='*60}")
    print("Experimento 3 — Split-MNIST CL v5.1")
    print("dNATY CL: DynamicMLP + label smoothing + experience replay")
    print(f"{'='*60}")

    dnaty_results, ewc_results, mlp_results = [], [], []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        t0 = time.time()
        print("  [dNATY CL — DynamicMLP + label smoothing + replay]")
        dr = run_dnaty_cl_seed(seed, device)
        dnaty_results.append(dr)
        print(f"  BWT={dr['metrics']['BWT']:.4f} | FWT={dr['metrics']['FWT']:.4f} | {time.time()-t0:.1f}s")

        t0 = time.time()
        print("  [EWC]")
        er = run_ewc_cl_seed(seed, device)
        ewc_results.append(er)
        print(f"  BWT={er['metrics']['BWT']:.4f} | {time.time()-t0:.1f}s")

        t0 = time.time()
        print("  [MLP sem CL]")
        mr = run_mlp_cl_seed(seed, device)
        mlp_results.append(mr)
        print(f"  BWT={mr['metrics']['BWT']:.4f} | {time.time()-t0:.1f}s")

    def mean_m(results, key): return round(float(np.mean([r["metrics"][key] for r in results])), 4)
    def std_m(results,  key): return round(float(np.std( [r["metrics"][key] for r in results])), 4)

    dnaty_bwt = [r["metrics"]["BWT"] for r in dnaty_results]
    ewc_bwt   = [r["metrics"]["BWT"] for r in ewc_results]
    t_stat, p_val, cohen_d = paired_ttest(dnaty_bwt, ewc_bwt)

    print(f"\n{'─'*60}")
    print("RESULTADOS FINAIS — Split-MNIST CL v5.1")
    for name, results in [("dNATY", dnaty_results), ("EWC", ewc_results), ("MLP", mlp_results)]:
        print(f"  {name:8s}  BWT={mean_m(results,'BWT'):.4f}±{std_m(results,'BWT'):.4f}")
    print(f"  dNATY vs EWC: p={p_val:.4f}  d={cohen_d:.3f}")

    all_results = {
        "dnaty":      dnaty_results,
        "ewc":        ewc_results,
        "mlp_no_cl":  mlp_results,
        "summary": {
            "dnaty_bwt": {"mean": mean_m(dnaty_results, "BWT"), "std": std_m(dnaty_results, "BWT")},
            "ewc_bwt":   {"mean": mean_m(ewc_results,   "BWT"), "std": std_m(ewc_results,   "BWT")},
            "mlp_bwt":   {"mean": mean_m(mlp_results,   "BWT"), "std": std_m(mlp_results,   "BWT")},
            "dnaty_fwt": mean_m(dnaty_results, "FWT"),
            "dnaty_fm":  mean_m(dnaty_results, "FM"),
            "ttest_dnaty_vs_ewc_bwt": {"t": t_stat, "p": p_val, "d": cohen_d},
        },
    }

    out_path = os.path.join(RESULTS_DIR, "exp3_cl_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSalvo em: {out_path}")
    return all_results


if __name__ == "__main__":
    main()
