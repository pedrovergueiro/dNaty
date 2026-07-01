"""
Experiment 5 -- Split-CIFAR-10: Continual Learning on a naturalistic image benchmark.

5 tasks. CIFAR-10 is split into 5 class pairs:
  Task 0: airplane, automobile  (0, 1)
  Task 1: bird, cat             (2, 3)
  Task 2: deer, dog             (4, 5)
  Task 3: frog, horse           (6, 7)
  Task 4: ship, truck           (8, 9)

Input: 32×32×3 flattened to 3072 dimensions (MLP). Class-incremental setting.
This benchmark is substantially harder than Split-MNIST:
  - naturalistic images (vs. handwritten digits)
  - 3072-dim input (vs. 784)
  - MLP without convolutional inductive bias

Goal: measure relative BWT (forgetting), not absolute accuracy. An MLP
trained on CIFAR without convolutions will have lower absolute accuracy than
a CNN -- that is expected and not the claim. The claim is that balanced replay
reduces forgetting compared to EWC and plain fine-tuning on this harder domain.

Methods: dNATY balanced replay, EWC (lambda=400), MLP baseline.
"""
from __future__ import annotations
import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from dnaty.analysis.cl_metrics import compute_cl_metrics
from dnaty.analysis.stats import paired_ttest
from dnaty.core.arch import DynamicMLP

SEEDS = [0, 1, 2]
N_TASKS = 5
CLASSES_PER_TASK = 2
N_EPOCHS_CL = 20
REPLAY_SIZE = 200
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TASK_CLASS_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
INPUT_DIM = 3072   # 32*32*3


class SplitCIFAR10Task:
    """Load one Split-CIFAR-10 task into RAM as flattened float32 tensors."""

    def __init__(self, task_id: int, device: str = "cpu", data_dir: str = "./data"):
        classes = TASK_CLASS_PAIRS[task_id]
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_full = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform)
        test_full  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

        def filter_ds(ds):
            targets = torch.tensor(ds.targets)
            idx = [i for i, t in enumerate(targets) if int(t) in classes]
            return Subset(ds, idx)

        def load_all(ds):
            loader = DataLoader(ds, batch_size=len(ds), shuffle=False, num_workers=0)
            x, y = next(iter(loader))
            x = x.flatten(1).float()        # (N, 3072)
            # remap class labels to {0, 1}
            y_mapped = torch.zeros_like(y)
            for new_label, orig_label in enumerate(classes):
                y_mapped[y == orig_label] = new_label
            return x.to(device), y_mapped.to(device)

        self.train_x, self.train_y = load_all(filter_ds(train_full))
        self.val_x,   self.val_y   = load_all(filter_ds(test_full))
        self.n_train = len(self.train_x)
        self.device  = device
        self.n_classes = CLASSES_PER_TASK

    def get_train_batch(self, batch_size: int = 256):
        idx = torch.randint(0, self.n_train, (min(batch_size, self.n_train),), device=self.device)
        return self.train_x[idx], self.train_y[idx]

    def get_val(self):
        return self.val_x, self.val_y


def eval_task(model: nn.Module, task: SplitCIFAR10Task, device: str) -> float:
    model.eval()
    vx, vy = task.get_val()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(vx), 256):
            xb, yb = vx[i:i+256].to(device), vy[i:i+256].to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / max(total, 1)


def _compute_baselines(tasks: list[SplitCIFAR10Task], crit: nn.Module,
                       n_epochs: int, device: str) -> np.ndarray:
    baselines = np.zeros(N_TASKS)
    for t, task in enumerate(tasks):
        m = DynamicMLP([INPUT_DIM, 512, 256], ["relu", "relu"], n_classes=CLASSES_PER_TASK).to(device)
        opt = optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(n_epochs):
            for _ in range(max(1, task.n_train // 256)):
                xb, yb = task.get_train_batch(256)
                opt.zero_grad()
                crit(m(xb), yb).backward()
                opt.step()
        baselines[t] = eval_task(m, task, device)
    return baselines


# ---------------------------------------------------------------------------
# dNATY -- balanced episodic replay
# ---------------------------------------------------------------------------

def run_dnaty_seed(seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tasks = [SplitCIFAR10Task(t, device=device) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))
    # Larger hidden layers for the harder CIFAR domain
    model = DynamicMLP([INPUT_DIM, 512, 256], ["relu", "relu"], n_classes=CLASSES_PER_TASK).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    replay_x: list[torch.Tensor] = []
    replay_y: list[torch.Tensor] = []

    for t, task in enumerate(tasks):
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        n_batches = max(1, task.n_train // 256)
        for _ in range(N_EPOCHS_CL):
            model.train()
            for _ in range(n_batches):
                xb, yb = task.get_train_batch(256)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                if replay_x:
                    rx = torch.cat(replay_x).to(device)
                    ry = torch.cat(replay_y).to(device)
                    idx_r = torch.randperm(len(rx))[:min(128, len(rx))]
                    loss = loss + crit(model(rx[idx_r]), ry[idx_r])
                loss.backward()
                opt.step()
        idx_r = torch.randperm(task.n_train)[:REPLAY_SIZE]
        replay_x.append(task.train_x[idx_r].cpu())
        replay_y.append(task.train_y[idx_r].cpu())
        for j in range(t + 1):
            R[t, j] = eval_task(model, tasks[j], device)

    baselines = _compute_baselines(tasks, crit, N_EPOCHS_CL, device)
    metrics = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics, "baselines": baselines.tolist()}


# ---------------------------------------------------------------------------
# EWC baseline
# ---------------------------------------------------------------------------

def run_ewc_seed(seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tasks = [SplitCIFAR10Task(t, device=device) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))
    model = DynamicMLP([INPUT_DIM, 512, 256], ["relu", "relu"], n_classes=CLASSES_PER_TASK).to(device)
    crit = nn.CrossEntropyLoss()
    ewc_lambda = 400.0
    fisher_list: list[dict] = []
    opt_params_list: list[dict] = []

    def ewc_penalty() -> torch.Tensor:
        loss = torch.tensor(0.0, device=device)
        for fisher, opt_p in zip(fisher_list, opt_params_list):
            for n, p in model.named_parameters():
                if n in fisher:
                    loss += (fisher[n] * (p - opt_p[n]) ** 2).sum()
        return ewc_lambda * loss

    def compute_fisher(task: SplitCIFAR10Task, n_samples: int = 300) -> dict:
        model.eval()
        fisher = {n: torch.zeros_like(p, device=device) for n, p in model.named_parameters()}
        count = 0
        while count < n_samples:
            xb, yb = task.get_train_batch(64)
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

    for t, task in enumerate(tasks):
        opt = optim.Adam(model.parameters(), lr=1e-3)
        n_batches = max(1, task.n_train // 256)
        for _ in range(N_EPOCHS_CL):
            model.train()
            for _ in range(n_batches):
                xb, yb = task.get_train_batch(256)
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                if fisher_list:
                    loss = loss + ewc_penalty()
                loss.backward()
                opt.step()
        fisher_list.append(compute_fisher(task))
        opt_params_list.append({n: p.data.clone() for n, p in model.named_parameters()})
        for j in range(t + 1):
            R[t, j] = eval_task(model, tasks[j], device)

    baselines = _compute_baselines(tasks, crit, N_EPOCHS_CL, device)
    metrics = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics, "baselines": baselines.tolist()}


# ---------------------------------------------------------------------------
# MLP baseline (no CL)
# ---------------------------------------------------------------------------

def run_mlp_seed(seed: int, device: str) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tasks = [SplitCIFAR10Task(t, device=device) for t in range(N_TASKS)]
    R = np.zeros((N_TASKS, N_TASKS))
    model = DynamicMLP([INPUT_DIM, 512, 256], ["relu", "relu"], n_classes=CLASSES_PER_TASK).to(device)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)

    for t, task in enumerate(tasks):
        n_batches = max(1, task.n_train // 256)
        for _ in range(N_EPOCHS_CL):
            model.train()
            for _ in range(n_batches):
                xb, yb = task.get_train_batch(256)
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                crit(model(xb), yb).backward()
                opt.step()
        for j in range(t + 1):
            R[t, j] = eval_task(model, tasks[j], device)

    baselines = _compute_baselines(tasks, crit, N_EPOCHS_CL, device)
    metrics = compute_cl_metrics(R, baselines)
    return {"seed": seed, "R": R.tolist(), "metrics": metrics, "baselines": baselines.tolist()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"\n{'='*60}")
    print("Experiment 5 -- Split-CIFAR-10 (5 tasks, class-incremental)")
    print("NOTE: MLP on CIFAR; absolute accuracy is lower than CNN baseline.")
    print("Metric of interest: relative BWT (forgetting), not absolute accuracy.")
    print(f"{'='*60}")

    dnaty_results, ewc_results, mlp_results = [], [], []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        t0 = time.time()
        print("  [dNATY CL]")
        dr = run_dnaty_seed(seed, device)
        dnaty_results.append(dr)
        print(f"  BWT={dr['metrics']['BWT']:.4f} | FWT={dr['metrics']['FWT']:.4f} | {time.time()-t0:.1f}s")

        t0 = time.time()
        print("  [EWC]")
        er = run_ewc_seed(seed, device)
        ewc_results.append(er)
        print(f"  BWT={er['metrics']['BWT']:.4f} | {time.time()-t0:.1f}s")

        t0 = time.time()
        print("  [MLP no CL]")
        mr = run_mlp_seed(seed, device)
        mlp_results.append(mr)
        print(f"  BWT={mr['metrics']['BWT']:.4f} | {time.time()-t0:.1f}s")

    def mean_m(results, key): return round(float(np.mean([r["metrics"][key] for r in results])), 4)
    def std_m(results, key):  return round(float(np.std( [r["metrics"][key] for r in results])), 4)

    dnaty_bwt = [r["metrics"]["BWT"] for r in dnaty_results]
    ewc_bwt   = [r["metrics"]["BWT"] for r in ewc_results]
    t_stat, p_val, cohen_d = paired_ttest(dnaty_bwt, ewc_bwt)

    print(f"\n{'-'*50}")
    print("FINAL RESULTS -- Split-CIFAR-10 (5 tasks)")
    for name, results in [("dNATY", dnaty_results), ("EWC", ewc_results), ("MLP", mlp_results)]:
        print(f"  {name:8s} BWT={mean_m(results,'BWT'):.4f}+/-{std_m(results,'BWT'):.4f}  "
              f"FWT={mean_m(results,'FWT'):.4f}  FM={mean_m(results,'FM'):.4f}")
    print(f"  dNATY vs EWC: p={p_val:.4f} d={cohen_d:.3f}")

    all_results = {
        "experiment": "Split-CIFAR-10 (5 tasks, class-incremental, MLP)",
        "note": "Absolute accuracy reflects MLP without conv. Metric is BWT (forgetting), not absolute accuracy.",
        "task_class_pairs": TASK_CLASS_PAIRS,
        "n_epochs": N_EPOCHS_CL,
        "replay_size": REPLAY_SIZE,
        "dnaty": dnaty_results,
        "ewc": ewc_results,
        "mlp_no_cl": mlp_results,
        "summary": {
            "dnaty_bwt": {"mean": mean_m(dnaty_results, "BWT"), "std": std_m(dnaty_results, "BWT")},
            "dnaty_fwt": mean_m(dnaty_results, "FWT"),
            "dnaty_fm":  mean_m(dnaty_results, "FM"),
            "ewc_bwt":   {"mean": mean_m(ewc_results,   "BWT"), "std": std_m(ewc_results,   "BWT")},
            "mlp_bwt":   {"mean": mean_m(mlp_results,   "BWT"), "std": std_m(mlp_results,   "BWT")},
            "ttest_dnaty_vs_ewc_bwt": {"t": t_stat, "p": p_val, "d": cohen_d},
        },
    }

    out_path = os.path.join(RESULTS_DIR, "exp5_split_cifar10_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")
    return all_results


if __name__ == "__main__":
    main()
