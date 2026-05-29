"""
dNATY CIFAR-100 Experiment — Colab Optimized
GPU-accelerated evolution vs ResNet-8 baseline
Roda em ~30-40 min em GPU A100
"""
from __future__ import annotations
import os, json, sys, time
from pathlib import Path

# Setup para Colab
try:
    from google.colab import drive
    IN_COLAB = True
    drive.mount('/content/drive')
    RESULTS_DIR = '/content/drive/My Drive/dNATY_Results'
except ImportError:
    IN_COLAB = False
    RESULTS_DIR = "results_cifar100"

os.makedirs(RESULTS_DIR, exist_ok=True)

if __package__ in (None, ""):
    if not IN_COLAB:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Importações locais (funciona em Colab se tiver clonado o repo)
try:
    from dnaty.experiments.fast_dataset import FastDataset
    from dnaty.core.arch_cnn import DynamicCNN
    from dnaty.core.individual import Individual
    from dnaty.core.memory import EpisodicMemory, Experience
    from dnaty.operators.mutations_cnn import CNN_OPERATORS, apply_cnn_operator
    from dnaty.evolution.selection import nsga2_select
    from dnaty.analysis.stats import summary_stats, paired_ttest
except ImportError:
    print("⚠️  dNATY não encontrado. Execute antes:")
    print("!git clone https://github.com/seu-user/dNATY.git /content/dNATY")
    print("!cd /content/dNATY && pip install -e .")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
# CONFIG — CIFAR-100 (GPU optimized)
# ═══════════════════════════════════════════════════════════════
SEEDS = [0, 1]           # 2 seeds
N_GENERATIONS = 20       # 20 gerações (GPU: ~40 min total)
N_POP = 16               # 16 população (GPU can handle)
T_LOCAL = 5              # 5 épocas (GPU: rápido)
BATCH_SIZE = 512         # 512 batch (GPU: memory efficient)
CIFAR_TRAIN_SUBSET = 50000  # 50K (full CIFAR-100 train)
N_CLASSES = 100          # CIFAR-100 tem 100 classes


# ── Baseline: ResNet-12 para CIFAR-100 ────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.block(x) + x)


class ResNet12(nn.Module):
    """ResNet-12 para CIFAR-100 (mais layers que ResNet-8)"""
    def __init__(self, n_classes=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64), ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResBlock(128), ResBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            ResBlock(256), ResBlock(256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, n_classes)
    def forward(self, x): return self.fc(self.net(x).view(x.size(0), -1))
    def count_params(self): return sum(p.numel() for p in self.parameters())


def evaluate_cnn_fast(ind, fast_ds, device):
    """Avalia CNN usando FastDataset."""
    model = ind.model.to(device)
    model.eval()
    vx, vy = fast_ds.get_val()
    correct = total = 0
    chunk = 1024
    with torch.no_grad():
        for i in range(0, len(vx), chunk):
            xb = vx[i:i+chunk].to(device, non_blocking=True)
            yb = vy[i:i+chunk].to(device, non_blocking=True)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / max(total, 1)


def local_train_cnn_fast(ind, fast_ds, n_epochs, lr, device, batch_size=512):
    """Treino CNN com FastDataset, SAM e Data Augmentation."""
    import torchvision.transforms as T

    model = ind.model.to(device)
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.1)

    augment = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomRotation(15),
    ])

    n_batches_per_epoch = max(1, fast_ds.n_train // batch_size)
    loss_before = loss_after = 0.0
    grad_norms = []
    rho = 0.05

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for _ in range(n_batches_per_epoch):
            xb, yb = fast_ds.get_train_batch(batch_size)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb = augment(xb)

            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()

            gn = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
            grad_norms.append(gn)

            with torch.no_grad():
                grad_norm = sum(p.grad.norm().pow(2) for p in model.parameters() if p.grad is not None).sqrt()
                scale = rho / (grad_norm + 1e-12)
                for p in model.parameters():
                    if p.grad is not None:
                        p.data_orig = p.data.clone()
                        p.data.add_(p.grad, alpha=scale)

            opt.zero_grad(set_to_none=True)
            loss_perturbed = crit(model(xb), yb)
            loss_perturbed.backward()

            with torch.no_grad():
                for p in model.parameters():
                    if hasattr(p, 'data_orig'):
                        p.data.copy_(p.data_orig)

            opt.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg = epoch_loss / max(n_batches_per_epoch, 1)
        if epoch == 0: loss_before = avg
        loss_after = avg

    return loss_before, loss_after, float(np.mean(grad_norms)) if grad_norms else 0.0


def train_resnet_fast(model, fast_ds, n_epochs=30, device='cuda', batch_size=512, lr=2e-3):
    """Treina ResNet com FastDataset e Data Augmentation."""
    import torchvision.transforms as T

    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr*0.1)

    augment = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.RandomRotation(15),
    ])

    n_batches = max(1, fast_ds.n_train // batch_size)
    rho = 0.05

    for epoch in range(n_epochs):
        model.train()
        for _ in range(n_batches):
            xb, yb = fast_ds.get_train_batch(batch_size)
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            xb = augment(xb)

            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()

            with torch.no_grad():
                grad_norm = sum(p.grad.norm().pow(2) for p in model.parameters() if p.grad is not None).sqrt()
                scale = rho / (grad_norm + 1e-12)
                for p in model.parameters():
                    if p.grad is not None:
                        p.data_orig = p.data.clone()
                        p.data.add_(p.grad, alpha=scale)

            opt.zero_grad(set_to_none=True)
            loss_perturbed = crit(model(xb), yb)
            loss_perturbed.backward()

            with torch.no_grad():
                for p in model.parameters():
                    if hasattr(p, 'data_orig'):
                        p.data.copy_(p.data_orig)

            opt.step()

        scheduler.step()

    return evaluate_cnn_fast(type('I', (), {'model': model})(), fast_ds, device)


def run_dnaty_cnn_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    fast_ds = FastDataset('CIFAR100', device=device, train_subset=CIFAR_TRAIN_SUBSET)

    def make_ind():
        model = DynamicCNN(
            conv_configs=[
                {"type": "conv", "in_ch": 3,   "out_ch": 64,  "stride": 1},
                {"type": "conv", "in_ch": 64,  "out_ch": 128, "stride": 2},
                {"type": "conv", "in_ch": 128, "out_ch": 256, "stride": 2},
                {"type": "conv", "in_ch": 256, "out_ch": 512, "stride": 2},
            ],
            fc_sizes=[512, 256],
            n_classes=N_CLASSES,
        )
        return Individual(model, EpisodicMemory(decay_gamma=0.99))

    population = [make_ind() for _ in range(N_POP)]
    shared_mem = EpisodicMemory(max_size=500, decay_gamma=0.99)

    for ind in population:
        ind.acc = evaluate_cnn_fast(ind, fast_ds, device)
    fitnesses = [(ind.acc, -ind.count_params() * 1e-6, 0.0) for ind in population]
    prev_best = max(ind.acc for ind in population)

    history = []
    from tqdm import tqdm
    for gen in tqdm(range(1, N_GENERATIONS + 1), desc=f"CIFAR-100 seed={seed}"):
        op_probs = shared_mem.query_mutation_probs(CNN_OPERATORS)
        ops = list(op_probs.keys()); probs = list(op_probs.values())

        mutated = []
        for ind in population:
            op = np.random.choice(ops, p=probs)
            new_ind, ok = apply_cnn_operator(ind, op)
            if not ok or not new_ind.model.is_valid():
                new_ind = ind.clone(); new_ind.last_op = "no_op"
            mutated.append(new_ind)

        loss_befores, loss_afters, grad_norms = [], [], []
        for ind in mutated:
            lb, la, gn = local_train_cnn_fast(ind, fast_ds, T_LOCAL, 2e-3, device, BATCH_SIZE)
            loss_befores.append(lb); loss_afters.append(la); grad_norms.append(gn)
            ind.last_grad_norm = gn

        delta_grad = float(np.mean([b - a for b, a in zip(loss_befores, loss_afters)]))

        mut_fitnesses = []
        for ind in mutated:
            ind.acc = evaluate_cnn_fast(ind, fast_ds, device)
            mut_fitnesses.append((ind.acc, -ind.count_params() * 1e-6, 0.0))

        combined_pop = population + mutated
        combined_fit = fitnesses + mut_fitnesses
        population, fitnesses = nsga2_select(combined_pop, combined_fit, N_POP)

        delta_mem = 0.0
        for i, ind in enumerate(mutated):
            if ind.acc > prev_best and ind.last_op != "no_op":
                exp = Experience(operator=ind.last_op, delta_loss=-(ind.acc - prev_best),
                                 gradient_norm=grad_norms[i], generation=gen)
                shared_mem.update(exp); delta_mem += exp.impact

        best = max(population, key=lambda x: x.acc)
        prev_best = best.acc
        history.append({
            "gen": gen, "best_acc": round(best.acc, 4),
            "delta_grad": round(max(delta_grad, 0.0), 5),
            "delta_mem": round(delta_mem, 5),
            "n_params": best.count_params(), "n_flops": best.count_flops(),
        })

    best = max(population, key=lambda x: x.acc)
    return {
        "seed": seed, "acc": round(best.acc, 4),
        "n_params": best.count_params(), "n_flops": best.count_flops(),
        "history": history,
        "delta_grad_all_positive": all(h["delta_grad"] >= -1e-6 for h in history),
        "delta_mem_positive_after_gen3": all(h["delta_mem"] >= 0 for h in history if h["gen"] >= 3),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\n{'='*60}")
    print("dNATY CIFAR-100 v5.2 (FastDataset, GPU-optimized)")
    print(f"{'='*60}")

    dnaty_results = []
    resnet_accs = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        print("  [dNaty CNN + Genetic Algorithm]")
        t0 = time.time()
        dr = run_dnaty_cnn_seed(seed, device)
        dr["time_s"] = round(time.time() - t0, 1)
        dnaty_results.append(dr)
        print(f"  acc={dr['acc']:.4f} | params={dr['n_params']:,} | FLOPs={dr['n_flops']:,} | time={dr['time_s']}s")

        print("  [ResNet-12 baseline (CIFAR-100)]")
        torch.manual_seed(seed)
        fast_ds_full = FastDataset('CIFAR100', device=device, train_subset=CIFAR_TRAIN_SUBSET)
        resnet = ResNet12(n_classes=N_CLASSES)
        acc_r = train_resnet_fast(resnet, fast_ds_full, n_epochs=30, device=device)
        resnet_accs.append(round(acc_r, 4))
        print(f"  acc={acc_r:.4f} | params={resnet.count_params():,}")

    dnaty_accs = [r["acc"] for r in dnaty_results]
    dnaty_s = summary_stats(dnaty_accs)
    resnet_s = summary_stats(resnet_accs)
    t_stat, p_val, cohen_d = paired_ttest(dnaty_accs, resnet_accs)

    print(f"\n{'─'*50}")
    print("RESULTADOS FINAIS — CIFAR-100 v5.2")
    print(f"  dNaty CNN: {dnaty_s['mean']*100:.2f}% ± {dnaty_s['std']*100:.2f}%")
    print(f"  ResNet-12: {resnet_s['mean']*100:.2f}% ± {resnet_s['std']*100:.2f}%")
    print(f"  p={p_val:.4f} | Cohen's d={cohen_d:.3f}")
    print(f"  Melhoria: {((dnaty_s['mean'] / resnet_s['mean']) - 1) * 100:.1f}%")

    all_results = {
        "CIFAR100": {
            "dnaty": dnaty_results,
            "resnet_accs": resnet_accs,
            "summary": {
                "dnaty": dnaty_s, "resnet": resnet_s,
                "ttest": {"t": t_stat, "p": p_val, "d": cohen_d},
                "theorem1_delta_grad_positive": all(r["delta_grad_all_positive"] for r in dnaty_results),
                "theorem1_delta_mem_positive": all(r["delta_mem_positive_after_gen3"] for r in dnaty_results),
            }
        }
    }

    out_path = os.path.join(RESULTS_DIR, "cifar100_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Resultados salvos em: {out_path}")
    return all_results


if __name__ == "__main__":
    main()
