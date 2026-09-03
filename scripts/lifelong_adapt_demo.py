"""
Lifelong adaptation demo (v2.2.0) — measure it yourself.

Simulates a deployed model facing distribution drift over 4 "months", then
compares three strategies on the SAME drift sequence:

  frozen      -- do nothing (what most edge deployments do today)
  naive       -- fine-tune on each new batch alone (no replay, no plasticity)
  dnaty       -- result.adapt(): reservoir replay + dormant-neuron rebirth

Reports, per strategy: accuracy on the newest distribution and retention on
the first (oldest) distribution. Synthetic tabular data, CPU, ~1 minute.

Run:  python scripts/lifelong_adapt_demo.py
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from dnaty.core.arch import DynamicMLP
from dnaty.result import CompressResult

SEED = 0
N_FEATURES = 16
N_CLASSES = 3
N_PER_PHASE = 1500
PHASES = 4  # phase 0 = training distribution; 1..3 = drift steps


def make_phase(phase: int, n: int, seed: int):
    """Rotating class boundaries: each phase rotates the decision directions."""
    rng = np.random.default_rng(seed + phase)
    x = rng.normal(0, 1, size=(n, N_FEATURES)).astype(np.float32)
    angle = phase * np.pi / 6  # 30 degrees of drift per phase
    w1 = np.zeros(N_FEATURES); w1[0], w1[1] = np.cos(angle), np.sin(angle)
    w2 = np.zeros(N_FEATURES); w2[2], w2[3] = np.cos(angle), -np.sin(angle)
    logits = np.stack([x @ w1, x @ w2, -(x @ w1) - (x @ w2)], axis=1)
    y = logits.argmax(axis=1).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.inference_mode():
        return float((model(x).argmax(1) == y).float().mean())


def pretrain(model: nn.Module, x, y, epochs=30):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            crit(model(x[idx]), y[idx]).backward()
            opt.step()
    model.eval()


def naive_finetune(model: nn.Module, x, y, epochs=3, lr=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for i in range(0, len(x), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            crit(model(x[idx]), y[idx]).backward()
            opt.step()
    model.eval()


def wrap(model: DynamicMLP) -> CompressResult:
    p, f = model.count_params(), model.count_flops()
    return CompressResult(model=model, original_flops=f, compressed_flops=f,
                          original_params=p, compressed_params=p, accuracy=0.0,
                          flops_reduction=0.0, generations=0,
                          arch=list(model.layer_sizes[1:]))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    phases = [make_phase(p, N_PER_PHASE, SEED) for p in range(PHASES)]
    test_sets = [make_phase(p, 600, SEED + 100) for p in range(PHASES)]

    base = DynamicMLP([N_FEATURES, 64, 32], n_classes=N_CLASSES)
    pretrain(base, *phases[0])
    print(f"pretrained on phase 0: acc={accuracy(base, *test_sets[0]):.3f}\n")

    frozen = copy.deepcopy(base)
    naive = copy.deepcopy(base)
    dnaty_result = wrap(copy.deepcopy(base))
    # Seed the replay buffer with the training distribution, as a real
    # deployment would (adapt() ingests what it sees).
    dnaty_result.adapt(*phases[0], epochs=1, lr=1e-5, plasticity=False, seed=SEED)

    for p in range(1, PHASES):
        x, y = phases[p]
        naive_finetune(naive, x, y, epochs=10, lr=1e-3)
        m = dnaty_result.adapt(x, y, epochs=10, lr=1e-3, seed=SEED)
        print(f"phase {p}: dnaty adapt acc {m['acc_before']:.3f} -> "
              f"{m['acc_after']:.3f}  (reborn={m['n_reborn']}, "
              f"replay={m['replay_size']})")

    print(f"\n{'strategy':<10} {'newest (phase 3)':>18} {'oldest (phase 0)':>18}")
    for name, model in (("frozen", frozen), ("naive", naive),
                        ("dnaty", dnaty_result.model)):
        new_acc = accuracy(model, *test_sets[-1])
        old_acc = accuracy(model, *test_sets[0])
        print(f"{name:<10} {new_acc:>18.3f} {old_acc:>18.3f}")

    print("\nfrozen keeps the old task but never learns the new one; naive "
          "learns the new one\nand forgets the old; adapt() aims at both. "
          "Numbers vary by seed - edit SEED and rerun.")


if __name__ == "__main__":
    main()
