#!/usr/bin/env python3
"""
dNATY compress() — demo rapido.

Mostra como usar a API publica para comprimir qualquer modelo PyTorch.

Uso:
    python scripts/demo_compress.py           # MNIST, 20 gens (rapido)
    python scripts/demo_compress.py --full    # 30 gens, mais preciso
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from dnaty import compress
from dnaty.core.arch import DynamicMLP
from dnaty.experiments.fast_dataset import FastDataset

parser = argparse.ArgumentParser()
parser.add_argument("--full",    action="store_true", help="30 gens ao inves de 20")
parser.add_argument("--dataset", default="MNIST", choices=["MNIST", "FashionMNIST"])
args = parser.parse_args()

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
N_GEN    = 30 if args.full else 20
SUBSET   = 15_000

print(f"dNATY compress() demo | {args.dataset} | {N_GEN} gens | {DEVICE.upper()}")
print("-" * 60)

# Modelo de partida: MLP grande
model = DynamicMLP(
    layer_sizes=[784, 512, 256, 128],
    activations=["relu", "relu", "relu"],
    n_classes=10,
).to(DEVICE)

print(f"Modelo original : {sum(p.numel() for p in model.parameters()):,} params")

ds = FastDataset(args.dataset, device=DEVICE, train_subset=SUBSET)

t0 = time.perf_counter()
result = compress(
    model,
    ds,
    target_flops=0.5,
    n_generations=N_GEN,
    n_pop=12,
    device=DEVICE,
    verbose=True,
    seed=42,
)
elapsed = time.perf_counter() - t0

print()
print("=" * 60)
print(result.summary())
print(f"Tempo total: {elapsed:.0f}s")
print("=" * 60)

# Salva o modelo comprimido
out = Path("model_compressed.pt")
torch.save(result.model.state_dict(), out)
print(f"\nModelo salvo em: {out}")
print("Para usar: from dnaty.core.arch import DynamicMLP + torch.load(...)")
