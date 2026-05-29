#!/usr/bin/env python3
"""
Simple example: Compress a model with dNATY

Usage:
    python example_compress.py
"""

import torch
import torch.nn as nn
from dnaty import compress
from dnaty.experiments.fast_dataset import FastDataset

print("="*70)
print("dNATY — Compress Your Model in 3 Lines")
print("="*70)

# Step 1: Create or load your model
print("\n[1] Loading model...")
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print(f"    Model: {sum(p.numel() for p in model.parameters()):,} parameters")

# Step 2: Load training data
print("\n[2] Loading training data...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"    Device: {device}")
ds = FastDataset("MNIST", device=device, train_subset=10_000)
print(f"    Dataset: MNIST subset ({ds.n_train} training samples)")

# Step 3: Compress!
print("\n[3] Starting compression (this takes ~2-5 min)...")
result = compress(
    model,
    ds,
    target_flops=0.5,       # Aim for 50% FLOPs reduction
    n_generations=30,       # 30 evolutionary generations
    n_pop=15,               # Population size
    device=device,
    verbose=True,
    seed=42
)

# Step 4: Results
print("\n" + "="*70)
print("COMPRESSION RESULTS")
print("="*70)
print(result.summary())

print(f"\nOriginal:   {result.original_params:>10,} params | {result.original_flops:>12,} FLOPs")
print(f"Compressed: {result.compressed_params:>10,} params | {result.compressed_flops:>12,} FLOPs")
print(f"Reduction:  {result.params_reduction_pct:>9.1f}% params | {result.flops_reduction_pct:>11.1f}% FLOPs")
print(f"Accuracy:   {result.accuracy:>10.4f}")

# Step 5: Save the model
print("\n[4] Saving compressed model...")
torch.save(result.model.state_dict(), "model_compressed.pt")
print("    Saved to: model_compressed.pt")

# Step 6: Use the compressed model
print("\n[5] Testing compressed model...")
result.model.eval()
x = torch.randn(10, 784).to(device)
with torch.no_grad():
    y = result.model(x)
print(f"    Inference shape: {y.shape}")
print(f"    Output sample: {y[0]}")

print("\n" + "="*70)
print("SUCCESS! Your model is now compressed and ready to deploy.")
print("="*70)
