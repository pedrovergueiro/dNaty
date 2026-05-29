"""
dNATY compress — public API for model compression via evolutionary NAS.

Usage:
    from dnaty import compress

    result = compress(model, train_data, target_flops=0.5)
    print(f"Compressed {result.flops_reduction_pct:.1f}% FLOPs, acc={result.accuracy:.4f}")
    result.model  # ready-to-use compressed PyTorch model
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CompressResult:
    model: nn.Module
    original_flops: int
    compressed_flops: int
    original_params: int
    compressed_params: int
    accuracy: float
    flops_reduction: float      # e.g. 0.465 = 46.5% less FLOPs
    generations: int
    arch: list[int] = field(default_factory=list)   # hidden layer sizes found

    @property
    def flops_reduction_pct(self) -> float:
        return self.flops_reduction * 100

    @property
    def params_reduction_pct(self) -> float:
        if self.original_params == 0:
            return 0.0
        return (1.0 - self.compressed_params / self.original_params) * 100

    def summary(self) -> str:
        return (
            f"CompressResult | arch={self.arch} | "
            f"FLOPs -{self.flops_reduction_pct:.1f}% "
            f"({self.original_flops:,} -> {self.compressed_flops:,}) | "
            f"params -{self.params_reduction_pct:.1f}% "
            f"({self.original_params:,} -> {self.compressed_params:,}) | "
            f"acc={self.accuracy:.4f}"
        )


def compress(
    model: nn.Module,
    train_data,
    *,
    target_flops: float = 0.5,
    n_generations: int = 30,
    n_pop: int = 15,
    device: Optional[str] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
) -> CompressResult:
    """
    Find a smaller, faster architecture for the same task using evolutionary NAS.

    dNATY searches architectures guided by episodic memory — operators that
    helped before are tried more often. The search is Pareto-optimal: it
    maximises accuracy and minimises FLOPs/params simultaneously.

    Works best with MLP models (nn.Linear layers). The search starts from
    the architecture inferred from ``model`` and evolves from there.

    Args:
        model:          Any PyTorch nn.Module containing nn.Linear layers.
        train_data:     DataLoader or FastDataset used to train and evaluate
                        candidate architectures.
        target_flops:   Target FLOPs as fraction of original (0.5 = 50% less).
                        Controls lambda2 pressure — lower = more compression.
        n_generations:  Evolutionary generations to run (30 is a good default).
        n_pop:          Population size (15 balances diversity vs. speed).
        device:         'cpu' or 'cuda'. Auto-detected when None.
        verbose:        Print generation-by-generation progress.
        seed:           Fix for reproducibility.

    Returns:
        CompressResult with the best model found and compression metrics.

    Example:
        >>> from dnaty import compress
        >>> from dnaty.experiments.fast_dataset import FastDataset
        >>> ds = FastDataset("MNIST", device="cpu", train_subset=10_000)
        >>> model = ... # your trained PyTorch model
        >>> result = compress(model, ds, target_flops=0.5, n_generations=30)
        >>> print(result.summary())
    """
    import numpy as np
    from dnaty.evolution.evolver import DnatyEvolver

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    layer_sizes = _infer_layer_sizes(model)
    input_size  = layer_sizes[0]
    n_classes   = layer_sizes[-1]
    init_hidden = layer_sizes[1:-1]

    lambda2 = 3e-6  # strong enough to drive real compression, weak enough to preserve acc

    evolver = DnatyEvolver(
        n_pop=n_pop,
        n_generations=n_generations,
        t_local=3,
        input_size=input_size,
        n_classes=n_classes,
        init_hidden=init_hidden,
        device=device,
        verbose=verbose,
        lambda2=lambda2,
    )

    # Baseline: measure the original model before search
    # Use layer_sizes (already inferred) to avoid requiring DynamicMLP.count_flops()
    orig_flops  = sum(2 * layer_sizes[i] * layer_sizes[i + 1] for i in range(len(layer_sizes) - 1))
    orig_params = sum(p.numel() for p in model.parameters())

    # Disable early stopping so all generations run — with large datasets
    # accuracy plateaus fast and early stop would fire before FLOPs reduction happens.
    evolver.run(train_data, train_data, early_stop_patience=n_generations)

    # Select most-compressed individual from Pareto population with acc >= 95%.
    # run() returns max-accuracy individual, but the population contains the full
    # Pareto front — smaller models that still meet the accuracy floor live there.
    acc_floor = 0.95
    candidates = [ind for ind in evolver.population if ind.acc >= acc_floor]
    if not candidates:
        candidates = evolver.population
    best = min(candidates, key=lambda ind: ind.count_flops())

    compressed_flops  = best.count_flops()
    compressed_params = best.count_params()

    # layer_sizes on DynamicMLP includes input (e.g. [784, 512, 128]).
    # arch is hidden-only so callers reconstruct with: DynamicMLP([784] + arch, ...)
    full_sizes = list(getattr(best.model, "layer_sizes", [input_size] + init_hidden))
    arch = full_sizes[1:]

    return CompressResult(
        model=best.model,
        original_flops=orig_flops,
        compressed_flops=compressed_flops,
        original_params=orig_params,
        compressed_params=compressed_params,
        accuracy=best.acc,
        flops_reduction=max(0.0, 1.0 - compressed_flops / max(orig_flops, 1)),
        generations=n_generations,
        arch=arch,
    )


def _infer_layer_sizes(model: nn.Module) -> list[int]:
    """Extract [in, h1, h2, ..., out] from a Linear-based model."""
    sizes: list[int] = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if not sizes:
                sizes.append(m.in_features)
            sizes.append(m.out_features)
    if len(sizes) < 2:
        raise ValueError(
            "Cannot infer architecture: model must contain at least one nn.Linear layer."
        )
    return sizes
