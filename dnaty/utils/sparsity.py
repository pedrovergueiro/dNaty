"""
Structural N:M sparsity for Linear layers.

Applies N:M weight masking (default 2:4) to nn.Linear layers:
  - For every group of M consecutive weights in a row, zero out the N smallest.
  - 2:4 sparsity means 50% of weights become zero in a structured pattern
    compatible with NVIDIA sparse tensor cores (A100+) and INT8 inference.

On CPU (no sparse tensor cores), this reduces model size and memory bandwidth
but does not directly improve FLOP count. On edge devices with INT8 sparse
support (e.g., Jetson Orin), actual speedup is 1.5–2×.

Usage:
    from dnaty.utils.sparsity import apply_nm_sparsity, sparsity_stats

    model = apply_nm_sparsity(model, n=2, m=4)
    stats = sparsity_stats(model)
    print(f"Sparsity: {stats['global_sparsity_pct']:.1f}%")
    result.export_onnx("model.onnx")  # sparse weights baked in
"""
from __future__ import annotations

import torch
import torch.nn as nn


def apply_nm_sparsity(
    model: nn.Module,
    n: int = 2,
    m: int = 4,
    inplace: bool = True,
) -> nn.Module:
    """
    Apply N:M structured sparsity to all nn.Linear layers.

    For each row of a weight matrix, groups consecutive M weights and zeros
    out the N with smallest absolute value. Result: exactly N/M weight
    fraction zeroed (50% for 2:4).

    The mask is applied permanently (baked into .data) — the model is ready
    for ONNX export with sparse weights. No backward-pass masking hooks are
    added; this is post-training sparsification, not sparse training.

    Args:
        model:   nn.Module with nn.Linear layers.
        n:       Number of zeros per group of m (default 2).
        m:       Group size (default 4). Must divide each row's width evenly;
                 rows that cannot be grouped are skipped with a warning.
        inplace: Modify model in-place (default True). Set False to get a copy.

    Returns:
        The sparsified model (same object if inplace=True).
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        W = module.weight.data  # (out_features, in_features)
        out_f, in_f = W.shape

        if in_f % m != 0:
            # Skip layers whose width isn't divisible by m
            continue

        # Reshape to (out_features, in_features // m, m)
        W_grouped = W.view(out_f, in_f // m, m)

        # Find indices of the N smallest absolute values in each group
        _, idx = torch.topk(W_grouped.abs(), k=n, dim=-1, largest=False)

        # Build mask: 1 where we keep, 0 where we zero
        mask = torch.ones_like(W_grouped)
        mask.scatter_(-1, idx, 0.0)

        # Apply mask and write back
        W_sparse = (W_grouped * mask).view(out_f, in_f)
        module.weight.data.copy_(W_sparse)

    return model


def sparsity_stats(model: nn.Module) -> dict:
    """
    Compute sparsity statistics for all nn.Linear layers.

    Returns:
        {
          "global_sparsity_pct": float,   # % of zero weights across all Linear
          "layer_stats": [                 # per-layer breakdown
            {"name": str, "shape": tuple, "sparsity_pct": float},
            ...
          ]
        }
    """
    total_weights = 0
    total_zeros = 0
    layer_stats = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        W = module.weight.data
        n_weights = W.numel()
        n_zeros = int((W == 0).sum().item())
        pct = 100.0 * n_zeros / max(n_weights, 1)
        total_weights += n_weights
        total_zeros += n_zeros
        layer_stats.append({
            "name": name,
            "shape": tuple(W.shape),
            "sparsity_pct": round(pct, 2),
        })

    global_pct = 100.0 * total_zeros / max(total_weights, 1)
    return {
        "global_sparsity_pct": round(global_pct, 2),
        "total_weights": total_weights,
        "zero_weights": total_zeros,
        "layer_stats": layer_stats,
    }
