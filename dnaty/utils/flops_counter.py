"""
Hook-based FLOPs counter — cost model per operation type.

Motivation (Mechanical-Flatbed, r/computervision):
  Shape-based counters miss the input spatial resolution: a Conv2d(64,64,3)
  on a 32×32 feature map has 4× more FLOPs than on a 16×16 map, yet identical
  parameter count. This module uses forward hooks to measure actual MACs.

Supported ops:
  nn.Linear       → 2 × in_features × out_features  (per sample)
  nn.Conv2d       → 2 × k² × C_in/groups × C_out × H_out × W_out
  nn.BatchNorm1d/2d → elements (cheap normalisation pass)
  nn.ConvTranspose2d → same formula as Conv2d but H/W are upsampled

Usage:
    from dnaty.utils.flops_counter import count_flops, flops_by_layer

    total = count_flops(model, input_shape=(3, 32, 32))
    detail = flops_by_layer(model, input_shape=(3, 32, 32))
    for name, ops in detail.items():
        print(f"  {name}: {ops:,}")
"""
from __future__ import annotations
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn


def _make_hook(store: dict[str, int], name: str):
    def hook(module: nn.Module, inp: tuple, out: Any) -> None:
        x = inp[0]
        batch = x.shape[0] if x.dim() >= 2 else 1

        if isinstance(module, nn.Linear):
            # 2 MACs per weight element (multiply + accumulate)
            store[name] = 2 * module.in_features * module.out_features * batch

        elif isinstance(module, nn.Conv2d):
            hout = out.shape[-2]
            wout = out.shape[-1]
            k = module.kernel_size[0] * module.kernel_size[1]
            cin_per_group = module.in_channels // module.groups
            store[name] = 2 * k * cin_per_group * module.out_channels * hout * wout * batch

        elif isinstance(module, nn.ConvTranspose2d):
            hout = out.shape[-2]
            wout = out.shape[-1]
            k = module.kernel_size[0] * module.kernel_size[1]
            cin_per_group = module.in_channels // module.groups
            store[name] = 2 * k * cin_per_group * module.out_channels * hout * wout * batch

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            # 2 ops per element: normalise + affine
            store[name] = 2 * x.numel()

    return hook


@contextmanager
def _hooks(model: nn.Module, store: dict[str, int]):
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d,
                          nn.BatchNorm1d, nn.BatchNorm2d)):
            handles.append(m.register_forward_hook(_make_hook(store, name)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def flops_by_layer(
    model: nn.Module,
    input_shape: tuple,
    device: str = "cpu",
) -> dict[str, int]:
    """Return per-layer FLOPs dict for a single-sample forward pass.

    Args:
        model:       Any PyTorch model.
        input_shape: Shape WITHOUT batch dim, e.g. (784,) or (3, 32, 32).
        device:      Device for the dummy tensor.

    Returns:
        Dict mapping layer name → FLOPs (integer, single sample).
    """
    store: dict[str, int] = {}
    dummy = torch.zeros(1, *input_shape, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad(), _hooks(model, store):
        model(dummy)
    if was_training:
        model.train()
    return store


def count_flops(
    model: nn.Module,
    input_shape: tuple,
    device: str = "cpu",
) -> int:
    """Total FLOPs for one forward pass (single sample).

    Handles Linear, Conv2d, ConvTranspose2d, BatchNorm — correctly accounts
    for spatial resolution, groups, and depthwise separable layers.
    """
    return sum(flops_by_layer(model, input_shape, device).values())
