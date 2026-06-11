"""Tests for the compress() public API."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dnaty import compress
from dnaty.core.arch import DynamicMLP


def _tiny_loader(n: int = 200, in_features: int = 32, n_classes: int = 4) -> DataLoader:
    torch.manual_seed(0)
    x = torch.randn(n, in_features)
    y = torch.randint(0, n_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=64)


def test_compress_returns_compress_result():
    model = DynamicMLP([32, 64, 32], ["relu", "relu"], n_classes=4)
    loader = _tiny_loader()
    result = compress(model, loader, n_generations=3, n_pop=3, verbose=False, finetune_epochs=0)
    assert result is not None
    assert hasattr(result, "model")
    assert hasattr(result, "flops_reduction")
    assert hasattr(result, "accuracy")


def test_compress_output_shape_preserved():
    model = DynamicMLP([32, 64, 32], ["relu", "relu"], n_classes=4)
    loader = _tiny_loader()
    result = compress(model, loader, n_generations=2, n_pop=2, verbose=False, finetune_epochs=0)
    result.model.eval()
    with torch.no_grad():
        out = result.model(torch.randn(5, 32))
    assert out.shape == (5, 4)


def test_compress_accuracy_in_valid_range():
    model = DynamicMLP([32, 64], ["relu"], n_classes=4)
    loader = _tiny_loader()
    result = compress(model, loader, n_generations=2, n_pop=2, verbose=False, finetune_epochs=0)
    assert 0.0 <= result.accuracy <= 1.0


def test_compress_summary_contains_flops():
    model = DynamicMLP([32, 64], ["relu"], n_classes=4)
    loader = _tiny_loader()
    result = compress(model, loader, n_generations=2, n_pop=2, verbose=False, finetune_epochs=0)
    s = result.summary()
    assert "FLOPs" in s
    assert "acc=" in s


def test_compress_seed_is_accepted():
    model = DynamicMLP([32, 64], ["relu"], n_classes=4)
    loader = _tiny_loader()
    result = compress(model, loader, n_generations=2, n_pop=2, verbose=False, seed=7, finetune_epochs=0)
    assert result is not None
