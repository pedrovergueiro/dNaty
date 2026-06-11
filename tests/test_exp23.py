"""Smoke tests for datasets, monitoring, and FLOPs counter."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty.experiments.fast_dataset import FastDataset
from dnaty.monitoring.drift import DriftDetector
from dnaty.utils.flops_counter import count_flops, flops_by_layer
from dnaty.core.arch import DynamicMLP


def test_cifar_fast_dataset_batch_shape() -> None:
    ds = FastDataset("CIFAR10", device="cpu", train_subset=1000)
    xb, yb = ds.get_train_batch(32)
    assert tuple(xb.shape) == (32, 3, 32, 32)
    assert tuple(yb.shape) == (32,)


def test_mnist_fast_dataset_batch_shape() -> None:
    ds = FastDataset("MNIST", device="cpu", train_subset=500)
    xb, yb = ds.get_train_batch(16)
    assert tuple(xb.shape) == (16, 784)  # FastDataset flattens MNIST
    assert tuple(yb.shape) == (16,)
    assert set(yb.unique().tolist()).issubset(set(range(10)))


def test_drift_detector_no_drift() -> None:
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, (500, 8)).astype(np.float32)
    test = rng.normal(0, 1, (200, 8)).astype(np.float32)

    det = DriftDetector(psi_threshold=0.2)
    det.fit(train)
    report = det.score(test)

    assert "psi_mean" in report
    assert "drifted" in report
    assert report["n_samples"] == 200
    assert not report["drifted"]


def test_drift_detector_detects_drift() -> None:
    rng = np.random.default_rng(1)
    train = rng.normal(0, 1, (500, 4)).astype(np.float32)
    shifted = rng.normal(5, 1, (200, 4)).astype(np.float32)  # large shift

    det = DriftDetector(psi_threshold=0.2)
    det.fit(train)
    assert det.is_drifted(shifted)


def test_flops_counter_mlp() -> None:
    model = DynamicMLP([784, 128, 64], ["relu", "relu"], n_classes=10)
    total = count_flops(model, input_shape=(784,))
    assert total > 0

    by_layer = flops_by_layer(model, input_shape=(784,))
    assert len(by_layer) > 0
    assert sum(by_layer.values()) == total


def test_flops_counter_respects_architecture() -> None:
    small = DynamicMLP([784, 32], ["relu"], n_classes=10)
    large = DynamicMLP([784, 512], ["relu"], n_classes=10)
    assert count_flops(small, (784,)) < count_flops(large, (784,))
