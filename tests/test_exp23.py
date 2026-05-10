"""Smoke tests for CIFAR and continual-learning experiment plumbing."""
from __future__ import annotations

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty.experiments.exp3_cl import FastTaskDataset
from dnaty.experiments.fast_dataset import FastDataset


def test_cifar_fast_dataset_batch_shape() -> None:
    ds = FastDataset("CIFAR10", device="cpu", train_subset=1000)
    xb, yb = ds.get_train_batch(32)
    assert tuple(xb.shape) == (32, 3, 32, 32)
    assert tuple(yb.shape) == (32,)


def test_split_mnist_task_dataset_binary_labels() -> None:
    task0 = FastTaskDataset(0, device="cpu", train_subset=200)
    xb, yb = task0.get_train_batch(32)
    assert tuple(xb.shape) == (32, 784)
    assert set(yb.unique().tolist()).issubset({0, 1})


def test_cifar_tiny_evolution_smoke() -> None:
    import dnaty.experiments.exp2_cifar as e2

    old = {
        "N_GENERATIONS": e2.N_GENERATIONS,
        "N_POP": e2.N_POP,
        "T_LOCAL": e2.T_LOCAL,
        "CIFAR_TRAIN_SUBSET": e2.CIFAR_TRAIN_SUBSET,
        "BATCH_SIZE": e2.BATCH_SIZE,
    }
    try:
        e2.N_GENERATIONS = 2
        e2.N_POP = 2
        e2.T_LOCAL = 1
        e2.CIFAR_TRAIN_SUBSET = 512
        e2.BATCH_SIZE = 128
        result = e2.run_dnaty_cnn_seed(0, "cpu")
        assert 0.0 <= result["acc"] <= 1.0
        assert result["history"]
    finally:
        for key, value in old.items():
            setattr(e2, key, value)


def test_continual_learning_tiny_smoke() -> None:
    import dnaty.experiments.exp3_cl as e3

    old_epochs = e3.N_EPOCHS_CL
    old_subset = e3.TRAIN_SUBSET_CL
    try:
        e3.N_EPOCHS_CL = 2
        e3.TRAIN_SUBSET_CL = 100
        result = e3.run_dnaty_cl_seed(0, "cpu")
        assert "metrics" in result
        assert "BWT" in result["metrics"]
    finally:
        e3.N_EPOCHS_CL = old_epochs
        e3.TRAIN_SUBSET_CL = old_subset


def main() -> None:
    print("Running dNaty exp2/exp3 smoke tests...")
    start = time.time()
    test_cifar_fast_dataset_batch_shape()
    test_split_mnist_task_dataset_binary_labels()
    test_cifar_tiny_evolution_smoke()
    test_continual_learning_tiny_smoke()
    print(f"TODOS OS TESTES OK ({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
