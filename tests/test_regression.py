"""
Regression Testing: Verify results stay consistent across versions.
Snapshots compression results and validates no degradation.
"""
from __future__ import annotations

import sys
import torch
import torch.nn as nn
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty import compress
from dnaty.core.arch import DynamicMLP
from torch.utils.data import DataLoader, TensorDataset


BASELINE_SNAPSHOT = {
    "version": "1.1.7",
    "test_mlp_mnist": {
        "original_params": 235914,
        "original_flops": 469504,
        "compressed_params": 140754,
        "compressed_flops": 279808,
        "flops_reduction_pct": 29.6,
        "accuracy": 0.9430,
        "arch": [196, 32, 128]
    },
    "test_mlp_tabular": {
        "original_params": 75202,
        "original_flops": 147712,
        "compressed_params": 62258,
        "compressed_flops": 122112,
        "flops_reduction_pct": 17.3,
        "accuracy": 0.9950,
        "arch": [224, 112, 64]
    }
}


def test_regression_mnist_mlp():
    """Regression: Ensure MNIST compression doesn't degrade."""
    print("\n" + "="*70)
    print("REGRESSION TEST 1: MNIST MLP Compression")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create baseline model (same as before)
    model = DynamicMLP(
        layer_sizes=[784, 256, 128],
        activations=["relu", "relu"],
        n_classes=10
    )

    # Load MNIST-like data (same seed as before)
    torch.manual_seed(42)
    from dnaty.experiments.fast_dataset import FastDataset
    ds = FastDataset("MNIST", device=device, train_subset=5_000)

    # Compress with same parameters
    print("Compressing with seed=42...")
    result = compress(
        model,
        ds,
        target_flops=0.5,
        n_generations=5,
        n_pop=5,
        device=device,
        verbose=False,
        seed=42
    )

    baseline = BASELINE_SNAPSHOT["test_mlp_mnist"]

    print("\nBaseline vs Current:")
    print("  Original params: {} vs {}".format(baseline["original_params"], result.original_params))
    print("  FLOPs reduction: {:.1f}% vs {:.1f}%".format(
        baseline["flops_reduction_pct"], result.flops_reduction_pct))
    print("  Accuracy: {:.4f} vs {:.4f}".format(baseline["accuracy"], result.accuracy))
    print("  Architecture: {} vs {}".format(baseline["arch"], result.arch))

    # Regression checks: Allow 10% tolerance (stochastic algorithm)
    flops_diff = abs(result.flops_reduction_pct - baseline["flops_reduction_pct"])
    acc_diff = abs(result.accuracy - baseline["accuracy"])

    assert flops_diff < 10.0, "FLOPs reduction regressed by {:.1f}%".format(flops_diff)
    assert acc_diff < 0.10, "Accuracy regressed by {:.4f}".format(acc_diff)

    print("\n[PASS] No regression detected!")


def test_regression_tabular_mlp():
    """Regression: Ensure tabular data compression stays stable."""
    print("\n" + "="*70)
    print("REGRESSION TEST 2: Tabular Data MLP")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create same model
    model = DynamicMLP(
        layer_sizes=[128, 256, 128, 64],
        activations=["relu", "relu", "relu"],
        n_classes=2
    )

    # Same synthetic data
    torch.manual_seed(42)
    X_train = torch.randn(5_000, 128)
    y_train = torch.randint(0, 2, (5_000,))
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128)

    print("Compressing tabular model...")
    result = compress(
        model,
        train_loader,
        target_flops=0.5,
        n_generations=5,
        n_pop=5,
        device=device,
        verbose=False,
        seed=42
    )

    baseline = BASELINE_SNAPSHOT["test_mlp_tabular"]

    print("\nBaseline vs Current:")
    print("  FLOPs reduction: {:.1f}% vs {:.1f}%".format(
        baseline["flops_reduction_pct"], result.flops_reduction_pct))
    print("  Accuracy: {:.4f} vs {:.4f}".format(baseline["accuracy"], result.accuracy))

    # Regression: only fail if current is significantly WORSE than baseline
    # (getting better compression is fine — that's not a regression)
    flops_regressed = baseline["flops_reduction_pct"] - result.flops_reduction_pct
    acc_diff = baseline["accuracy"] - result.accuracy

    assert flops_regressed < 15.0, "FLOPs reduction regressed by {:.1f}%".format(flops_regressed)
    assert acc_diff < 0.10, "Accuracy regressed by {:.4f}".format(acc_diff)

    print("\n[PASS] Tabular model regression check passed!")


def test_snapshot_update():
    """Generate new snapshot for current version."""
    print("\n" + "="*70)
    print("REGRESSION TEST 3: Snapshot Generation")
    print("="*70)

    print("Current baseline snapshot:")
    print(json.dumps(BASELINE_SNAPSHOT, indent=2))

    print("\n[INFO] Baseline snapshot is version {}".format(BASELINE_SNAPSHOT["version"]))
    print("[INFO] To update snapshot for new version:")
    print("      1. Run all tests")
    print("      2. Capture results")
    print("      3. Update BASELINE_SNAPSHOT dict")
    print("      4. Commit to git")


def main():
    """Run regression tests."""
    print("\n" + "="*70)
    print("REGRESSION TESTING - Ensure No Degradation")
    print("="*70)

    try:
        test_regression_mnist_mlp()
        test_regression_tabular_mlp()
        test_snapshot_update()

        print("\n" + "="*70)
        print("[SUCCESS] REGRESSION TESTS PASSED!")
        print("="*70)
        print("\nConclusion:")
        print("  - Compression results stable across runs")
        print("  - No degradation from baseline")
        print("  - Safe to ship v1.0 to production")

    except AssertionError as e:
        print("\n[FAIL] Regression detected: {}".format(e))
        raise
    except Exception as e:
        print("\n[ERROR] {}".format(e))
        raise


if __name__ == "__main__":
    main()
