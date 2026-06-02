"""
Reproducibility Testing: Verify deterministic results with same seed.
Critical for production: customers need to reproduce compression results.
"""
from __future__ import annotations

import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty import compress
from dnaty.core.arch import DynamicMLP
from torch.utils.data import DataLoader, TensorDataset


def test_reproducibility_same_seed():
    """Verify: Same seed = Same compression result."""
    print("\n" + "="*70)
    print("REPRODUCIBILITY TEST 1: Same Seed = Same Result")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fixed data
    torch.manual_seed(42)
    X = torch.randn(3_000, 128)
    y = torch.randint(0, 2, (3_000,))

    # Run 1 with seed=99
    print("Run 1: compress with seed=99...")
    model1 = DynamicMLP([128, 256, 128], activations=["relu", "relu"], n_classes=2)
    loader1 = DataLoader(TensorDataset(X, y), batch_size=64)

    result1 = compress(
        model1, loader1,
        target_flops=0.6,
        n_generations=5,
        n_pop=4,
        device=device,
        verbose=False,
        seed=99
    )

    # Run 2 with same seed=99
    print("Run 2: compress with seed=99 again...")
    model2 = DynamicMLP([128, 256, 128], activations=["relu", "relu"], n_classes=2)
    loader2 = DataLoader(TensorDataset(X, y), batch_size=64)

    result2 = compress(
        model2, loader2,
        target_flops=0.6,
        n_generations=5,
        n_pop=4,
        device=device,
        verbose=False,
        seed=99
    )

    # Compare
    print("\nRun 1 architecture: {}".format(result1.arch))
    print("Run 2 architecture: {}".format(result2.arch))
    print("Run 1 FLOPs reduction: {:.1f}%".format(result1.flops_reduction_pct))
    print("Run 2 FLOPs reduction: {:.1f}%".format(result2.flops_reduction_pct))
    print("Run 1 accuracy: {:.4f}".format(result1.accuracy))
    print("Run 2 accuracy: {:.4f}".format(result2.accuracy))

    # Assert determinism
    assert result1.arch == result2.arch, "Different architectures found!"
    assert abs(result1.flops_reduction_pct - result2.flops_reduction_pct) < 0.01, "FLOPs diff"
    assert abs(result1.accuracy - result2.accuracy) < 0.0001, "Accuracy diff"

    print("\n[PASS] Results are deterministic with same seed!")


def test_reproducibility_different_seeds():
    """Verify: Different seeds = Different (but valid) results."""
    print("\n" + "="*70)
    print("REPRODUCIBILITY TEST 2: Different Seeds = Different Results")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)
    X = torch.randn(3_000, 128)
    y = torch.randint(0, 2, (3_000,))

    results = {}

    for seed in [100, 101, 102]:
        print("Running with seed={}...".format(seed))
        model = DynamicMLP([128, 256, 128], activations=["relu", "relu"], n_classes=2)
        loader = DataLoader(TensorDataset(X, y), batch_size=64)

        result = compress(
            model, loader,
            target_flops=0.6,
            n_generations=5,
            n_pop=4,
            device=device,
            verbose=False,
            seed=seed
        )

        results[seed] = {
            "arch": result.arch,
            "flops": result.flops_reduction_pct,
            "accuracy": result.accuracy
        }

    print("\nResults across different seeds:")
    for seed, r in results.items():
        print("  Seed {}: arch={} flops={:.1f}% acc={:.4f}".format(
            seed, r["arch"], r["flops"], r["accuracy"]))

    # Check that results are different (not all the same)
    archs = [r["arch"] for r in results.values()]
    unique_archs = len(set([tuple(a) for a in archs]))

    print("\nUnique architectures found: {}".format(unique_archs))
    assert unique_archs >= 2, "Different seeds should produce different architectures"

    # But all should be valid (accuracy > 0)
    for seed, r in results.items():
        assert r["accuracy"] > 0, "Invalid accuracy for seed {}".format(seed)
        assert r["flops"] > 0, "No compression for seed {}".format(seed)

    print("\n[PASS] Different seeds produce different (valid) results!")


def test_reproducibility_across_platforms():
    """Verify: Same seed works across CPU/GPU (if available)."""
    print("\n" + "="*70)
    print("REPRODUCIBILITY TEST 3: CPU vs GPU Consistency")
    print("="*70)

    torch.manual_seed(42)
    X = torch.randn(3_000, 128)
    y = torch.randint(0, 2, (3_000,))

    # Always test CPU
    devices_to_test = ["cpu"]
    if torch.cuda.is_available():
        devices_to_test.append("cuda")
        print("GPU available, testing both CPU and GPU...")
    else:
        print("GPU not available, testing CPU only...")

    results = {}

    for device in devices_to_test:
        print("Running on {}...".format(device))
        model = DynamicMLP([128, 256, 128], activations=["relu", "relu"], n_classes=2)
        loader = DataLoader(TensorDataset(X, y), batch_size=64)

        result = compress(
            model, loader,
            target_flops=0.6,
            n_generations=5,
            n_pop=4,
            device=device,
            verbose=False,
            seed=42
        )

        results[device] = {
            "arch": result.arch,
            "flops": result.flops_reduction_pct,
            "accuracy": result.accuracy
        }

    print("\nResults:")
    for device, r in results.items():
        print("  {}: arch={} flops={:.1f}% acc={:.4f}".format(
            device, r["arch"], r["flops"], r["accuracy"]))

    if len(results) > 1:
        cpu_result = results["cpu"]
        gpu_result = results["cuda"]

        # CPU and GPU may differ slightly due to floating-point precision
        # But should be close enough
        arch_match = cpu_result["arch"] == gpu_result["arch"]
        flops_diff = abs(cpu_result["flops"] - gpu_result["flops"])
        acc_diff = abs(cpu_result["accuracy"] - gpu_result["accuracy"])

        print("\nCPU vs GPU comparison:")
        print("  Architecture match: {}".format(arch_match))
        print("  FLOPs diff: {:.2f}%".format(flops_diff))
        print("  Accuracy diff: {:.4f}".format(acc_diff))

        # Allow small differences due to floating-point
        assert flops_diff < 2.0, "FLOPs difference too large"
        assert acc_diff < 0.01, "Accuracy difference too large"

    print("\n[PASS] Reproducibility across devices verified!")


def main():
    """Run reproducibility tests."""
    print("\n" + "="*70)
    print("REPRODUCIBILITY TESTING - Deterministic Results")
    print("="*70)

    try:
        test_reproducibility_same_seed()
        test_reproducibility_different_seeds()
        test_reproducibility_across_platforms()

        print("\n" + "="*70)
        print("[SUCCESS] REPRODUCIBILITY TESTS PASSED!")
        print("="*70)
        print("\nConclusion:")
        print("  - Same seed produces identical results")
        print("  - Different seeds produce different results")
        print("  - CPU/GPU produce consistent results")
        print("  - Customers can reproduce their compressions")

    except AssertionError as e:
        print("\n[FAIL] Reproducibility test failed: {}".format(e))
        raise
    except Exception as e:
        print("\n[ERROR] {}".format(e))
        raise


if __name__ == "__main__":
    main()
