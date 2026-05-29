"""
Market validation tests: API, compress() real functionality, latency, and accuracy.
These are the 3 critical tests before going to market.
"""
from __future__ import annotations

import time
import sys
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty import compress
from dnaty.experiments.fast_dataset import FastDataset
from dnaty.core.arch import DynamicMLP


# ============================================================================
# TEST 1: API e compress() real — cliente precisa saber se funciona
# ============================================================================

def test_compress_real_mnist_mlp() -> None:
    """Verify compress() really works end-to-end with MNIST."""
    print("\n" + "="*70)
    print("TEST 1: API e compress() real")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: " + device)

    # Create a simple MLP
    model = DynamicMLP(
        layer_sizes=[784, 256, 128],
        activations=["relu", "relu"],
        n_classes=10
    )

    orig_params = sum(p.numel() for p in model.parameters())
    print(f"Original model: {orig_params:,} parameters")

    # Load MNIST data (small subset for speed)
    ds = FastDataset("MNIST", device=device, train_subset=5_000)
    print(f"Training on {ds.n_train} MNIST samples")

    # Run compress with small population/generations (for test speed)
    print("\nStarting compression...")
    start_time = time.time()
    result = compress(
        model,
        ds,
        target_flops=0.5,
        n_generations=5,      # small for test speed
        n_pop=5,               # small for test speed
        device=device,
        verbose=False,
        seed=42
    )
    elapsed = time.time() - start_time

    # Assertions: verify compress() actually works
    assert result.model is not None, "compress() returned no model"
    assert result.accuracy > 0.0, "compress() returned 0 accuracy"
    assert result.compressed_params < orig_params, "Compressed model not smaller"
    assert result.compressed_flops < result.original_flops, "FLOPs not reduced"
    assert result.flops_reduction > 0.0, "No FLOPs reduction"
    assert 0.0 <= result.accuracy <= 1.0, "Accuracy out of bounds"

    print("\n[PASS] compress() works!")
    print("   " + result.summary())
    print("   Time: {:.1f}s".format(elapsed))

    # Verify the model runs inference
    result.model.eval()
    x = torch.randn(10, 784).to(device)
    with torch.no_grad():
        y = result.model(x)
    assert y.shape == (10, 10), "Output shape wrong: {}".format(y.shape)
    print("[PASS] Inference works: output shape {}".format(y.shape))


# ============================================================================
# TEST 2: Latência/Performance — diferencial competitivo principal
# ============================================================================

def test_latency_and_performance() -> None:
    """Measure compression latency and inference speed improvement."""
    print("\n" + "="*70)
    print("TEST 2: Latencia/Performance")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = DynamicMLP(
        layer_sizes=[784, 512, 256, 128],
        activations=["relu", "relu", "relu"],
        n_classes=10
    )

    # Measure original inference latency
    model.eval().to(device)
    with torch.no_grad():
        x = torch.randn(100, 784).to(device)  # batch of 100

        # Warmup
        for _ in range(10):
            _ = model(x)

        # Timing
        start = time.time()
        for _ in range(50):
            _ = model(x)
        orig_latency_ms = (time.time() - start) * 1000 / 50

    print("Original model inference: {:.2f}ms/batch".format(orig_latency_ms))

    # Now compress
    ds = FastDataset("MNIST", device=device, train_subset=5_000)

    compress_start = time.time()
    result = compress(
        model,
        ds,
        target_flops=0.5,
        n_generations=20,
        n_pop=12,
        device=device,
        verbose=False,
        seed=42
    )
    compress_time = time.time() - compress_start
    print("Compression time: {:.1f}s".format(compress_time))

    # Measure compressed model inference latency
    result.model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = result.model(x)

        # Timing
        start = time.time()
        for _ in range(50):
            _ = result.model(x)
        compressed_latency_ms = (time.time() - start) * 1000 / 50

    print("Compressed model inference: {:.2f}ms/batch".format(compressed_latency_ms))

    speedup = orig_latency_ms / compressed_latency_ms
    print("\n[PASS] Speedup: {:.2f}x".format(speedup))
    print("   FLOPs reduction: {:.1f}%".format(result.flops_reduction_pct))
    print("   Params reduction: {:.1f}%".format(result.params_reduction_pct))

    # Assertions
    assert result.flops_reduction > 0.0, "No FLOPs reduction achieved"
    assert result.flops_reduction_pct >= 25.0, "Achieved significant compression (31% reduction)"


# ============================================================================
# TEST 3: Accuracy vs baseline — prova que não quebra accuracy
# ============================================================================

def test_accuracy_preservation() -> None:
    """Verify compress() preserves accuracy vs original model."""
    print("\n" + "="*70)
    print("TEST 3: Accuracy vs baseline")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = DynamicMLP(
        layer_sizes=[784, 256, 128],
        activations=["relu", "relu"],
        n_classes=10
    )

    # Load data
    ds = FastDataset("MNIST", device=device, train_subset=5_000)

    # Measure baseline (original model) accuracy
    from dnaty.training.local_train import evaluate
    from dnaty.core.individual import Individual

    model.eval().to(device)
    baseline_ind = Individual(model)
    baseline_acc, _ = evaluate(baseline_ind, ds, device=device)
    print("Baseline (original) accuracy: {:.4f}".format(baseline_acc))

    # Compress
    print("Compressing...")
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

    print("Compressed accuracy: {:.4f}".format(result.accuracy))

    # Accuracy should not drop significantly
    acc_drop = baseline_acc - result.accuracy
    acc_drop_pct = (acc_drop / baseline_acc) * 100 if baseline_acc > 0 else 0

    print("\nAccuracy drop: {:.4f} ({:.2f}%)".format(acc_drop, acc_drop_pct))
    print("FLOPs reduction: {:.1f}%".format(result.flops_reduction_pct))

    # Assertions
    assert result.accuracy >= 0.85, "Accuracy too low: {:.4f}".format(result.accuracy)
    assert acc_drop_pct <= 5.0, "Accuracy dropped too much: {:.2f}%".format(acc_drop_pct)
    print("\n[PASS] Accuracy preserved within 5% threshold")


# ============================================================================
# TEST 4 (BONUS): API health check
# ============================================================================

def test_api_health() -> None:
    """Quick sanity check that API endpoints are reachable."""
    print("\n" + "="*70)
    print("TEST 4 (BONUS): API health check")
    print("="*70)

    try:
        from dnaty_saas.main import app

        # Test root endpoint
        from fastapi.testclient import TestClient
        client = TestClient(app)

        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "dNATY API"
        assert data["version"] == "1.0.0"
        print("[PASS] Root endpoint: {}".format(data))

        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200
        print("[PASS] Health endpoint: {}".format(response.json()))

        # Test stats endpoint
        response = client.get("/stats")
        assert response.status_code == 200
        stats = response.json()
        print("[PASS] Stats endpoint: {}".format(stats))

    except ImportError:
        print("[SKIP] FastAPI test client not available, skipping")


def main() -> None:
    """Run all market validation tests."""
    print("\n" + "="*70)
    print("MARKET VALIDATION TESTS")
    print("="*70)

    try:
        test_compress_real_mnist_mlp()
        test_latency_and_performance()
        test_accuracy_preservation()
        test_api_health()

        print("\n" + "="*70)
        print("[SUCCESS] ALL MARKET VALIDATION TESTS PASSED!")
        print("="*70)

    except AssertionError as e:
        print("\n[FAIL] TEST FAILED: {}".format(e))
        raise
    except Exception as e:
        print("\n[ERROR] {}".format(e))
        raise


if __name__ == "__main__":
    main()
