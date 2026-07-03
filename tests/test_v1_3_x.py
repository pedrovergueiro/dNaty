"""
Tests for dNATY v1.3.x — hardware-aware latency NAS.

Covers:
  - measure_latency: ONNX Runtime microbenchmark
  - hw_detect: hardware detection + scaling tables
  - estimate_latency: cross-device latency estimation
  - LatencyPredictor: GBM surrogate (fallback + trained)
  - LatencyEvolver: bi-objective NAS (acc + latency)
  - compress(target="latency"): new public API
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _tiny_mlp(in_=784, hidden=64, out=10):
    return nn.Sequential(nn.Linear(in_, hidden), nn.ReLU(), nn.Linear(hidden, out))


def _loader(n=200, in_=784, classes=10):
    X = torch.randn(n, in_)
    y = torch.randint(0, classes, (n,))
    return DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)


# ── latency_bench ─────────────────────────────────────────────────────────────

def test_measure_latency_returns_expected_keys():
    from dnaty.utils.latency_bench import measure_latency
    model = _tiny_mlp()
    result = measure_latency(model, input_shape=(784,), n_warmup=2, n_runs=10)
    assert set(result.keys()) == {"p50_ms", "p95_ms", "mean_ms", "fps"}


def test_measure_latency_positive_values():
    from dnaty.utils.latency_bench import measure_latency
    model = _tiny_mlp()
    result = measure_latency(model, input_shape=(784,), n_warmup=2, n_runs=10)
    assert result["p50_ms"] > 0
    assert result["p95_ms"] >= result["p50_ms"]
    assert result["fps"] > 0


def test_smaller_model_faster_or_equal():
    from dnaty.utils.latency_bench import measure_latency
    small = _tiny_mlp(hidden=16)
    large = nn.Sequential(
        nn.Linear(784, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(),
        nn.Linear(256, 10),
    )
    lat_small = measure_latency(small, (784,), n_warmup=5, n_runs=30)
    lat_large = measure_latency(large, (784,), n_warmup=5, n_runs=30)
    # Small model should not be slower than large (allow 5x slack for CPU noise)
    assert lat_small["mean_ms"] <= lat_large["mean_ms"] * 5


# ── hw_detect ─────────────────────────────────────────────────────────────────

def test_detect_hw_returns_expected_keys():
    from dnaty.utils.hw_detect import detect_hw
    hw = detect_hw()
    assert "arch" in hw
    assert "device_class" in hw
    assert "cores" in hw
    assert "scale_vs_x86" in hw
    assert hw["cores"] >= 1
    assert hw["scale_vs_x86"] > 0


def test_latency_scale_known_devices():
    from dnaty.utils.hw_detect import latency_scale
    assert latency_scale("cpu") == 1.0
    assert latency_scale("x86_64") == 1.0
    assert latency_scale("rpi4") == 9.0
    assert latency_scale("rpi5") == 4.5
    assert latency_scale("apple_m1") == 0.6


def test_latency_scale_aliases():
    from dnaty.utils.hw_detect import latency_scale
    assert latency_scale("m1") == latency_scale("apple_m1")
    assert latency_scale("raspberry_pi_4") == latency_scale("rpi4")


def test_estimate_latency():
    from dnaty.utils.hw_detect import estimate_latency
    # 1ms on CPU → 9ms estimated on RPi4
    est = estimate_latency(1.0, from_device="cpu", to_device="rpi4")
    assert abs(est - 9.0) < 0.01

    # Same device → identity
    est2 = estimate_latency(2.5, from_device="cpu", to_device="cpu")
    assert abs(est2 - 2.5) < 0.01


# ── LatencyPredictor ──────────────────────────────────────────────────────────

def test_latency_predictor_fallback_no_model():
    from dnaty.utils.latency_predictor import LatencyPredictor
    pred = LatencyPredictor(model_path="/nonexistent/path.json")
    assert not pred.is_trained()
    # Fallback should return a non-negative number
    feats = {"n_layers": 2, "widths": [128, 64], "total_params": 50000,
              "total_flops": 100000, "input_size": 784}
    result = pred.predict_ms(feats)
    assert result >= 0


def test_latency_predictor_train_predict():
    from dnaty.utils.latency_predictor import LatencyPredictor
    import numpy as np

    # Generate synthetic training data
    rng = np.random.default_rng(42)
    records = []
    for _ in range(200):
        n_layers = int(rng.integers(1, 4))
        widths = [int(rng.integers(32, 256)) for _ in range(n_layers)]
        params = sum(widths) * 100
        flops = params * 2
        # Simple linear latency: proportional to params
        lat_ms = params * 1e-6 + rng.normal(0, 0.001)
        records.append({
            "features": {"n_layers": n_layers, "widths": widths,
                         "total_params": params, "total_flops": flops,
                         "input_size": 784},
            "latency_ms": max(0.001, float(lat_ms)),
        })

    pred = LatencyPredictor(model_path="/nonexistent/path.json")
    stats = pred.train(records)
    assert pred.is_trained()
    assert "mape_pct" in stats
    assert "r2" in stats
    assert stats["n_train"] + stats["n_val"] == 200

    # Predict on a new architecture
    feats = {"n_layers": 2, "widths": [128, 64], "total_params": 20000,
              "total_flops": 40000, "input_size": 784}
    p = pred.predict_ms(feats)
    assert p > 0


# ── LatencyEvolver ────────────────────────────────────────────────────────────

def test_latency_evolver_runs():
    from dnaty.evolution.evolver import LatencyEvolver
    loader = _loader()
    evolver = LatencyEvolver(
        n_pop=4, n_generations=2, t_local=1,
        input_size=784, n_classes=10, verbose=False,
        target_device="cpu",
    )
    best, history = evolver.run(loader, loader)
    assert best is not None
    assert len(history) == 2
    assert hasattr(best, "latency_ms")
    assert best.latency_ms > 0


def test_latency_evolver_rpi4_scale():
    from dnaty.evolution.evolver import LatencyEvolver
    from dnaty.utils.hw_detect import latency_scale
    loader = _loader()

    evolver_cpu = LatencyEvolver(
        n_pop=3, n_generations=1, t_local=1,
        input_size=784, n_classes=10, verbose=False,
        target_device="cpu",
    )
    evolver_rpi = LatencyEvolver(
        n_pop=3, n_generations=1, t_local=1,
        input_size=784, n_classes=10, verbose=False,
        target_device="rpi4",
    )
    evolver_cpu.run(loader, loader)
    evolver_rpi.run(loader, loader)

    cpu_lat = evolver_cpu.population[0].latency_ms
    rpi_lat = evolver_rpi.population[0].latency_ms

    # RPi4 latency should be ~9x CPU (allow 3x–30x range for noise + different archs)
    ratio = rpi_lat / max(cpu_lat, 1e-9)
    assert 3 <= ratio <= 30, f"Expected RPi4/CPU ratio 3–30x, got {ratio:.1f}x"


def test_latency_evolver_pareto_front():
    from dnaty.evolution.evolver import LatencyEvolver
    loader = _loader()
    evolver = LatencyEvolver(
        n_pop=5, n_generations=2, t_local=1,
        input_size=784, n_classes=10, verbose=False,
        target_device="rpi4",
    )
    evolver.run(loader, loader)
    front = evolver.pareto_front()
    assert len(front) == 5
    assert all("accuracy" in p and "latency_ms" in p and "params" in p for p in front)
    assert all(p["latency_ms"] > 0 for p in front)


# ── compress(target="latency") ────────────────────────────────────────────────

def test_compress_target_latency_cpu():
    from dnaty.compress import compress
    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )
    loader = _loader()
    result = compress(model, loader, target="latency", hw_target="cpu",
                      n_generations=2, n_pop=4, verbose=False, finetune_epochs=0)
    assert result is not None
    assert result.accuracy >= 0
    assert result.arch is not None


def test_compress_target_latency_rpi4():
    from dnaty.compress import compress
    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )
    loader = _loader()
    result = compress(model, loader, target="latency", hw_target="rpi4",
                      n_generations=2, n_pop=4, verbose=False, finetune_epochs=0)
    assert result is not None
    assert result.accuracy >= 0


def test_compress_target_flops_unchanged():
    """Existing target='flops' (default) still works — no regression."""
    from dnaty.compress import compress
    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )
    loader = _loader()
    result = compress(model, loader, target="flops", target_flops=0.5,
                      n_generations=2, n_pop=4, verbose=False, finetune_epochs=0)
    assert result is not None
    assert result.accuracy >= 0


# ── N:M Sparsity ─────────────────────────────────────────────────────────────

def test_apply_nm_sparsity_2_4():
    from dnaty.utils.sparsity import apply_nm_sparsity, sparsity_stats
    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )
    apply_nm_sparsity(model, n=2, m=4)
    stats = sparsity_stats(model)
    # 2:4 = 50% sparsity on all eligible layers
    assert 45 <= stats["global_sparsity_pct"] <= 55, f"Expected ~50%, got {stats['global_sparsity_pct']}%"


def test_sparsity_stats_keys():
    from dnaty.utils.sparsity import sparsity_stats
    model = _tiny_mlp()
    stats = sparsity_stats(model)
    assert "global_sparsity_pct" in stats
    assert "total_weights" in stats
    assert "zero_weights" in stats
    assert "layer_stats" in stats
    assert len(stats["layer_stats"]) >= 1


def test_compress_with_sparsity():
    from dnaty.compress import compress
    from dnaty.utils.sparsity import sparsity_stats
    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )
    loader = _loader()
    result = compress(model, loader, target_flops=0.5, sparsity="2:4",
                      n_generations=2, n_pop=4, verbose=False, finetune_epochs=0)
    stats = sparsity_stats(result.model)
    assert stats["global_sparsity_pct"] > 40


# ── QuantAwareEvolver ─────────────────────────────────────────────────────────

def test_quant_aware_evolver_runs():
    from dnaty.evolution.evolver import QuantAwareEvolver
    loader = _loader()
    evolver = QuantAwareEvolver(
        n_pop=3, n_generations=2, t_local=1,
        input_size=784, n_classes=10, verbose=False,
    )
    best, history = evolver.run(loader, loader)
    assert best is not None
    assert len(history) == 2
    assert 0 <= best.acc <= 1


def test_compress_quant_aware():
    from dnaty.compress import compress
    model = nn.Sequential(
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )
    loader = _loader()
    result = compress(model, loader, quant_aware=True,
                      n_generations=2, n_pop=3, verbose=False, finetune_epochs=0)
    assert result is not None
    assert 0 <= result.accuracy <= 1


# ── Public API exports ────────────────────────────────────────────────────────

def test_public_exports():
    import dnaty
    assert hasattr(dnaty, "LatencyEvolver")
    assert hasattr(dnaty, "measure_latency")
    assert hasattr(dnaty, "detect_hw")
    assert hasattr(dnaty, "latency_scale")
    assert hasattr(dnaty, "estimate_latency")
    assert hasattr(dnaty, "LatencyPredictor")
    assert dnaty.__version__.startswith("2.")
    assert hasattr(dnaty, "QuantAwareEvolver")
    assert hasattr(dnaty, "apply_nm_sparsity")
    assert hasattr(dnaty, "sparsity_stats")
