"""Tests for v1.2.x features: FLOPs-guided swap, budget-aware CnnEvolver, auto_retrigger, export_failure_report."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty.monitoring.drift import DriftDetector
from dnaty.monitoring.tracker import ProductionTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker(n_features=8):
    model = nn.Sequential(nn.Linear(n_features, 16), nn.ReLU(), nn.Linear(16, 3))
    det = DriftDetector(psi_threshold=0.2)
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, (300, n_features)).astype(np.float32)
    det.fit(train)
    tracker = ProductionTracker(model, drift_detector=det, drift_every=50, device="cpu")
    return tracker, train


# ---------------------------------------------------------------------------
# Item 4: swap_conv_to_dw picks highest-FLOPs layer
# ---------------------------------------------------------------------------

def test_swap_conv_to_dw_targets_highest_flops_layer():
    """swap_conv_to_dw should swap the layer with largest in_ch*out_ch."""
    from dnaty.core.arch_cnn import DynamicCNN
    from dnaty.core.individual import Individual
    from dnaty.core.memory import EpisodicMemory
    from dnaty.operators.mutations_cnn import swap_conv_to_dw

    # Two eligible conv layers: first has 16*16=256, second has 32*64=2048 (larger)
    configs = [
        {"type": "conv", "in_ch": 3,  "out_ch": 16, "stride": 1},
        {"type": "conv", "in_ch": 16, "out_ch": 32, "stride": 1},
        {"type": "conv", "in_ch": 32, "out_ch": 64, "stride": 1},
    ]
    model = DynamicCNN(configs, fc_sizes=[64], n_classes=10, in_channels=3)
    ind = Individual(model, EpisodicMemory())

    # Run 10 times — deterministic: should always pick index 2 (32*64=2048 is max)
    swapped_indices = set()
    for _ in range(10):
        new_ind, ok = swap_conv_to_dw(ind)
        if ok:
            for i, c in enumerate(new_ind.model.conv_configs):
                if c["type"] == "depthwise":
                    swapped_indices.add(i)

    # Should always be index 2 (highest FLOPs), never index 0 or 1
    assert swapped_indices == {2}, f"Expected only index 2 swapped, got {swapped_indices}"


def test_swap_conv_to_dw_no_eligible_layers():
    """swap_conv_to_dw returns False when all layers are already depthwise."""
    from dnaty.core.arch_cnn import DynamicCNN
    from dnaty.core.individual import Individual
    from dnaty.core.memory import EpisodicMemory
    from dnaty.operators.mutations_cnn import swap_conv_to_dw

    configs = [
        {"type": "depthwise", "in_ch": 3,  "out_ch": 32, "stride": 1},
        {"type": "conv",      "in_ch": 32, "out_ch": 4,  "stride": 2},  # out_ch < 8? No, 4 < 8
    ]
    model = DynamicCNN(configs, fc_sizes=[32], n_classes=10, in_channels=3)
    ind = Individual(model, EpisodicMemory())
    _, ok = swap_conv_to_dw(ind)
    assert not ok


# ---------------------------------------------------------------------------
# Item 4: CnnEvolver budget-aware boost
# ---------------------------------------------------------------------------

def test_cnn_evolver_has_target_flops_param():
    """CnnEvolver accepts target_flops and budget_boost_factor."""
    from dnaty.evolution.evolver import CnnEvolver
    ev = CnnEvolver(n_pop=4, n_generations=1, target_flops=0.3, budget_boost_factor=5.0)
    assert ev.target_flops == 0.3
    assert ev.budget_boost_factor == 5.0
    assert ev._baseline_flops is None  # not set until population is initialised


def test_cnn_evolver_default_target_flops():
    """CnnEvolver defaults to target_flops=0.5 (same as compress_cnn default)."""
    from dnaty.evolution.evolver import CnnEvolver
    ev = CnnEvolver()
    assert ev.target_flops == 0.5


def test_cnn_evolver_baseline_set_after_init():
    """_baseline_flops is captured right after _init_population."""
    from dnaty.evolution.evolver import CnnEvolver
    ev = CnnEvolver(n_pop=3, n_generations=1, n_classes=10)
    ev._init_population()
    assert ev._baseline_flops is not None
    assert ev._baseline_flops > 0


# ---------------------------------------------------------------------------
# Item 5: auto_retrigger
# ---------------------------------------------------------------------------

def test_auto_retrigger_does_not_fire_below_threshold():
    tracker, train = _make_tracker()
    called = []

    def fake_compress(data):
        called.append(True)
        return None

    # 0 consecutive drifts — should not trigger
    triggered = tracker.auto_retrigger(fake_compress, train, consecutive_drifts=3)
    assert not triggered
    assert not called


def test_auto_retrigger_fires_when_threshold_met():
    tracker, train = _make_tracker()

    # Manually set consecutive drift count to threshold
    tracker._consecutive_drift_count = 3

    new_model_calls = []

    def fake_compress(data):
        new_model_calls.append(data)
        return nn.Sequential(nn.Linear(8, 3))

    triggered = tracker.auto_retrigger(fake_compress, train, consecutive_drifts=3)
    assert triggered
    assert len(new_model_calls) == 1
    # Counter should be reset
    assert tracker._consecutive_drift_count == 0


def test_auto_retrigger_calls_on_trigger_callback():
    tracker, train = _make_tracker()
    tracker._consecutive_drift_count = 2

    callback_calls = []

    triggered = tracker.auto_retrigger(
        compress_fn=lambda data: None,
        train_data=train,
        consecutive_drifts=2,
        on_trigger=lambda t: callback_calls.append(t),
    )
    assert triggered
    assert len(callback_calls) == 1


def test_auto_retrigger_refits_baseline_after_trigger():
    tracker, train = _make_tracker()
    tracker._consecutive_drift_count = 1

    tracker.auto_retrigger(
        compress_fn=lambda data: None,
        train_data=train,
        consecutive_drifts=1,
    )
    # Baseline should still be fitted (re-fitted in auto_retrigger)
    assert tracker._baseline_fitted


# ---------------------------------------------------------------------------
# Item 6: export_failure_report
# ---------------------------------------------------------------------------

def test_export_failure_report_empty():
    """Report works with zero failures."""
    tracker, _ = _make_tracker()
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "report.json")
        report = tracker.export_failure_report(path)

    assert report["n_failures"] == 0
    assert report["pca_coords"] == []
    assert report["samples"] == []


def test_export_failure_report_writes_json():
    tracker, _ = _make_tracker()
    rng = np.random.default_rng(42)
    n = 20
    inputs = rng.normal(0, 1, (n, 8)).astype(np.float32)
    preds = np.zeros(n, dtype=int)
    gt = np.ones(n, dtype=int)  # all wrong

    tracker.record_outcome(preds, gt, inputs=inputs)
    assert len(tracker._failure_buffer) == n

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "report.json")
        report = tracker.export_failure_report(path)
        assert Path(path).exists()
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))

    assert loaded["n_failures"] == n
    assert loaded["failure_rate"] == pytest.approx(1.0)
    assert len(loaded["samples"]) == n
    assert len(loaded["pca_coords"]) == n
    assert all(len(c) == 2 for c in loaded["pca_coords"])


def test_export_failure_report_class_breakdown():
    tracker, _ = _make_tracker()
    inputs = np.zeros((3, 8), dtype=np.float32)

    # Three failures: 0→1, 0→1, 0→2
    tracker.record_outcome(np.array([1, 1, 2]), np.array([0, 0, 0]), inputs=inputs)

    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "report.json")
        report = tracker.export_failure_report(path)

    assert report["class_breakdown"]["0->1"] == 2
    assert report["class_breakdown"]["0->2"] == 1


def test_export_failure_report_sqlite():
    """Failure report is persisted to SQLite when db_uri is provided."""
    import sqlite3

    tracker, _ = _make_tracker()
    inputs = np.ones((5, 8), dtype=np.float32)
    tracker.record_outcome(np.zeros(5, dtype=int), np.ones(5, dtype=int), inputs=inputs)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        db_path = str(Path(d) / "failures.db")
        json_path = str(Path(d) / "report.json")
        tracker.export_failure_report(json_path, db_uri=db_path)

        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute("SELECT COUNT(*) FROM failures").fetchone()
        finally:
            conn.close()
        assert rows[0] == 5


def test_record_outcome_without_inputs_does_not_store():
    """record_outcome without inputs should not fill the failure buffer."""
    tracker, _ = _make_tracker()
    preds = np.zeros(10, dtype=int)
    gt = np.ones(10, dtype=int)
    tracker.record_outcome(preds, gt)  # no inputs arg
    assert len(tracker._failure_buffer) == 0


def test_reset_clears_failure_buffer():
    tracker, _ = _make_tracker()
    inputs = np.zeros((3, 8), dtype=np.float32)
    tracker.record_outcome(np.array([1, 1, 1]), np.array([0, 0, 0]), inputs=inputs)
    assert len(tracker._failure_buffer) == 3
    tracker.reset()
    assert len(tracker._failure_buffer) == 0
    assert tracker._consecutive_drift_count == 0


import pytest
