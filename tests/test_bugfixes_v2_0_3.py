"""
Regression tests for the v2.0.3 bug-hunt fixes:

1. Skip-connection projections registered as submodules (trained, saved, moved).
2. quant_aware=True with target="latency" actually uses INT8 fitness.
3. parent_acc travels on the mutant (proxy_filter misattribution).
4. Final accuracy measured in eval() mode; evaluate() restores train/eval mode.
5. Latency predictor features include every hidden layer.
6. Adaptive accuracy floor in compress() selection.
7. DriftDetector counts out-of-range samples instead of dropping them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty.core.arch import DynamicMLP
from dnaty.core.individual import Individual
from dnaty.operators.mutations import add_skip, add_residual, add_neuron
from dnaty.result import CompressResult, load


def _skip_model(seed: int = 0):
    """DynamicMLP with at least one projected skip connection."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = DynamicMLP([20, 32, 48], n_classes=3)
    ind = Individual(m)
    for _ in range(100):
        new_ind, applied = add_skip(ind)
        if applied and any(pi is not None for *_, pi in new_ind.model.skip_connections):
            return new_ind
    pytest.fail("could not create a projected skip connection in 100 attempts")


# ── 1. Skip connections are real submodules ──────────────────────────────────

def test_skip_proj_registered_in_parameters():
    ind = _skip_model()
    model = ind.model
    param_ids = {id(p) for p in model.parameters()}
    for *_, proj_idx in model.skip_connections:
        if proj_idx is not None:
            proj = model.skip_projs[proj_idx]
            assert all(id(p) in param_ids for p in proj.parameters()), \
                "skip projection weights must be visible to the optimizer"


def test_skip_proj_in_state_dict():
    ind = _skip_model()
    keys = list(ind.model.state_dict().keys())
    assert any(k.startswith("skip_projs") for k in keys)


def test_skip_survives_save_load_roundtrip():
    ind = _skip_model()
    model = ind.model.eval()
    x = torch.randn(8, 20)
    out_before = model(x)

    r = CompressResult(
        model=model, original_flops=1, compressed_flops=1, original_params=1,
        compressed_params=1, accuracy=0.5, flops_reduction=0.0, generations=1,
        arch=[32, 48],
    )
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "skip.pt")
        r.save(p)
        r2 = load(p)
    assert r2.model.skip_connections == model.skip_connections
    out_after = r2.model(x)
    assert torch.allclose(out_before, out_after, atol=1e-6), \
        "save/load must preserve skip-connection outputs"


def test_skip_counted_in_params_and_flops():
    ind = _skip_model()
    model = ind.model
    plain = DynamicMLP(model.layer_sizes, model.activations, model.n_classes)
    assert model.count_params() > plain.count_params()
    assert model.count_flops() > plain.count_flops()


def test_valid_skip_preserved_across_rebuild():
    """A rebuild mutation must keep skips whose endpoint sizes are unchanged."""
    torch.manual_seed(1)
    np.random.seed(1)
    # extra hidden layer (64) NOT part of the 32/32 pair, so some mutations
    # can hit it without touching the skip endpoints
    m = DynamicMLP([20, 32, 32, 48, 64], n_classes=3)
    ind = Individual(m)
    new_ind, applied = add_residual(ind)  # identity skip between the two 32s
    assert applied and new_ind.model.skip_connections
    # add_neuron on a DIFFERENT layer must not drop the identity skip; try many
    # seeds — at least one mutation should leave the 32/32 pair intact.
    kept = False
    for s in range(30):
        np.random.seed(s)
        reb, ok = add_neuron(new_ind)
        if ok and reb.model.skip_connections:
            kept = True
            break
    assert kept, "dimensionally-valid skips must survive rebuild mutations"


# ── 2. quant_aware + target=latency ──────────────────────────────────────────

def test_quant_latency_evolver_exists_and_combines():
    from dnaty.evolution.evolver import (
        QuantLatencyEvolver, QuantAwareEvolver, LatencyEvolver,
    )
    assert issubclass(QuantLatencyEvolver, QuantAwareEvolver)
    assert issubclass(QuantLatencyEvolver, LatencyEvolver)
    # INT8 eval comes from QuantAware; latency fitness from Latency
    assert QuantLatencyEvolver._eval_population is QuantAwareEvolver._eval_population
    assert QuantLatencyEvolver._fitness is LatencyEvolver._fitness


def test_compress_latency_routes_quant_aware(monkeypatch):
    """compress(target='latency', quant_aware=True) must instantiate QuantLatencyEvolver."""
    import importlib
    compress_mod = importlib.import_module("dnaty.compress")
    from dnaty.evolution import evolver as evolver_mod

    used = {}

    class _Spy(evolver_mod.QuantLatencyEvolver):
        def __init__(self, *a, **k):
            used["cls"] = "quant"
            raise RuntimeError("spy stop")

    monkeypatch.setattr(evolver_mod, "QuantLatencyEvolver", _Spy)
    model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
    X = torch.randn(64, 10)
    y = torch.randint(0, 2, (64,))
    with pytest.raises(RuntimeError, match="spy stop"):
        compress_mod.compress(model, (X, y), target="latency", quant_aware=True,
                              n_generations=1, n_pop=2, verbose=False,
                              finetune_epochs=0)
    assert used.get("cls") == "quant"


# ── 3. parent_acc misattribution ─────────────────────────────────────────────

def test_mutants_carry_parent_acc():
    from dnaty.evolution.evolver import DnatyEvolver
    ev = DnatyEvolver(n_pop=4, n_generations=1, input_size=10, n_classes=2,
                      init_hidden=[16], verbose=False)
    ev._init_population()
    for i, ind in enumerate(ev.population):
        ind.acc = 0.1 * i  # distinct parent accuracies
    mutated = ev._mutate_population(ev.population)
    for parent, child in zip(ev.population, mutated):
        assert child.parent_acc == parent.acc


def test_update_memory_uses_parent_acc_not_position():
    from dnaty.evolution.evolver import DnatyEvolver
    ev = DnatyEvolver(n_pop=2, n_generations=1, input_size=10, n_classes=2,
                      init_hidden=[16], verbose=False)
    ev._init_population()
    child = ev.population[0].clone()
    child.last_op = "add_neuron"
    child.acc = 0.9
    child.parent_acc = 0.5          # real parent: big improvement
    # positional prev_accs would say the parent already had 0.95 (no improvement)
    delta = ev._update_memory([child], prev_best_acc=0.95,
                              grad_norms=[1.0], prev_accs=[0.95])
    assert delta > 0, "improvement must be credited via parent_acc, not list position"


# ── 4. evaluate(): eval-mode option + mode restoration ───────────────────────

def test_evaluate_restores_model_mode():
    from dnaty.training.local_train import evaluate
    torch.manual_seed(0)
    m = DynamicMLP([10, 16], n_classes=2)
    ind = Individual(m)
    X = torch.randn(32, 10)
    y = torch.randint(0, 2, (32,))
    from torch.utils.data import DataLoader, TensorDataset
    loader = DataLoader(TensorDataset(X, y), batch_size=16)

    m.eval()
    evaluate(ind, loader, "cpu")                       # train-mode eval
    assert not ind.model.training, "evaluate() must restore the caller's mode"
    m.train()
    evaluate(ind, loader, "cpu", use_train_mode=False)  # eval-mode eval
    assert ind.model.training


def test_compress_returns_model_in_eval_mode():
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
    X = torch.randn(128, 10).numpy()
    y = torch.randint(0, 2, (128,)).numpy()
    from dnaty import compress
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = compress(model, (X, y), n_generations=1, n_pop=2,
                          verbose=False, finetune_epochs=0)
    assert not result.model.training, "returned model must be in eval() mode"
    assert 0.0 <= result.accuracy <= 1.0


# ── 5. Latency predictor features ────────────────────────────────────────────

def test_predictor_widths_include_all_hidden_layers():
    from dnaty.evolution.evolver import LatencyEvolver
    ev = LatencyEvolver.__new__(LatencyEvolver)  # skip heavy __init__
    ev.input_size = 784
    ev._scale = 1.0

    captured = {}

    class _FakePredictor:
        def predict_ms(self, feats):
            captured.update(feats)
            return 1.0

    ev._predictor = _FakePredictor()
    m = DynamicMLP([784, 256, 128, 64], n_classes=10)
    ind = Individual(m)
    ev._predict_latency_ms(ind)
    assert captured["widths"] == [256, 128, 64], \
        "widths must include every hidden layer (matches build_latency_dataset)"


# ── 7. DriftDetector out-of-range samples ────────────────────────────────────

def test_drift_detector_counts_out_of_range_samples():
    from dnaty.monitoring import DriftDetector
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, size=(2000, 3))
    det = DriftDetector().fit(base)

    shifted = rng.normal(50, 1, size=(500, 3))   # entirely out of baseline range
    mild    = rng.normal(0.5, 1, size=(500, 3))  # partial shift

    psi_shifted = det.score(shifted)["psi_mean"]
    psi_mild    = det.score(mild)["psi_mean"]
    assert det.score(shifted)["drifted"]
    assert psi_shifted > psi_mild, \
        "a total distribution shift must score at least as high as a mild one"
