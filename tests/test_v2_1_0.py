"""
Tests for v2.1.0 features:

1. Transferable episodic memory (operator priors)
   - EpisodicMemory.to_prior / seed_from_prior round-trip and normalisation
   - save_prior / load_prior JSON round-trip (+ bare {op: score} tolerance)
   - seeding biases query_mutation_probs; weight=0 and degenerate priors are no-ops
   - DnatyEvolver(warm_start=...) seeds the shared memory
   - compress(warm_start=...) accepts a dict, a path, and an EpisodicMemory
   - result.export_memory / save_memory round-trips; determinism preserved

2. Pareto-front API
   - result.pareto_front is non-empty, non-dominated, and sorted by FLOPs
   - pareto_front_csv writes a readable CSV
   - backward-compatible CompressResult construction / dnaty.load
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dnaty
from dnaty import compress, save_prior, load_prior
from dnaty.core.memory import EpisodicMemory, Experience, PRIOR_FORMAT_VERSION
from dnaty.evolution.evolver import DnatyEvolver
from dnaty.operators.mutations import OPERATORS
from dnaty.result import CompressResult, load


# ── fixtures ─────────────────────────────────────────────────────────────────

def _tiny_task(n=600, d=20, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype("float32")
    y = ((X[:, 0] + 2 * X[:, 1] + rng.randn(n) * 0.3) > 0).astype("int64") \
        + (X[:, 2] > 0.5).astype("int64")
    return X, y


def _baseline_model(d=20, n_classes=3):
    return nn.Sequential(
        nn.Linear(d, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, n_classes),
    )


def _populated_memory(seed=0):
    """An EpisodicMemory with a clear operator preference baked in."""
    rng = np.random.RandomState(seed)
    mem = EpisodicMemory(decay_gamma=0.99)
    # Make "remove_neuron" clearly the strongest operator.
    for _ in range(30):
        mem.update(Experience("remove_neuron", delta_loss=-0.5,
                               gradient_norm=2.0, generation=0))
    for _ in range(5):
        mem.update(Experience("add_neuron", delta_loss=-0.1,
                               gradient_norm=1.0, generation=0))
    return mem


# ── 1. prior export / seed round-trip ────────────────────────────────────────

def test_to_prior_shape():
    mem = _populated_memory()
    prior = mem.to_prior()
    assert prior["version"] == PRIOR_FORMAT_VERSION
    assert prior["format"] == "dnaty.operator_prior"
    assert "remove_neuron" in prior["scores"]
    assert prior["scores"]["remove_neuron"] > prior["scores"]["add_neuron"]


def test_seed_from_prior_biases_probs():
    """Seeding a fresh memory must make the strong operator more likely."""
    src = _populated_memory()
    prior = src.to_prior()

    fresh = EpisodicMemory(decay_gamma=0.99)
    n = fresh.seed_from_prior(prior, weight=2.0)
    assert n > 0

    probs = fresh.query_mutation_probs(OPERATORS, tau=1.0)
    uniform = 1.0 / len(OPERATORS)
    assert probs["remove_neuron"] > uniform
    assert probs["remove_neuron"] > probs["add_neuron"]


def test_seed_weight_zero_is_noop():
    fresh = EpisodicMemory()
    assert fresh.seed_from_prior(_populated_memory().to_prior(), weight=0) == 0
    probs = fresh.query_mutation_probs(OPERATORS)
    # Empty scores -> uniform.
    assert all(abs(p - 1.0 / len(OPERATORS)) < 1e-9 for p in probs.values())


def test_seed_degenerate_prior_is_noop():
    """A prior where every operator scored identically carries no signal."""
    flat = {"scores": {op: 1.0 for op in OPERATORS}}
    fresh = EpisodicMemory()
    assert fresh.seed_from_prior(flat, weight=2.0) == 0


def test_higher_weight_sharper_prior():
    prior = _populated_memory().to_prior()
    gentle = EpisodicMemory(); gentle.seed_from_prior(prior, weight=1.0)
    strong = EpisodicMemory(); strong.seed_from_prior(prior, weight=4.0)
    p_gentle = gentle.query_mutation_probs(OPERATORS)["remove_neuron"]
    p_strong = strong.query_mutation_probs(OPERATORS)["remove_neuron"]
    assert p_strong > p_gentle


# ── 2. JSON persistence ──────────────────────────────────────────────────────

def test_save_load_prior_roundtrip(tmp_path):
    prior = _populated_memory().to_prior()
    p = tmp_path / "prior.json"
    save_prior(prior, str(p))
    loaded = load_prior(str(p))
    assert loaded["scores"] == pytest.approx(prior["scores"])


def test_load_prior_tolerates_bare_mapping(tmp_path):
    import json
    p = tmp_path / "bare.json"
    p.write_text(json.dumps({"remove_neuron": 3.0, "add_neuron": 1.0}))
    loaded = load_prior(str(p))
    assert "scores" in loaded
    assert loaded["scores"]["remove_neuron"] == 3.0


def test_memory_save_prior_method(tmp_path):
    p = tmp_path / "m.json"
    _populated_memory().save_prior(str(p))
    assert load_prior(str(p))["scores"]["remove_neuron"] > 0


# ── 3. evolver warm-start ────────────────────────────────────────────────────

def test_evolver_warm_start_seeds_memory():
    prior = _populated_memory().to_prior()
    ev = DnatyEvolver(n_pop=6, n_generations=2, input_size=20, n_classes=3,
                      verbose=False, warm_start=prior, warm_start_weight=2.0)
    assert ev.n_warm_started > 0
    probs = ev.shared_memory.query_mutation_probs(OPERATORS)
    assert probs["remove_neuron"] > 1.0 / len(OPERATORS)


def test_evolver_warm_start_rejects_bad_type():
    with pytest.raises(TypeError):
        DnatyEvolver(n_pop=4, n_generations=1, input_size=20, n_classes=3,
                     verbose=False, warm_start=12345)


def test_evolver_warm_start_accepts_memory_instance():
    ev = DnatyEvolver(n_pop=4, n_generations=1, input_size=20, n_classes=3,
                      verbose=False, warm_start=_populated_memory())
    assert ev.n_warm_started > 0


# ── 4. compress() integration ────────────────────────────────────────────────

def test_compress_produces_prior_and_front():
    X, y = _tiny_task()
    r = compress(_baseline_model(), (X, y), target_flops=0.5, n_generations=5,
                 n_pop=8, seed=42, finetune_epochs=1, verbose=False)
    assert r.operator_priors.get("scores")           # non-empty prior
    assert len(r.pareto_front) >= 1
    assert r.export_memory()["scores"]


def test_compress_warm_start_dict_path_memory(tmp_path):
    X, y = _tiny_task()
    r0 = compress(_baseline_model(), (X, y), target_flops=0.5, n_generations=5,
                  n_pop=8, seed=1, finetune_epochs=1, verbose=False)

    # (a) dict
    r_dict = compress(_baseline_model(), (X, y), target_flops=0.5, n_generations=4,
                      n_pop=8, seed=1, finetune_epochs=1, verbose=False,
                      warm_start=r0.export_memory())
    assert r_dict.accuracy >= 0

    # (b) path
    p = tmp_path / "prior.json"
    r0.save_memory(str(p))
    r_path = compress(_baseline_model(), (X, y), target_flops=0.5, n_generations=4,
                      n_pop=8, seed=1, finetune_epochs=1, verbose=False,
                      warm_start=str(p))
    assert r_path.accuracy >= 0


def test_compress_warm_start_is_deterministic():
    X, y = _tiny_task()
    prior = _populated_memory().to_prior()
    kw = dict(target_flops=0.5, n_generations=5, n_pop=8, seed=7,
              finetune_epochs=1, verbose=False, warm_start=prior)
    a = compress(_baseline_model(), (X, y), **kw)
    b = compress(_baseline_model(), (X, y), **kw)
    assert a.arch == b.arch
    assert a.accuracy == pytest.approx(b.accuracy, abs=1e-6)
    assert a.compressed_flops == b.compressed_flops


# ── 5. Pareto front correctness ──────────────────────────────────────────────

def test_pareto_front_is_non_dominated_and_sorted():
    X, y = _tiny_task()
    r = compress(_baseline_model(), (X, y), target_flops=0.5, n_generations=6,
                 n_pop=10, seed=3, finetune_epochs=1, verbose=False)
    pf = r.pareto_front
    assert pf
    # sorted by FLOPs ascending
    flops = [p["flops"] for p in pf]
    assert flops == sorted(flops)
    # no point is dominated by another
    for a in pf:
        for b in pf:
            if a is b:
                continue
            strictly_better = (
                b["accuracy"] >= a["accuracy"] and b["flops"] <= a["flops"]
                and (b["accuracy"] > a["accuracy"] or b["flops"] < a["flops"])
            )
            assert not strictly_better, f"{b} dominates {a}"
    # each entry carries the fields the API promises
    for p in pf:
        assert set(p) >= {"arch", "accuracy", "flops", "params", "flops_reduction_pct"}


def test_pareto_front_csv(tmp_path):
    X, y = _tiny_task()
    r = compress(_baseline_model(), (X, y), target_flops=0.5, n_generations=5,
                 n_pop=8, seed=5, finetune_epochs=1, verbose=False)
    p = tmp_path / "front.csv"
    r.pareto_front_csv(str(p))
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert len(rows) == len(r.pareto_front)
    assert "accuracy" in rows[0] and "flops" in rows[0]


def test_pareto_summary_string():
    X, y = _tiny_task()
    r = compress(_baseline_model(), (X, y), target_flops=0.5, n_generations=4,
                 n_pop=8, seed=9, finetune_epochs=1, verbose=False)
    s = r.pareto_summary()
    assert "non-dominated" in s


# ── 6. backward compatibility ────────────────────────────────────────────────

def test_compressresult_without_new_fields():
    """Old-style construction (no pareto_front / operator_priors) still works."""
    m = _baseline_model()
    r = CompressResult(
        model=m, original_flops=10, compressed_flops=5, original_params=10,
        compressed_params=5, accuracy=0.9, flops_reduction=0.5, generations=1,
    )
    assert r.pareto_front == []
    assert r.operator_priors == {}


def test_loaded_result_save_memory_raises(tmp_path):
    """A result reconstructed by dnaty.load() has no prior -> save_memory errors."""
    from dnaty.core.arch import DynamicMLP
    m = DynamicMLP([20, 32], n_classes=3)
    r = CompressResult(
        model=m, original_flops=10, compressed_flops=5, original_params=10,
        compressed_params=5, accuracy=0.9, flops_reduction=0.5, generations=1,
    )
    pt = tmp_path / "m.pt"
    r.save(str(pt))
    reloaded = load(str(pt))
    assert reloaded.operator_priors == {}
    with pytest.raises(ValueError):
        reloaded.save_memory(str(tmp_path / "nope.json"))
