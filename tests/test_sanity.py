"""Fast sanity tests for core dNaty components."""
from __future__ import annotations

import numpy as np
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty.analysis.cl_metrics import compute_cl_metrics
from dnaty.core.arch import DynamicMLP
from dnaty.core.individual import Individual
from dnaty.core.memory import EpisodicMemory, Experience
from dnaty.evolution.selection import fast_non_dominated_sort
from dnaty.operators.mutations import OPERATORS, apply_operator
from dnaty.training.local_train import evaluate, local_train


def test_episodic_memory_prefers_successful_operator() -> None:
    mem = EpisodicMemory(max_size=100, decay_gamma=0.99)
    for i in range(10):
        mem.update(Experience("add_neuron", delta_loss=-0.05, gradient_norm=1.0, generation=i))
    for i in range(10):
        mem.update(Experience("remove_neuron", delta_loss=0.05, gradient_norm=1.0, generation=i))

    probs = mem.query_mutation_probs(OPERATORS)
    assert probs["add_neuron"] > probs["remove_neuron"]
    assert Experience("add_neuron", delta_loss=0.1, gradient_norm=5.0, generation=0).impact == 0.0

    mem2 = EpisodicMemory(max_size=5)
    for i in range(10):
        mem2.update(Experience("add_neuron", delta_loss=-0.01, gradient_norm=1.0, generation=i))
    assert len(mem2.experiences) <= 5


def test_nsga2_fronts_known_points() -> None:
    fitnesses = [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5), (0.2, 0.2)]
    fronts = fast_non_dominated_sort(fitnesses)
    assert 3 not in fronts[0]
    assert len(fronts) >= 2


def test_structural_operators_keep_forward_valid() -> None:
    for op in OPERATORS:
        model = DynamicMLP([784, 64, 32], ["relu", "relu"], 10)
        ind = Individual(model)
        new_ind, _ = apply_operator(ind, op)
        out = new_ind.model(torch.randn(4, 784))
        assert out.shape == (4, 10), op


def test_local_training_reduces_loss_on_synthetic_batch() -> None:
    torch.manual_seed(0)
    x = torch.randn(200, 784)
    y = torch.randint(0, 10, (200,))
    loader = DataLoader(TensorDataset(x, y), batch_size=32)
    ind = Individual(DynamicMLP([784, 64, 32], ["relu", "relu"], 10))

    loss_before, loss_after, grad_norm = local_train(ind, loader, n_epochs=3)
    assert grad_norm > 0
    assert loss_before - loss_after >= -1e-4
    acc, _ = evaluate(ind, loader)
    assert 0.0 <= acc <= 1.0


def test_cl_metrics_for_known_forgetting_matrix() -> None:
    r = np.array(
        [
            [0.90, 0.00, 0.00, 0.00, 0.00],
            [0.85, 0.88, 0.00, 0.00, 0.00],
            [0.82, 0.85, 0.87, 0.00, 0.00],
            [0.80, 0.83, 0.85, 0.89, 0.00],
            [0.78, 0.81, 0.83, 0.87, 0.91],
        ]
    )
    baselines = np.array([0.90, 0.88, 0.87, 0.89, 0.91])
    metrics = compute_cl_metrics(r, baselines)
    assert metrics["BWT"] < 0
    assert metrics["FM"] > 0


def main() -> None:
    print("Running dNaty sanity tests...")
    test_episodic_memory_prefers_successful_operator()
    test_nsga2_fronts_known_points()
    test_structural_operators_keep_forward_valid()
    test_local_training_reduces_loss_on_synthetic_batch()
    test_cl_metrics_for_known_forgetting_matrix()
    print("TODOS OS TESTES PASSARAM OK")


if __name__ == "__main__":
    main()
