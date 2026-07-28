"""
Regression tests for the algorithm bug-hunt fixes:

1. BatchNorm size-1 batch/chunk no longer crashes local_train()/evaluate().
2. CNN prune_channels preserves trained weights (unmodified blocks copied
   exactly, pruned block keeps its overlapping channel slice).
3. NSGA-II crowding_distance ignores a degenerate (constant) objective instead
   of handing it spurious infinite distances.
4. local_train no longer adds a zero-gradient structural-cost constant to the
   loss (it was a no-op that only offset the reported loss).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty.core.arch import DynamicMLP
from dnaty.core.arch_cnn import DynamicCNN
from dnaty.core.individual import Individual
from dnaty.core.memory import EpisodicMemory
from dnaty.training.local_train import local_train, evaluate
from dnaty.operators.mutations_cnn import prune_channels
from dnaty.evolution.selection import crowding_distance


# ── 1. BatchNorm size-1 batch no longer crashes ─────────────────────────────

def _mlp_ind():
    return Individual(DynamicMLP([20, 64, 32], ["relu", "relu"], 3))


@pytest.mark.parametrize("n", [257, 513, 1])  # N % 256 == 1 triggers a size-1 tail
def test_evaluate_survives_size1_tail(n):
    ind = _mlp_ind()
    X = torch.randn(n, 20)
    y = torch.randint(0, 3, (n,))
    dl = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=False)
    acc, loss = evaluate(ind, dl, "cpu")  # use_train_mode=True (NAS default)
    assert 0.0 <= acc <= 1.0
    assert np.isfinite(loss)


def test_local_train_survives_size1_tail():
    ind = _mlp_ind()
    X = torch.randn(257, 20)
    y = torch.randint(0, 3, (257,))
    dl = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=False)
    lb, la, gn = local_train(ind, dl, n_epochs=1, device="cpu")
    assert all(np.isfinite(v) for v in (lb, la, gn))


def test_evaluate_fastdataset_chunk_size1():
    """FastDataset path chunks by 2048; len(val) % 2048 == 1 gave a size-1 chunk."""
    class _FakeVal:
        def __init__(self, n):
            self.vx = torch.randn(n, 20)
            self.vy = torch.randint(0, 3, (n,))
        def get_val(self):
            return self.vx, self.vy

    ind = _mlp_ind()
    acc, loss = evaluate(ind, _FakeVal(4097), "cpu")
    assert 0.0 <= acc <= 1.0 and np.isfinite(loss)


def test_bn_mode_restored_after_tiny_forward():
    """The size-1 fallback must leave BatchNorm layers back in train mode."""
    ind = _mlp_ind()
    X = torch.randn(257, 20)
    y = torch.randint(0, 3, (257,))
    dl = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=False)
    local_train(ind, dl, n_epochs=1, device="cpu")
    ind.model.train()
    evaluate(ind, dl, "cpu")
    bns = [m for m in ind.model.modules() if isinstance(m, nn.BatchNorm1d)]
    assert bns and all(m.training for m in bns), "BN must be restored to train mode"


# ── 2. CNN prune_channels preserves trained weights ─────────────────────────

def test_prune_channels_preserves_unmodified_blocks():
    torch.manual_seed(0)
    np.random.seed(0)
    ind = Individual(DynamicCNN(), EpisodicMemory())
    with torch.no_grad():
        for p in ind.model.parameters():
            p.fill_(0.123)

    new_ind, ok = prune_channels(ind)
    assert ok and new_ind.last_op == "prune_channels"

    # Default CNN prunes an out_ch>32 block (block 1 or 2); block 0 is untouched
    # and must be copied verbatim, not re-initialised.
    w0 = new_ind.model.conv_layers[0].block[0].weight
    assert torch.allclose(w0, torch.full_like(w0, 0.123)), \
        "unmodified conv block must keep its trained weights"


def test_prune_channels_pruned_block_keeps_overlap():
    """The pruned block keeps its surviving channel slice (not random init)."""
    torch.manual_seed(1)
    np.random.seed(1)
    # Single big block so prune_channels is forced to pick it.
    cfg = [{"type": "conv", "in_ch": 3, "out_ch": 64, "stride": 1}]
    ind = Individual(DynamicCNN(conv_configs=cfg, fc_sizes=[64]), EpisodicMemory())
    old_w = ind.model.conv_layers[0].block[0].weight.detach().clone()

    new_ind, ok = prune_channels(ind)
    assert ok
    new_w = new_ind.model.conv_layers[0].block[0].weight
    o = min(old_w.shape[0], new_w.shape[0])
    assert new_w.shape[0] < old_w.shape[0], "out_ch should shrink"
    assert torch.allclose(new_w[:o], old_w[:o]), \
        "surviving output channels must be copied from the parent"


def test_prune_channels_still_valid_and_runs():
    ind = Individual(DynamicCNN(), EpisodicMemory())
    new_ind, ok = prune_channels(ind)
    assert ok and new_ind.model.is_valid()
    out = new_ind.model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)


# ── 3. NSGA-II crowding ignores a constant objective ────────────────────────

def test_crowding_ignores_constant_third_objective():
    # (acc, -cost, 0.0) — the third objective is the unused constant placeholder.
    fit = [(0.92, -1.2, 0.0), (0.94, -1.4, 0.0), (0.90, -1.0, 0.0)]
    front = [0, 1, 2]
    cd = crowding_distance(fit, front)
    # True 2-D boundaries are idx 2 (min acc) and idx 1 (max acc); idx 0 is
    # interior and must NOT be forced to infinity by the constant objective.
    assert cd[0] != float("inf"), "constant objective must not inflate an interior point"
    assert cd[1] == float("inf") and cd[2] == float("inf")


def test_crowding_two_objectives_unchanged():
    """Genuine 2-objective crowding still marks both real extremes as infinite."""
    fit = [(0.90, -1.0), (0.92, -1.2), (0.94, -1.4)]
    cd = crowding_distance(fit, [0, 1, 2])
    assert cd[0] == float("inf") and cd[2] == float("inf")
    assert cd[1] != float("inf")


# ── 4. local_train has no zero-gradient cost term ───────────────────────────

def test_local_train_runs_without_cost_penalty():
    """lambda1/lambda2 are accepted (backward compat) but do not affect training."""
    ind = _mlp_ind()
    X = torch.randn(256, 20)
    y = torch.randint(0, 3, (256,))
    dl = DataLoader(TensorDataset(X, y), batch_size=128, shuffle=False)
    lb, la, gn = local_train(ind, dl, n_epochs=2, lambda1=1e-2, lambda2=1e-2, device="cpu")
    assert all(np.isfinite(v) for v in (lb, la, gn))


# ── 5. anova_tukey handles independent groups of unequal size ───────────────

def test_anova_tukey_unequal_independent_groups():
    from dnaty.analysis.stats import anova_tukey
    # Different sizes used to crash the old paired-t-test post-hoc.
    groups = {
        "dnaty": [0.91, 0.92, 0.93, 0.94, 0.95],
        "ewc":   [0.80, 0.82, 0.81],
        "naive": [0.70, 0.71],
    }
    out = anova_tukey(groups)
    assert set(out["pairs"]) == {"dnaty vs ewc", "dnaty vs naive", "ewc vs naive"}
    for pair in out["pairs"].values():
        # Bonferroni-corrected p must be >= the raw p and stay a valid probability.
        assert 0.0 <= pair["p"] <= 1.0
        assert pair["p"] >= pair["p_uncorrected"] - 1e-9
