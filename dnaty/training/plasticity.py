"""
Plasticity preservation for lifelong edge inference (v2.2.0).

Deep networks trained continually lose the ability to learn new things over
time ("plasticity loss", Dohare et al., Nature 2024). The continual-backprop
remedy maps directly onto dNATY's machinery: track per-neuron *contribution
utility*, and periodically reinitialise the most dormant hidden units so the
network keeps free capacity for new data — without disturbing what it already
computes.

Three pieces, usable separately or via PlasticityController:

  NeuronUtilityTracker  -- EMA of |activation| x ||outgoing weights|| per
                           hidden unit (contribution utility).
  reinit_dormant        -- selective reinit of the lowest-utility units.
                           Incoming weights get a fresh Kaiming init; outgoing
                           weights are ZEROED, so the reborn unit is silent —
                           the only change to the model's function is the
                           removal of the dormant unit's old contribution,
                           which is negligible by construction (that is what
                           low utility means).
  effective_rank        -- entropy-based effective rank of the hidden feature
                           matrix (Roy & Vetterli), to monitor rank collapse.

Usage:
    from dnaty.training.plasticity import PlasticityController

    ctrl = PlasticityController(model, reinit_fraction=0.05, check_every=100)
    for xb, yb in stream:
        ctrl.observe(xb)          # cheap: one forward with hooks
        ...train step...
        n = ctrl.maybe_reinit()   # fires every `check_every` observes
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


def _hidden_triples(model: nn.Module) -> list[tuple[nn.Linear, nn.BatchNorm1d, nn.Module]]:
    """(Linear, BatchNorm1d, activation) triple per hidden layer of a DynamicMLP."""
    sizes = getattr(model, "layer_sizes", None)
    net = getattr(model, "net", None)
    if sizes is None or net is None or len(sizes) < 2:
        raise TypeError(
            "plasticity utilities require a DynamicMLP-style model "
            "(layer_sizes + net of [Linear, BN, Act] triples)"
        )
    triples = []
    for j in range(len(sizes) - 1):
        triples.append((net[3 * j], net[3 * j + 1], net[3 * j + 2]))
    return triples


def _outgoing_linears(model: nn.Module, hidden_idx: int) -> list[tuple[nn.Linear, bool]]:
    """Every Linear that consumes hidden layer `hidden_idx`'s output.

    Returns (linear, is_main_path). The main consumer is the next hidden
    Linear (or the classifier for the last hidden layer); skip projections
    whose source is this layer also consume it.
    """
    n_hidden = len(model.layer_sizes) - 1
    outs: list[tuple[nn.Linear, bool]] = []
    if hidden_idx + 1 < n_hidden:
        outs.append((model.net[3 * (hidden_idx + 1)], True))
    else:
        outs.append((model.net[-1], True))  # classifier
    # layer_outputs[0] is the input, layer_outputs[i+1] is hidden i's output,
    # so a skip with src == hidden_idx + 1 reads this layer.
    for src, _dst, proj_idx in getattr(model, "skip_connections", []):
        if src == hidden_idx + 1 and proj_idx is not None:
            outs.append((model.skip_projs[proj_idx], False))
    return outs


def _identity_skip_sources(model: nn.Module) -> set[int]:
    """Hidden-layer indices that feed an identity (projection-less) skip."""
    return {
        src - 1
        for src, _dst, proj_idx in getattr(model, "skip_connections", [])
        if proj_idx is None and src >= 1
    }


class NeuronUtilityTracker:
    """
    Contribution utility per hidden unit, accumulated as an EMA over observed
    batches:  u_j <- (1 - beta) * u_j + beta * mean|h_j| * ||W_out[:, j]||.

    Units with persistently low utility are "dormant": they neither fire nor
    influence downstream layers, and are the safest candidates for reinit.
    """

    def __init__(self, model: nn.Module, ema_beta: float = 0.05):
        self.model = model
        self.beta = ema_beta
        self.n_observed = 0
        widths = list(model.layer_sizes[1:])
        self._util: list[torch.Tensor] = [torch.zeros(w) for w in widths]

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """One forward pass; accumulate mean |post-activation| per unit."""
        acts: list[torch.Tensor] = []
        hooks = []

        def _hook(_m, _inp, out):
            acts.append(out.detach().abs().mean(dim=0).cpu())

        triples = _hidden_triples(self.model)
        for _lin, _bn, act in triples:
            hooks.append(act.register_forward_hook(_hook))
        try:
            self.model(x)
        finally:
            for h in hooks:
                h.remove()

        if len(acts) != len(self._util) or any(
            a.shape != u.shape for a, u in zip(acts, self._util)
        ):
            # Architecture changed under us (e.g. a mutation) — restart tracking.
            self._util = [torch.zeros_like(a) for a in acts]
            self.n_observed = 0

        for j, a in enumerate(acts):
            out_norm = torch.zeros_like(a)
            for lin, _main in _outgoing_linears(self.model, j):
                out_norm = out_norm + lin.weight.detach().norm(dim=0).cpu()
            u = a * out_norm
            self._util[j] = (1 - self.beta) * self._util[j] + self.beta * u
        self.n_observed += 1

    def utilities(self) -> list[torch.Tensor]:
        """Per-hidden-layer utility tensors (one value per unit)."""
        return [u.clone() for u in self._util]

    def dormant_mask(self, fraction: float = 0.1) -> list[torch.Tensor]:
        """Boolean mask per layer marking the lowest-utility `fraction` of units.

        At least one unit per layer is always left alive.
        """
        masks = []
        for u in self._util:
            w = u.numel()
            k = min(int(math.floor(w * fraction)), w - 1)
            mask = torch.zeros(w, dtype=torch.bool)
            if k > 0:
                idx = torch.argsort(u)[:k]
                mask[idx] = True
            masks.append(mask)
        return masks


@torch.no_grad()
def reinit_dormant(
    model: nn.Module,
    dormant_masks: list[torch.Tensor],
    seed: int | None = None,
) -> int:
    """
    Reinitialise the masked hidden units of a DynamicMLP in place.

    Per reborn unit: incoming Linear row gets a fresh Kaiming-uniform init
    (bias 0), its BatchNorm channel is reset (running stats and affine), and
    every outgoing weight column is zeroed — so the reborn unit contributes
    exactly nothing until training recruits it. The resulting model computes
    exactly what it would with the dormant unit ablated: no random noise is
    injected, and the only functional change is the removal of the dormant
    unit's old contribution — negligible by construction when units are chosen
    by low utility. Units feeding an identity (projection-less) skip cannot be
    silenced through a weight column; there the reborn unit's post-BN
    activation still reaches the skip target.

    Returns the number of units reinitialised.
    """
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    triples = _hidden_triples(model)
    if len(dormant_masks) != len(triples):
        raise ValueError(
            f"got {len(dormant_masks)} masks for {len(triples)} hidden layers"
        )

    n_reborn = 0
    for j, ((lin, bn, _act), mask) in enumerate(zip(triples, dormant_masks)):
        idx = torch.nonzero(mask, as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        # Fresh incoming weights: Kaiming uniform over the full layer's fan-in.
        fan_in = lin.in_features
        bound = math.sqrt(6.0 / fan_in)
        fresh = (torch.rand((idx.numel(), fan_in), generator=gen) * 2 - 1) * bound
        lin.weight[idx] = fresh.to(lin.weight.dtype).to(lin.weight.device)
        lin.bias[idx] = 0.0
        # Reset the BatchNorm channel so stale statistics don't distort the
        # reborn unit's first batches.
        if isinstance(bn, (nn.BatchNorm1d,)):
            bn.running_mean[idx] = 0.0
            bn.running_var[idx] = 1.0
            if bn.affine:
                bn.weight[idx] = 1.0
                bn.bias[idx] = 0.0
        # Zero every outgoing column: function preservation.
        for out_lin, _main in _outgoing_linears(model, j):
            out_lin.weight[:, idx] = 0.0
        n_reborn += int(idx.numel())
    return n_reborn


@torch.no_grad()
def effective_rank(model: nn.Module, x: torch.Tensor, layer: int = -1) -> float:
    """
    Entropy-based effective rank (Roy & Vetterli) of a hidden feature matrix.

    A healthy layer keeps its effective rank close to its width; a collapsing
    layer (rank -> 1) has lost representational diversity — an early symptom
    of plasticity loss.

    Args:
        model: DynamicMLP-style model.
        x:     input batch (B >= 2 recommended).
        layer: hidden layer index (-1 = last hidden layer).

    Returns:
        float in [1, width of the chosen layer].
    """
    feats: list[torch.Tensor] = []
    hooks = []

    def _hook(_m, _inp, out):
        feats.append(out.detach())

    triples = _hidden_triples(model)
    act = triples[layer][2]
    hooks.append(act.register_forward_hook(_hook))
    try:
        model(x)
    finally:
        for h in hooks:
            h.remove()

    H = feats[-1].float().cpu()
    H = H - H.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(H)
    s = s[s > 1e-9]
    if s.numel() == 0:
        return 1.0
    p = s / s.sum()
    entropy = float(-(p * torch.log(p)).sum())
    return float(np.exp(entropy))


class PlasticityController:
    """
    Orchestrates utility tracking + scheduled dormant-neuron reinit.

    observe(x) accumulates utilities (one cheap forward, no gradients).
    maybe_reinit() reinitialises the `reinit_fraction` lowest-utility units of
    every hidden layer once every `check_every` observes, and records the
    effective rank of the last hidden layer so collapse is visible over time.

    Args:
        model:           DynamicMLP-style model, modified in place on reinit.
        reinit_fraction: fraction of units per layer reborn per firing (0-1).
        check_every:     observes between firings.
        min_observed:    minimum observes before the first firing (utilities
                         need a few batches to stabilise).
    """

    def __init__(
        self,
        model: nn.Module,
        reinit_fraction: float = 0.05,
        check_every: int = 100,
        min_observed: int = 10,
        ema_beta: float = 0.05,
        seed: int | None = None,
    ):
        if not (0.0 <= reinit_fraction < 1.0):
            raise ValueError("reinit_fraction must be in [0, 1)")
        self.model = model
        self.reinit_fraction = reinit_fraction
        self.check_every = max(1, int(check_every))
        self.min_observed = min_observed
        self.seed = seed
        self.tracker = NeuronUtilityTracker(model, ema_beta=ema_beta)
        self.total_reborn = 0
        self.n_firings = 0
        self.rank_history: list[float] = []
        self._since_last = 0
        self._last_x: torch.Tensor | None = None

    def observe(self, x: torch.Tensor) -> None:
        was_training = self.model.training
        self.model.eval()
        try:
            self.tracker.update(x)
        finally:
            self.model.train(was_training)
        self._since_last += 1
        self._last_x = x.detach()

    def maybe_reinit(self, force: bool = False) -> int:
        """Reinit dormant units if the schedule says so. Returns units reborn."""
        due = force or (
            self._since_last >= self.check_every
            and self.tracker.n_observed >= self.min_observed
        )
        if not due or self.reinit_fraction <= 0.0:
            return 0
        masks = self.tracker.dormant_mask(self.reinit_fraction)
        n = reinit_dormant(self.model, masks, seed=self.seed)
        if self._last_x is not None and self._last_x.shape[0] >= 2:
            was_training = self.model.training
            self.model.eval()
            try:
                self.rank_history.append(effective_rank(self.model, self._last_x))
            finally:
                self.model.train(was_training)
        self.total_reborn += n
        self.n_firings += 1
        self._since_last = 0
        return n

    def summary(self) -> dict:
        return {
            "observed": self.tracker.n_observed,
            "firings": self.n_firings,
            "total_reborn": self.total_reborn,
            "rank_history": list(self.rank_history),
        }
