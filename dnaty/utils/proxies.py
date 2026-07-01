"""
Zero-cost proxy ensemble for fast candidate pre-filtering.

Three complementary proxies, each computable at init (no training needed):
  expressivity  -- diversity of ReLU activation patterns across a batch (NASWOT-style)
  trainability  -- gradient signal strength at init (SynFlow / GradNorm proxy)
  efficiency    -- inverse structural complexity (rewards smaller candidates)

Proxy weights are updated adaptively via Spearman correlation with actual fitness
deltas, so the ensemble gets smarter over generations.

Usage:
    from dnaty.utils.proxies import score_candidate, ProxyEnsemble

    score = score_candidate(model, input_shape=(784,))
    # score["combined"] -> float (higher = better candidate)

    ensemble = ProxyEnsemble()
    scores = [ensemble.score(ind.model, input_shape=(784,)) for ind in candidates]
    ensemble.update(scores, actual_fitness_deltas)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Individual proxies
# ---------------------------------------------------------------------------

def _proxy_expressivity(model: nn.Module, dummy: torch.Tensor) -> float:
    """
    NASWOT-style expressivity: mean pairwise Hamming distance of ReLU patterns.

    Models with more diverse activation patterns across the batch have higher
    expressivity and tend to generalise better after training.
    Returns a value in [0, 1]; higher = more expressive.
    """
    acts: list[torch.Tensor] = []
    hooks = []

    def _hook(m: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        acts.append((out.detach() > 0).float().view(out.shape[0], -1))

    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU)):
            hooks.append(m.register_forward_hook(_hook))

    try:
        with torch.no_grad():
            model(dummy)
    finally:
        for h in hooks:
            h.remove()

    if not acts:
        return 0.5

    patterns = torch.cat(acts, dim=1)  # (B, total_units)
    B = patterns.shape[0]
    if B < 2:
        return 0.5

    diff = (patterns.unsqueeze(0) != patterns.unsqueeze(1)).float()
    hamming = diff.mean(-1)  # (B, B)
    mask = 1 - torch.eye(B, device=patterns.device)
    score = (hamming * mask).sum() / max(mask.sum().item(), 1.0)
    return float(score.clamp(0, 1).item())


def _proxy_trainability(model: nn.Module, dummy: torch.Tensor) -> float:
    """
    SynFlow-style trainability: sum of |grad * param| at init using all-ones input.

    Uses an all-ones input (not random) to avoid layer-collapse bias where
    some inputs cancel gradients. Higher score → more gradient flow → more trainable.
    Returns a positive float (normalised by param count).
    """
    ones = torch.ones_like(dummy)
    params = list(model.parameters())
    if not params:
        return 0.0

    orig_req_grads = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(True)

    try:
        out = model(ones)
        loss = out.sum()
        grads = torch.autograd.grad(
            loss, params, allow_unused=True, create_graph=False
        )
        synflow = sum(
            (g * p).abs().sum().item()
            for g, p in zip(grads, params)
            if g is not None
        )
        n_params = sum(p.numel() for p in params)
        return float(synflow / max(n_params, 1))
    except Exception:
        return 0.0
    finally:
        for p, rg in zip(params, orig_req_grads):
            p.requires_grad_(rg)


def _proxy_efficiency(model: nn.Module) -> float:
    """
    Structural efficiency: inversely proportional to parameter count.
    Rewards smaller, more compressed candidates. Returns value in (0, 1].
    """
    n_params = sum(p.numel() for p in model.parameters())
    return 1.0 / (1.0 + float(np.log1p(n_params) / np.log(1e7)))


# ---------------------------------------------------------------------------
# Ensemble scorer
# ---------------------------------------------------------------------------

def score_candidate(
    model: nn.Module,
    input_shape: tuple,
    batch_size: int = 32,
    weights: dict[str, float] | None = None,
) -> dict:
    """
    Compute weighted zero-cost proxy score for a candidate model.

    Args:
        model:       Candidate nn.Module to score (at init, before training).
        input_shape: Input shape WITHOUT batch dim, e.g. (784,) or (3, 32, 32).
        batch_size:  Batch size for activation-based proxies.
        weights:     Per-proxy weights {"expressivity", "trainability", "efficiency"}.
                     Defaults to equal weights (1/3 each).

    Returns:
        Dict with keys: expressivity, trainability, efficiency, combined, weights.
    """
    if weights is None:
        weights = {"expressivity": 1 / 3, "trainability": 1 / 3, "efficiency": 1 / 3}

    dummy = torch.randn(batch_size, *input_shape)
    was_training = model.training
    model.eval()

    expr  = _proxy_expressivity(model, dummy)
    train = _proxy_trainability(model, dummy.clone())
    eff   = _proxy_efficiency(model)

    if was_training:
        model.train()

    # Normalise trainability via log-sigmoid so it sits in (0, 1)
    def _log_sig(x: float, scale: float = 1.0) -> float:
        return float(1.0 / (1.0 + np.exp(-np.log1p(x) * scale)))

    scores = {
        "expressivity": expr,
        "trainability": _log_sig(train, scale=2.0),
        "efficiency":   eff,
    }
    combined = sum(weights.get(k, 0.0) * v for k, v in scores.items())
    scores["combined"] = combined
    scores["weights"]  = dict(weights)
    return scores


# ---------------------------------------------------------------------------
# Adaptive ensemble with proxy weight tracking
# ---------------------------------------------------------------------------

class ProxyEnsemble:
    """
    Adaptive proxy weight tracker.

    Tracks which zero-cost proxy best predicts actual fitness deltas and
    updates proxy weights via Spearman rank correlation (exponential moving
    average). Proxies that consistently predict good candidates gain more weight.

    Usage:
        ensemble = ProxyEnsemble()

        # Before training candidates:
        score_dicts = [ensemble.score(ind.model, (784,)) for ind in candidates]

        # Filter: keep top-N by combined score
        ranked = sorted(zip(candidates, score_dicts), key=lambda x: -x[1]["combined"])
        filtered = [ind for ind, _ in ranked[:n_keep]]

        # After evaluating fitness:
        ensemble.update(score_dicts, actual_fitness_deltas)
    """

    _PROXY_NAMES = ("expressivity", "trainability", "efficiency")

    def __init__(self, ema_alpha: float = 0.15):
        self._weights: dict[str, float] = {k: 1 / 3 for k in self._PROXY_NAMES}
        self._alpha = ema_alpha
        self._n_updates = 0

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def score(
        self,
        model: nn.Module,
        input_shape: tuple,
        batch_size: int = 16,
    ) -> dict:
        """Score a single candidate with current proxy weights."""
        return score_candidate(
            model, input_shape, batch_size=batch_size, weights=self._weights
        )

    def update(
        self,
        score_dicts: list[dict],
        actual_deltas: list[float],
    ) -> None:
        """
        Update proxy weights based on correlation with actual fitness deltas.

        Args:
            score_dicts:    List of dicts returned by score() for each candidate.
            actual_deltas:  List of fitness delta (post - pre) for each candidate.
        """
        if len(score_dicts) < 4:
            return

        from scipy.stats import spearmanr
        import warnings

        new_weights: dict[str, float] = {}
        for name in self._PROXY_NAMES:
            pred = [s.get(name, 0.0) for s in score_dicts]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    corr, _ = spearmanr(pred, actual_deltas)
                corr = 0.0 if (corr != corr) else max(0.0, float(corr))  # NaN guard
            except Exception:
                corr = 0.0
            new_weights[name] = corr

        total = sum(new_weights.values())
        if total > 1e-8:
            new_weights = {k: v / total for k, v in new_weights.items()}
        else:
            new_weights = {k: 1 / 3 for k in self._PROXY_NAMES}

        # EMA update — blend towards new weights
        for k in self._PROXY_NAMES:
            self._weights[k] = (
                (1 - self._alpha) * self._weights[k]
                + self._alpha * new_weights[k]
            )

        # Re-normalise
        s = sum(self._weights.values())
        self._weights = {k: v / s for k, v in self._weights.items()}
        self._n_updates += 1

    def __repr__(self) -> str:
        w = ", ".join(f"{k}={v:.2f}" for k, v in self._weights.items())
        return f"ProxyEnsemble(updates={self._n_updates}, weights=[{w}])"
