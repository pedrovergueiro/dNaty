"""
EpisodicMemory -- central component of dNATY.
Implements Eq. 1.4: score accumulation with temporal decay gamma.
Optimized: scores accumulated incrementally (O(1) per update).

v2.1.0 adds *transferable* operator priors: the accumulated per-operator
scores can be exported as a compact JSON-serialisable prior and used to
warm-start a later search on a related task (see `to_prior` / `seed_from_prior`
and the module-level `save_prior` / `load_prior`).
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np

# Bumped when the prior on-disk format changes in a backward-incompatible way.
PRIOR_FORMAT_VERSION = 1


@dataclass
class Experience:
    operator: str
    delta_loss: float
    gradient_norm: float
    generation: int
    weight: float = 1.0
    timestamp: int = 0

    @property
    def impact(self) -> float:
        """1[dL < 0] * |dL| * ||grad_L|| -- only experiences that improved."""
        if self.delta_loss >= 0:
            return 0.0
        return abs(self.delta_loss) * self.gradient_norm


class EpisodicMemory:
    """
    Episodic memory with temporal decay gamma.
    Scores accumulated incrementally -- O(1) per update, O(|ops|) per query.
    """

    def __init__(self, max_size: int = 500, decay_gamma: float = 0.99):
        self.experiences: list[Experience] = []
        self.max_size = max_size
        self.gamma = decay_gamma
        self._step = 0
        # Accumulated scores per operator -- incremental update
        self._scores: dict[str, float] = {}

    def update(self, exp: Experience) -> None:
        # Global decay of accumulated scores -- O(|ops|) not O(|mem|)
        for op in self._scores:
            self._scores[op] *= self.gamma

        imp = exp.impact
        if imp > 0:
            self._scores[exp.operator] = self._scores.get(exp.operator, 0.0) + imp

        exp.timestamp = self._step
        self._step += 1
        self.experiences.append(exp)

        if len(self.experiences) > self.max_size:
            self._prune()

    def _prune(self) -> None:
        # Remove oldest/least relevant experiences and recompute scores from scratch
        self.experiences.sort(
            key=lambda e: e.impact * (self.gamma ** max(0, self._step - e.timestamp)),
            reverse=True,
        )
        self.experiences = self.experiences[:self.max_size]
        self._scores = {}
        for e in self.experiences:
            if e.impact > 0:
                decay = self.gamma ** max(0, self._step - e.timestamp)
                self._scores[e.operator] = self._scores.get(e.operator, 0.0) + e.impact * decay

    def query_mutation_probs(self, operators: list[str], tau: float = 1.0) -> dict[str, float]:
        """Softmax over accumulated scores -- O(|ops|)."""
        if not operators:
            return {}
        vals = np.array(
            [self._scores.get(op, 0.0) for op in operators], dtype=np.float64
        ) / max(tau, 1e-8)
        vals -= vals.max()
        exp_vals = np.exp(vals)
        probs = exp_vals / exp_vals.sum()
        return {op: float(p) for op, p in zip(operators, probs)}

    def operator_counts(self, operators: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {op: 0 for op in operators}
        for e in self.experiences:
            if e.operator in counts and e.impact > 0:
                counts[e.operator] += 1
        return counts

    # ------------------------------------------------------------------ #
    # Transferable operator priors (v2.1.0)                               #
    # ------------------------------------------------------------------ #
    def to_prior(self) -> dict:
        """Export the accumulated operator scores as a transferable prior.

        The prior captures *which structural mutations helped* during this
        search — the knowledge worth carrying to the next, related task. It is
        a small JSON-serialisable dict (no model weights, no raw experiences),
        so it is safe to share and version-control.

        Returns:
            dict: {"version", "gamma", "n_experiences", "scores": {op: float}}.
        """
        return {
            "format": "dnaty.operator_prior",
            "version": PRIOR_FORMAT_VERSION,
            "gamma": self.gamma,
            "n_experiences": len(self.experiences),
            "scores": {op: float(s) for op, s in self._scores.items()},
        }

    def seed_from_prior(self, prior: dict, weight: float = 2.0) -> int:
        """Warm-start this memory's operator scores from an exported prior.

        The prior's raw magnitudes (which depend on the source run's gradient
        norms and length) are normalised away: scores are mean-centred and
        scaled to unit max-magnitude, then multiplied by `weight`. `weight`
        therefore acts as an inverse-temperature on the prior — how decisively
        early generations favour operators that worked on the previous task:

            weight = 0      no prior (equivalent to a cold start)
            weight = 1      gentle nudge (top operators ~2-3x baseline odds)
            weight = 2      default — informed but still exploratory
            weight >= 4     aggressive; risks premature convergence

        The seeded scores decay (via `gamma`) as task-specific experiences
        accumulate, so the prior fades and yields to real evidence over the run.

        Args:
            prior:  A dict produced by `to_prior()` (or the "scores" sub-dict).
            weight: Prior strength. Default 2.0.

        Returns:
            int: number of operators seeded (0 if the prior was empty/degenerate).
        """
        scores = prior.get("scores", prior) if isinstance(prior, dict) else {}
        if not scores or weight == 0:
            return 0
        ops = list(scores.keys())
        vals = np.array([float(scores[op]) for op in ops], dtype=np.float64)
        vals = vals - vals.mean()
        max_abs = np.abs(vals).max()
        if max_abs < 1e-12:
            return 0  # every operator scored identically — no usable signal
        vals = vals / max_abs * float(weight)
        for op, v in zip(ops, vals):
            self._scores[op] = float(v)
        return len(ops)

    def save_prior(self, path: str) -> None:
        """Write this memory's transferable prior to a JSON file."""
        save_prior(self.to_prior(), path)


def save_prior(prior: dict, path: str) -> None:
    """Persist an operator prior (from `EpisodicMemory.to_prior()`) to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prior, f, indent=2, sort_keys=True)


def load_prior(path: str) -> dict:
    """Load an operator prior previously written with `save_prior()`."""
    with open(path, "r", encoding="utf-8") as f:
        prior = json.load(f)
    if isinstance(prior, dict) and "scores" not in prior and all(
        isinstance(v, (int, float)) for v in prior.values()
    ):
        # Tolerate a bare {op: score} mapping.
        prior = {"scores": prior}
    return prior
