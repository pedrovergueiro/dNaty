"""
Meta-learned search controller (v2.2.0).

The episodic memory scores operators by *how much they helped so far* — a
context-free signal. The MetaController goes one step further: a per-operator
online linear model (LinUCB-style contextual bandit) sees the *state of the
search* (generation progress, best accuracy, size trend, gradient signal,
stagnation) and predicts the expected accuracy improvement of each mutation in
that state. Selection uses the upper confidence bound, so under-explored
operators keep getting tried.

The controller never replaces the episodic memory — its predictions are
*blended* with the memory's softmax, with a trust weight that grows as the
controller accumulates observations. Cold start therefore behaves exactly like
classic dNATY.

The learned policy is a small JSON-serialisable dict (`to_policy()` /
`from_policy()`), so — like the v2.1.0 operator priors — knowledge of *how to
search* transfers across runs and tasks:

    ev1 = DnatyEvolver(controller=True); ev1.run(...)
    policy = ev1.export_policy()
    ev2 = DnatyEvolver(controller=True, controller_policy=policy)  # warm

Pure numpy; incremental Sherman-Morrison updates, O(d^2) per observation.
"""
from __future__ import annotations

import json

import numpy as np

#: Length of the context vector produced by MetaController.make_context().
CONTEXT_DIM = 6

POLICY_FORMAT_VERSION = 1


class MetaController:
    """
    LinUCB contextual bandit over mutation operators.

    Per operator: ridge regression theta = A^-1 b predicting reward
    (child accuracy - parent accuracy) from the search context; selection
    score is theta^T x + alpha * sqrt(x^T A^-1 x) (optimism bonus).

    Args:
        alpha:  exploration strength (UCB bonus weight).
        warmup: observations at which controller trust reaches 0.5. Trust is
                n / (n + warmup); the blend with the memory softmax uses it.
        tau:    softmax temperature over UCB scores. Rewards are accuracy
                deltas (~1e-2), so tau defaults small.
        seed:   unused today, reserved for future stochastic policies.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        warmup: int = 24,
        tau: float = 0.02,
        seed: int | None = None,
    ):
        self.alpha = float(alpha)
        self.warmup = max(1, int(warmup))
        self.tau = float(tau)
        self.dim = CONTEXT_DIM
        self.n_obs = 0
        self._A_inv: dict[str, np.ndarray] = {}
        self._b: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # Context                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def make_context(
        gen: int,
        n_generations: int,
        best_acc: float,
        param_ratio: float,
        delta_grad: float,
        no_improve: int,
        patience: int = 8,
    ) -> np.ndarray:
        """Build the 6-d search-state context vector.

        Features (each bounded, so ridge stays well-conditioned):
          bias, search progress, current best accuracy, log size-trend vs the
          initial population (clipped to [-1, 1]), tanh of the last mean
          training-loss drop, stagnation fraction.
        """
        ratio = float(np.clip(np.log(max(param_ratio, 1e-8)), -1.0, 1.0))
        ctx = np.array(
            [
                1.0,
                min(gen / max(n_generations, 1), 1.0),
                float(np.clip(best_acc, 0.0, 1.0)),
                ratio,
                float(np.tanh(delta_grad)),
                min(no_improve / max(patience, 1), 1.0),
            ],
            dtype=np.float64,
        )
        # A NaN anywhere (e.g. a diverged training loss feeding delta_grad)
        # would poison the ridge matrices permanently — sanitize.
        return np.nan_to_num(ctx, nan=0.0, posinf=1.0, neginf=-1.0)

    # ------------------------------------------------------------------ #
    # Bandit                                                              #
    # ------------------------------------------------------------------ #
    def _ensure(self, op: str) -> None:
        if op not in self._A_inv:
            self._A_inv[op] = np.eye(self.dim)
            self._b[op] = np.zeros(self.dim)

    def ucb_scores(self, operators: list[str], context: np.ndarray) -> dict[str, float]:
        """Optimistic reward estimate per operator in this context."""
        x = np.asarray(context, dtype=np.float64)
        scores = {}
        for op in operators:
            self._ensure(op)
            A_inv = self._A_inv[op]
            theta = A_inv @ self._b[op]
            bonus = self.alpha * float(np.sqrt(max(x @ A_inv @ x, 0.0)))
            scores[op] = float(theta @ x) + bonus
        return scores

    def select_probs(self, operators: list[str], context: np.ndarray) -> dict[str, float]:
        """Softmax over UCB scores."""
        scores = self.ucb_scores(operators, context)
        vals = np.array([scores[op] for op in operators]) / max(self.tau, 1e-8)
        vals -= vals.max()
        e = np.exp(vals)
        p = e / e.sum()
        return {op: float(v) for op, v in zip(operators, p)}

    def update(self, op: str, context: np.ndarray, reward: float) -> None:
        """Observe (context, operator) -> reward. Sherman-Morrison, O(d^2).

        Non-finite observations are dropped — one NaN/inf would corrupt the
        per-operator ridge state for the rest of the run.
        """
        self._ensure(op)
        x = np.asarray(context, dtype=np.float64)
        if not (np.isfinite(reward) and np.all(np.isfinite(x))):
            return
        A_inv = self._A_inv[op]
        Ax = A_inv @ x
        denom = 1.0 + float(x @ Ax)
        self._A_inv[op] = A_inv - np.outer(Ax, Ax) / denom
        self._b[op] += float(reward) * x
        self.n_obs += 1

    @property
    def trust(self) -> float:
        """Blend weight in [0, 1): grows with observations, 0.5 at `warmup`."""
        return self.n_obs / (self.n_obs + self.warmup)

    def blend(
        self, memory_probs: dict[str, float], context: np.ndarray
    ) -> dict[str, float]:
        """Trust-weighted mix of controller probs and episodic-memory probs.

        Only operators present in memory_probs participate (the evolver's
        operator set is authoritative — MLP and CNN searches differ).
        """
        ops = list(memory_probs.keys())
        if not ops:
            return memory_probs
        w = self.trust
        if w <= 0.0:
            return dict(memory_probs)
        ctrl = self.select_probs(ops, context)
        mixed = {op: (1 - w) * memory_probs[op] + w * ctrl[op] for op in ops}
        total = sum(mixed.values())
        return {op: v / total for op, v in mixed.items()}

    # ------------------------------------------------------------------ #
    # Transferable policy                                                 #
    # ------------------------------------------------------------------ #
    def to_policy(self) -> dict:
        """Export the learned policy as a compact JSON-serialisable dict."""
        return {
            "format": "dnaty.controller_policy",
            "version": POLICY_FORMAT_VERSION,
            "dim": self.dim,
            "alpha": self.alpha,
            "warmup": self.warmup,
            "tau": self.tau,
            "n_obs": self.n_obs,
            "operators": {
                op: {
                    "A_inv": self._A_inv[op].tolist(),
                    "b": self._b[op].tolist(),
                }
                for op in self._A_inv
            },
        }

    @classmethod
    def from_policy(cls, policy: dict) -> "MetaController":
        """Rebuild a controller from `to_policy()` output (dict or JSON path)."""
        if isinstance(policy, (str, bytes)) or hasattr(policy, "__fspath__"):
            with open(policy, "r", encoding="utf-8") as f:
                policy = json.load(f)
        if policy.get("dim", CONTEXT_DIM) != CONTEXT_DIM:
            raise ValueError(
                f"policy context dim {policy.get('dim')} does not match "
                f"this build's CONTEXT_DIM={CONTEXT_DIM}"
            )
        ctrl = cls(
            alpha=policy.get("alpha", 0.3),
            warmup=policy.get("warmup", 24),
            tau=policy.get("tau", 0.02),
        )
        ctrl.n_obs = int(policy.get("n_obs", 0))
        for op, mats in policy.get("operators", {}).items():
            ctrl._A_inv[op] = np.array(mats["A_inv"], dtype=np.float64)
            ctrl._b[op] = np.array(mats["b"], dtype=np.float64)
        return ctrl

    def save_policy(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_policy(), f, indent=2, sort_keys=True)

    def __repr__(self) -> str:
        return (f"MetaController(ops={len(self._A_inv)}, n_obs={self.n_obs}, "
                f"trust={self.trust:.2f})")
