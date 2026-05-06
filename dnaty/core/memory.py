"""
EpisodicMemory — componente central do dNaty.
Implementa eq. 1.4 da formalização: acumulação com decaimento temporal γ.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Experience:
    operator: str
    delta_loss: float       # ΔL — negativo = melhora
    gradient_norm: float    # ‖∇L‖₂
    generation: int
    weight: float = 1.0
    timestamp: int = 0

    @property
    def impact(self) -> float:
        """𝟙[ΔL < 0] · |ΔL| · ‖∇L‖ — só experiências que melhoraram (não-circular)."""
        if self.delta_loss >= 0:
            return 0.0
        return abs(self.delta_loss) * self.gradient_norm


class EpisodicMemory:
    """
    Memória episódica com decaimento temporal γ.
    Lema 1: após g gerações, P(op ∈ O_bom | 𝓜_g) ≥ P(op ∈ O_bom | 𝓜_0) + κ(g).
    """

    def __init__(self, max_size: int = 1000, decay_gamma: float = 0.99):
        self.experiences: list[Experience] = []
        self.max_size = max_size
        self.gamma = decay_gamma
        self._step = 0

    def update(self, exp: Experience) -> None:
        # Decaimento em todas as experiências existentes
        for e in self.experiences:
            e.weight *= self.gamma
        exp.timestamp = self._step
        self._step += 1
        self.experiences.append(exp)
        if len(self.experiences) > self.max_size:
            self._prune()

    def _prune(self) -> None:
        self.experiences.sort(
            key=lambda e: e.weight * (self.gamma ** max(0, self._step - e.timestamp)),
            reverse=True,
        )
        self.experiences = self.experiences[: self.max_size]

    def query_mutation_probs(self, operators: list[str], tau: float = 1.0) -> dict[str, float]:
        """Softmax sobre scores acumulados — eq. 1.4."""
        scores = {op: 0.0 for op in operators}
        for exp in self.experiences:
            if exp.operator in scores and exp.impact > 0:
                t_decay = self.gamma ** max(0, self._step - exp.timestamp)
                scores[exp.operator] += exp.impact * t_decay
        vals = np.array([scores[op] for op in operators], dtype=np.float64) / max(tau, 1e-8)
        vals -= vals.max()
        exp_vals = np.exp(vals)
        probs = exp_vals / exp_vals.sum()
        return {op: float(p) for op, p in zip(operators, probs)}

    def operator_counts(self, operators: list[str]) -> dict[str, int]:
        counts = {op: 0 for op in operators}
        for e in self.experiences:
            if e.operator in counts and e.impact > 0:
                counts[e.operator] += 1
        return counts
