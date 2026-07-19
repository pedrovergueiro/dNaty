"""
dNATY individual: M_i = (theta_i, A_i, M_i)
"""
from __future__ import annotations
from copy import deepcopy
import torch
from dnaty.core.arch import DynamicMLP
from dnaty.core.memory import EpisodicMemory


class Individual:
    def __init__(self, model: DynamicMLP, memory: EpisodicMemory | None = None):
        self.model = model
        self.memory = memory or EpisodicMemory()
        self.last_op: str = "init"
        self.fitness: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (acc, -cost, -sharp)
        self.acc: float = 0.0
        self.last_grad_norm: float = 0.0
        self.last_delta_loss: float = 0.0
        # Accuracy of the parent at mutation time. Set by _mutate_population so
        # memory/proxy updates credit the right parent even when the mutant list
        # is reordered (proxy_filter oversampling). None = unknown.
        self.parent_acc: float | None = None

    def clone(self) -> "Individual":
        new_model = deepcopy(self.model)
        new_mem = deepcopy(self.memory)
        ind = Individual(new_model, new_mem)
        ind.last_op = self.last_op
        ind.fitness = self.fitness
        ind.acc = self.acc
        return ind

    def count_params(self) -> int:
        return self.model.count_params()

    def count_flops(self) -> int:
        return self.model.count_flops()
