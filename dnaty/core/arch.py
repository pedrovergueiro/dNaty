"""
Architecture representation as a directed acyclic graph (DAG).
A_i = (V_i, E_i, phi_i, Omega_i)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
}

_innovation_counter = 0


def next_innovation() -> int:
    global _innovation_counter
    _innovation_counter += 1
    return _innovation_counter


class DynamicMLP(nn.Module):
    """
    MLP with mutable architecture. Represented as a list of linear layers.
    Supports the 8 dense mutation operators + skip connections.
    """

    def __init__(self, layer_sizes: list[int], activations: list[str] | None = None, n_classes: int = 10):
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        self.n_classes = n_classes
        self.activations = activations or ["relu"] * (len(layer_sizes) - 1)
        self.innovation_ids = [next_innovation() for _ in range(len(layer_sizes) - 1)]
        self._build()

    def _build(self) -> None:
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            # BatchNorm before activation -- stabilizes training, allows higher LR
            layers.append(nn.BatchNorm1d(self.layer_sizes[i + 1]))
            act = self.activations[i] if i < len(self.activations) else "relu"
            layers.append(ACTIVATIONS.get(act, nn.ReLU)())
        layers.append(nn.Linear(self.layer_sizes[-1], self.n_classes))
        self.net = nn.Sequential(*layers)
        # Skip connections: (src, dst, proj_idx). proj_idx=None means identity
        # residual; otherwise it indexes skip_projs. Projections live in an
        # nn.ModuleList so they are trained, moved by .to(device), counted by
        # count_params(), and persisted in state_dict().
        self.skip_connections: list[tuple[int, int, int | None]] = []
        self.skip_projs = nn.ModuleList()

    def add_skip_connection(self, src: int, dst: int, proj: "nn.Linear | None") -> None:
        if proj is None:
            self.skip_connections.append((src, dst, None))
        else:
            self.skip_projs.append(proj)
            self.skip_connections.append((src, dst, len(self.skip_projs) - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        if x.dtype != torch.float32:
            # pandas .values / numpy default to float64 -- casting here beats the
            # cryptic "mat1 and mat2 must have the same dtype" deep in nn.Linear
            x = x.float()
        layer_outputs = [x]
        idx = 0
        for i in range(len(self.layer_sizes) - 1):
            linear = self.net[idx]
            bn     = self.net[idx + 1]
            act    = self.net[idx + 2]
            out = act(bn(linear(layer_outputs[-1])))
            for src, dst, proj_idx in self.skip_connections:
                if dst == i + 1 and src < len(layer_outputs):
                    skip_in = layer_outputs[src]
                    if proj_idx is not None:
                        skip_in = self.skip_projs[proj_idx](skip_in)
                    if skip_in.shape == out.shape:
                        out = out + skip_in
            layer_outputs.append(out)
            idx += 3  # Linear + BN + Activation
        return self.net[idx](layer_outputs[-1])

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_flops(self) -> int:
        flops = 0
        for i in range(len(self.layer_sizes) - 1):
            flops += 2 * self.layer_sizes[i] * self.layer_sizes[i + 1]
        flops += 2 * self.layer_sizes[-1] * self.n_classes
        # skip connection projections (Linear layers not in layer_sizes)
        for _src, _dst, proj_idx in self.skip_connections:
            if proj_idx is not None:
                proj = self.skip_projs[proj_idx]
                flops += 2 * proj.in_features * proj.out_features
        return flops

    def is_valid(self) -> bool:
        return all(s > 0 for s in self.layer_sizes) and len(self.layer_sizes) >= 2
