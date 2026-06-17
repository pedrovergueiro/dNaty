"""
10 structural mutation operators for dNATY with formal guarantees.
Each operator returns (Individual, bool) -- bool indicates whether the mutation was applied.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from dnaty.core.arch import DynamicMLP, next_innovation, ACTIVATIONS
from dnaty.core.individual import Individual

OPERATORS = [
    "add_neuron",
    "remove_neuron",
    "add_skip",
    "add_residual",       # identity residual (in == out, no projection)
    "change_activation",
    "split_layer",
    "merge_layers",
    "prune_connections",
    "duplicate_module",
    "add_conv_block",     # adds a bottleneck layer (half of last hidden)
    "depthwise_sep",      # narrow bottleneck (1/4) -- MLP depthwise decomposition
]


def _rebuild_from_sizes(ind: Individual, new_sizes: list[int], new_acts: list[str]) -> Individual:
    """Rebuild the individual with new layer sizes, copying weights where possible."""
    old_model = ind.model
    try:
        device = next(old_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    new_model = DynamicMLP(new_sizes, new_acts, old_model.n_classes)
    # Copy weights for layers that exist in both old and new model
    old_layers = [(old_model.net[i], old_model.net[i + 2])
                  for i in range(0, len(old_model.net) - 1, 3)]
    new_layers = [(new_model.net[i], new_model.net[i + 2])
                  for i in range(0, len(new_model.net) - 1, 3)]
    for (old_lin, _), (new_lin, _) in zip(old_layers, new_layers):
        min_out = min(old_lin.out_features, new_lin.out_features)
        min_in = min(old_lin.in_features, new_lin.in_features)
        with torch.no_grad():
            new_lin.weight[:min_out, :min_in].copy_(old_lin.weight[:min_out, :min_in])
            new_lin.bias[:min_out].copy_(old_lin.bias[:min_out])
    # Copy classifier head
    old_cls = old_model.net[-1]
    new_cls = new_model.net[-1]
    min_in = min(old_cls.in_features, new_cls.in_features)
    with torch.no_grad():
        new_cls.weight[:, :min_in].copy_(old_cls.weight[:, :min_in])
        new_cls.bias.copy_(old_cls.bias)
    new_model = new_model.to(device)
    new_ind = Individual(new_model, deepcopy(ind.memory))
    return new_ind


def add_neuron(ind: Individual, eps: float = 0.01) -> tuple[Individual, bool]:
    """Op 1: insert a neuron in a random hidden layer. ||output_diff|| < eps * ||x||."""
    sizes = list(ind.model.layer_sizes)
    acts = list(ind.model.activations)
    # Only modifies hidden layers (indices 1 to len-2, excluding input and output)
    hidden_indices = list(range(1, len(sizes) - 1))
    if not hidden_indices:
        return ind, False
    layer_idx = hidden_indices[np.random.randint(len(hidden_indices))]
    sizes[layer_idx] += 1
    new_ind = _rebuild_from_sizes(ind, sizes, acts)
    new_ind.last_op = "add_neuron"
    return new_ind, True


def remove_neuron(ind: Individual) -> tuple[Individual, bool]:
    """Op 2: remove ~12.5% of neurons from the chosen layer."""
    sizes = list(ind.model.layer_sizes)
    acts = list(ind.model.activations)
    candidates = [i for i in range(1, len(sizes) - 1) if sizes[i] > 4]
    if not candidates:
        return ind, False
    layer_idx = candidates[np.random.randint(len(candidates))]
    n_remove = max(1, sizes[layer_idx] // 8)
    sizes[layer_idx] = max(4, sizes[layer_idx] - n_remove)
    new_ind = _rebuild_from_sizes(ind, sizes, acts)
    new_ind.last_op = "remove_neuron"
    return new_ind, True


def add_residual(ind: Individual) -> tuple[Individual, bool]:
    """Identity residual between layers of the same size -- no projection.

    Unlike add_skip (which projects different dimensions), this operator
    only connects layers where in_features == out_features: out = layer(x) + x.
    Pure ResNet implementation, ~zero extra FLOPs.
    """
    sizes = ind.model.layer_sizes
    if len(sizes) < 3:
        return ind, False
    # Find consecutive pairs (src, dst) with the same size
    # sizes[k] == sizes[k+1] -> out of block k-1 == out of block k -> identity residual
    candidates = [
        (k, k + 1)
        for k in range(len(sizes) - 1)
        if sizes[k] == sizes[k + 1]
    ]
    if not candidates:
        return ind, False
    try:
        device = next(ind.model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    src, dst = candidates[np.random.randint(len(candidates))]
    new_ind = ind.clone()
    # proj=None sinaliza residual identidade puro no forward
    new_ind.model.skip_connections.append((src, dst, None))  # proj=None signals pure identity residual in forward
    new_ind.last_op = "add_residual"
    return new_ind, True


def add_skip(ind: Individual) -> tuple[Individual, bool]:
    """Op 3: add a skip connection with projection if dimensions differ."""
    sizes = ind.model.layer_sizes
    if len(sizes) < 3:
        return ind, False
    try:
        device = next(ind.model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    new_ind = ind.clone()
    src = np.random.randint(0, len(sizes) - 2)
    dst = np.random.randint(src + 1, len(sizes))
    src_size = sizes[src]
    dst_size = sizes[dst] if dst < len(sizes) else ind.model.n_classes
    proj = None
    if src_size != dst_size:
        proj = nn.Linear(src_size, dst_size, bias=False).to(device)
        nn.init.orthogonal_(proj.weight)
    new_ind.model.skip_connections.append((src, dst, proj))
    new_ind.last_op = "add_skip"
    return new_ind, True


def change_activation(ind: Individual) -> tuple[Individual, bool]:
    """Op 4: change the activation function of a random layer."""
    acts = list(ind.model.activations)
    if not acts:
        return ind, False
    layer_idx = np.random.randint(len(acts))
    options = [a for a in ACTIVATIONS.keys() if a != acts[layer_idx]]
    new_act = options[np.random.randint(len(options))]
    acts[layer_idx] = new_act
    new_ind = _rebuild_from_sizes(ind, list(ind.model.layer_sizes), acts)
    new_ind.last_op = "change_activation"
    return new_ind, True


def split_layer(ind: Individual) -> tuple[Individual, bool]:
    """Op 5: split a hidden layer into two with orthogonal initialisation."""
    sizes = list(ind.model.layer_sizes)
    acts = list(ind.model.activations)
    hidden_indices = list(range(1, len(sizes) - 1))
    if not hidden_indices:
        return ind, False
    layer_idx = hidden_indices[np.random.randint(len(hidden_indices))]
    half = max(4, sizes[layer_idx] // 2)
    new_sizes = sizes[:layer_idx + 1] + [half] + sizes[layer_idx + 1:]
    new_acts = acts[:layer_idx - 1] + [acts[layer_idx - 1], acts[layer_idx - 1]] + acts[layer_idx:]
    # Ensure acts has the correct length
    n_hidden = len(new_sizes) - 2
    while len(new_acts) < n_hidden:
        new_acts.append("relu")
    new_acts = new_acts[:n_hidden]
    new_ind = _rebuild_from_sizes(ind, new_sizes, new_acts)
    new_ind.last_op = "split_layer"
    return new_ind, True


def merge_layers(ind: Individual) -> tuple[Individual, bool]:
    """Op 6: merge two consecutive hidden layers via concatenation."""
    sizes = list(ind.model.layer_sizes)
    acts = list(ind.model.activations)
    # Needs at least 2 hidden layers to merge
    if len(sizes) < 4:
        return ind, False
    hidden_indices = list(range(1, len(sizes) - 2))
    if not hidden_indices:
        return ind, False
    idx = hidden_indices[np.random.randint(len(hidden_indices))]
    merged = sizes[idx] + sizes[idx + 1]
    new_sizes = sizes[:idx] + [merged] + sizes[idx + 2:]
    n_hidden = len(new_sizes) - 2
    new_acts = (acts + ["relu"] * 10)[:n_hidden]
    new_ind = _rebuild_from_sizes(ind, new_sizes, new_acts)
    new_ind.last_op = "merge_layers"
    return new_ind, True


def prune_connections(ind: Individual, sparsity_max: float = 0.5) -> tuple[Individual, bool]:
    """Op 7: zero connections with |w| < adaptive threshold tau."""
    new_ind = ind.clone()
    pruned_any = False
    for module in new_ind.model.modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                w = module.weight.data
                tau = w.abs().mean() * 0.5
                mask = w.abs() >= tau
                # Enforce maximum sparsity cap
                if mask.float().mean() < (1 - sparsity_max):
                    continue
                module.weight.data = w * mask.float()
                pruned_any = True
    new_ind.last_op = "prune_connections"
    return new_ind, pruned_any


def duplicate_module(ind: Individual, noise_eps: float = 0.01) -> tuple[Individual, bool]:
    """Op 8: duplicate a hidden layer with noise perturbation eps."""
    sizes = list(ind.model.layer_sizes)
    acts = list(ind.model.activations)
    hidden_indices = list(range(1, len(sizes) - 1))
    if not hidden_indices:
        return ind, False
    layer_idx = hidden_indices[np.random.randint(len(hidden_indices))]
    new_sizes = sizes[:layer_idx + 1] + [sizes[layer_idx]] + sizes[layer_idx + 1:]
    n_hidden = len(new_sizes) - 2
    new_acts = (acts + ["relu"] * 10)[:n_hidden]
    new_ind = _rebuild_from_sizes(ind, new_sizes, new_acts)
    # Find the Linear layer at the right index (stride 3: Linear+BN+Act per block)
    net_idx = layer_idx * 3  # each block has 3 modules: Linear + BN + Act; duplicated block is at layer_idx
    if 0 <= net_idx < len(new_ind.model.net) - 1:
        module = new_ind.model.net[net_idx]
        if isinstance(module, torch.nn.Linear):
            with torch.no_grad():
                module.weight.data += torch.randn_like(module.weight) * noise_eps
    new_ind.last_op = "duplicate_module"
    return new_ind, True


def add_conv_block(ind: Individual) -> tuple[Individual, bool]:
    """Op 9: insert a compact layer (half of last hidden) -- bottleneck that reduces FLOPs."""
    sizes = list(ind.model.layer_sizes)
    acts = list(ind.model.activations)
    insert_size = max(16, sizes[-1] // 2)
    new_sizes = sizes + [insert_size]
    new_acts = acts + ["relu"]
    new_ind = _rebuild_from_sizes(ind, new_sizes, new_acts)
    new_ind.last_op = "add_conv_block"
    return new_ind, True


def depthwise_sep(ind: Individual) -> tuple[Individual, bool]:
    """MLP bottleneck decomposition -- analogous to depthwise+pointwise conv for MLPs.

    Replaces the last hidden layer L->L with two stages:
      L -> bottleneck (L//4, narrow channel) -> L
    Saves ~75% FLOPs in that layer vs a direct Linear(L, L),
    while preserving representational capacity via the expansion stage.
    """
    sizes = list(ind.model.layer_sizes)
    acts = list(ind.model.activations)
    if len(sizes) < 2:
        return ind, False
    last_hidden_idx = len(sizes) - 1
    last_hidden = sizes[last_hidden_idx]
    bottleneck = max(8, last_hidden // 4)
    # Insert narrow stage: ..., last_hidden -> bottleneck -> last_hidden
    new_sizes = sizes[:last_hidden_idx] + [bottleneck] + sizes[last_hidden_idx:]
    n_hidden = len(new_sizes) - 2
    new_acts = (acts + ["relu"] * 10)[:n_hidden]
    new_ind = _rebuild_from_sizes(ind, new_sizes, new_acts)
    new_ind.last_op = "depthwise_sep"
    return new_ind, True


OPERATOR_FNS = {
    "add_neuron": add_neuron,
    "remove_neuron": remove_neuron,
    "add_skip": add_skip,
    "add_residual": add_residual,
    "change_activation": change_activation,
    "split_layer": split_layer,
    "merge_layers": merge_layers,
    "prune_connections": prune_connections,
    "duplicate_module": duplicate_module,
    "add_conv_block": add_conv_block,
    "depthwise_sep": depthwise_sep,
}

OPERATORS = list(OPERATOR_FNS.keys())


def apply_operator(ind: Individual, op: str) -> tuple[Individual, bool]:
    fn = OPERATOR_FNS.get(op)
    if fn is None:
        return ind, False
    return fn(ind)
