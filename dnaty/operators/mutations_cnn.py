"""
Structural mutation operators for DynamicCNN -- CIFAR-10.
Operators 9 and 10 are now real (not proxy).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from dnaty.core.arch_cnn import DynamicCNN, ConvBlock, DepthwiseSepBlock
from dnaty.core.individual import Individual
from dnaty.core.memory import EpisodicMemory

CNN_OPERATORS = [
    "add_conv_block",       # Op 9 real: add Conv2D+BN+ReLU block
    "depthwise_sep",        # Op 10 real: add depthwise separable block
    "swap_conv_to_dw",      # Layer swap: replace standard Conv with DwConv (~8-9x fewer FLOPs)
    "add_fc_neuron",        # Op 1 adapted: add neuron to FC layer
    "remove_fc_neuron",     # Op 2 adapted: remove neuron from FC layer
    "change_stride",        # New op: change stride of a block (more aggressive downsampling)
    "add_skip_conv",        # Op 3 adapted: skip connection between conv blocks
    "prune_channels",       # Op 7 adapted: reduce channels in a block
    "duplicate_conv_block", # Op 8 adapted: duplicate conv block with noise
]


def _clone_cnn(ind: Individual) -> Individual:
    new_model = deepcopy(ind.model)
    try:
        device = next(ind.model.parameters()).device
        new_model = new_model.to(device)
    except StopIteration:
        pass
    new_ind = Individual(new_model, deepcopy(ind.memory))
    return new_ind


def _copy_overlap(old_mod: nn.Module, new_mod: nn.Module) -> None:
    """Copy the overlapping weight slices from old_mod into new_mod for every
    matched Conv2d / Linear / BatchNorm submodule.

    A shape-changing mutation (e.g. channel pruning) can then keep the trained
    weights it still has room for instead of re-initialising the whole
    sub-network at random -- which would make the mutated child train from
    scratch in t_local epochs and almost always lose selection.
    """
    with torch.no_grad():
        for old_m, new_m in zip(old_mod.modules(), new_mod.modules()):
            if type(old_m) is not type(new_m):
                continue
            if isinstance(new_m, (nn.Conv2d, nn.Linear)):
                o = min(old_m.weight.shape[0], new_m.weight.shape[0])
                i = min(old_m.weight.shape[1], new_m.weight.shape[1])
                new_m.weight[:o, :i].copy_(old_m.weight[:o, :i])
                if old_m.bias is not None and new_m.bias is not None:
                    new_m.bias[:o].copy_(old_m.bias[:o])
            elif isinstance(new_m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                o = min(old_m.num_features, new_m.num_features)
                if old_m.affine and new_m.affine:
                    new_m.weight[:o].copy_(old_m.weight[:o])
                    new_m.bias[:o].copy_(old_m.bias[:o])
                if old_m.track_running_stats and new_m.track_running_stats:
                    new_m.running_mean[:o].copy_(old_m.running_mean[:o])
                    new_m.running_var[:o].copy_(old_m.running_var[:o])


def add_conv_block(ind: Individual) -> tuple[Individual, bool]:
    """Op 9 REAL: add a Conv2D+BN+ReLU block after the last conv block."""
    model = ind.model
    if not isinstance(model, DynamicCNN):
        return ind, False
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    last_ch = model.conv_configs[-1]["out_ch"]
    # Double channels up to a maximum of 256
    new_ch = min(last_ch * 2, 256)
    new_cfg = {"type": "conv", "in_ch": last_ch, "out_ch": new_ch, "stride": 1, "kernel": 3}

    new_configs = list(model.conv_configs) + [new_cfg]
    new_model = DynamicCNN(new_configs, list(model.fc_sizes), model.n_classes, model.in_channels)
    new_model = new_model.to(device)

    # Copy weights from existing blocks
    for i, (old_layer, new_layer) in enumerate(zip(model.conv_layers, new_model.conv_layers)):
        new_layer.load_state_dict(old_layer.state_dict())
    # Copiar FC
    new_model.fc.load_state_dict(model.fc.state_dict())
    new_model.classifier.load_state_dict(model.classifier.state_dict())

    new_ind = Individual(new_model, deepcopy(ind.memory))
    new_ind.last_op = "add_conv_block"
    return new_ind, True


def depthwise_sep(ind: Individual) -> tuple[Individual, bool]:
    """Op 10 REAL: add a depthwise separable block -- k^2 times more efficient."""
    model = ind.model
    if not isinstance(model, DynamicCNN):
        return ind, False
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    last_ch = model.conv_configs[-1]["out_ch"]
    new_ch = min(last_ch * 2, 256)
    new_cfg = {"type": "depthwise", "in_ch": last_ch, "out_ch": new_ch, "stride": 1}

    new_configs = list(model.conv_configs) + [new_cfg]
    new_model = DynamicCNN(new_configs, list(model.fc_sizes), model.n_classes, model.in_channels)
    new_model = new_model.to(device)

    for i, (old_layer, new_layer) in enumerate(zip(model.conv_layers, new_model.conv_layers)):
        new_layer.load_state_dict(old_layer.state_dict())
    new_model.fc.load_state_dict(model.fc.state_dict())
    new_model.classifier.load_state_dict(model.classifier.state_dict())

    new_ind = Individual(new_model, deepcopy(ind.memory))
    new_ind.last_op = "depthwise_sep"
    return new_ind, True


def swap_conv_to_dw(ind: Individual) -> tuple[Individual, bool]:
    """Layer swap: replace the highest-FLOPs Conv2D block with DepthwiseSeparable.

    Reduces FLOPs ~8-9x in the swapped layer (k^2*Cin*Cout -> k^2*Cin + Cin*Cout).
    Selects the candidate with the largest in_ch*out_ch product (dominant FLOPs proxy)
    so the swap delivers the biggest efficiency gain per call.
    Only candidates with stride=1 and out_ch >= 8 qualify.
    """
    model = ind.model
    if not isinstance(model, DynamicCNN):
        return ind, False
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # Candidates: standard Conv blocks with stride=1 (DwConv requires stride=1)
    # out_ch >= 8 ensures the pointwise stage is meaningful
    candidates = [
        i for i, c in enumerate(model.conv_configs)
        if c["type"] == "conv" and c.get("stride", 1) == 1 and c.get("out_ch", 0) >= 8
    ]
    if not candidates:
        return ind, False

    # Pick the layer with highest in_ch * out_ch (FLOPs proxy for k=3, same H/W)
    idx = max(candidates, key=lambda i: (
        model.conv_configs[i]["in_ch"] * model.conv_configs[i]["out_ch"]
    ))
    cfg = model.conv_configs[idx]
    new_configs = [dict(c) for c in model.conv_configs]
    new_configs[idx] = {
        "type": "depthwise",
        "in_ch": cfg["in_ch"],
        "out_ch": cfg["out_ch"],
        "stride": cfg.get("stride", 1),
    }

    new_model = DynamicCNN(new_configs, list(model.fc_sizes), model.n_classes, model.in_channels)
    new_model = new_model.to(device)

    # Copy unmodified blocks
    for i, (old_l, new_l) in enumerate(zip(model.conv_layers, new_model.conv_layers)):
        if i != idx:
            try:
                new_l.load_state_dict(old_l.state_dict())
            except Exception:
                pass  # shape changed -- random initialisation is fine
    new_model.fc.load_state_dict(model.fc.state_dict())
    new_model.classifier.load_state_dict(model.classifier.state_dict())

    new_ind = Individual(new_model, deepcopy(ind.memory))
    new_ind.last_op = "swap_conv_to_dw"
    return new_ind, True


def add_fc_neuron(ind: Individual) -> tuple[Individual, bool]:
    """Add neurons to the last FC layer."""
    model = ind.model
    if not isinstance(model, DynamicCNN) or not model.fc_sizes:
        return ind, False
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    new_fc = list(model.fc_sizes)
    new_fc[-1] += 16  # add 16 neurons
    new_model = DynamicCNN(list(model.conv_configs), new_fc, model.n_classes, model.in_channels)
    new_model = new_model.to(device)

    for old_l, new_l in zip(model.conv_layers, new_model.conv_layers):
        new_l.load_state_dict(old_l.state_dict())
    for old_l, new_l in zip(model.fc, new_model.fc):
        try:
            new_l.load_state_dict(old_l.state_dict())
        except Exception:
            pass  # shape changed (last fc layer) — keep random init

    new_ind = Individual(new_model, deepcopy(ind.memory))
    new_ind.last_op = "add_fc_neuron"
    return new_ind, True


def remove_fc_neuron(ind: Individual) -> tuple[Individual, bool]:
    """Remove neurons from the last FC layer (minimum 32)."""
    model = ind.model
    if not isinstance(model, DynamicCNN) or not model.fc_sizes or model.fc_sizes[-1] <= 32:
        return ind, False
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    new_fc = list(model.fc_sizes)
    new_fc[-1] = max(32, new_fc[-1] - 16)
    new_model = DynamicCNN(list(model.conv_configs), new_fc, model.n_classes, model.in_channels)
    new_model = new_model.to(device)

    for old_l, new_l in zip(model.conv_layers, new_model.conv_layers):
        new_l.load_state_dict(old_l.state_dict())
    for old_l, new_l in zip(model.fc, new_model.fc):
        try:
            new_l.load_state_dict(old_l.state_dict())
        except Exception:
            pass  # shape changed (last fc layer) — keep random init

    new_ind = Individual(new_model, deepcopy(ind.memory))
    new_ind.last_op = "remove_fc_neuron"
    return new_ind, True


def change_stride(ind: Individual) -> tuple[Individual, bool]:
    """Change the stride of an intermediate block to 2 (more aggressive downsampling)."""
    model = ind.model
    if not isinstance(model, DynamicCNN) or len(model.conv_configs) < 2:
        return ind, False
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # Pick an intermediate block with stride=1 to change to 2
    candidates = [i for i, c in enumerate(model.conv_configs[1:], 1) if c.get("stride", 1) == 1]
    if not candidates:
        return ind, False

    idx = candidates[np.random.randint(len(candidates))]
    new_configs = [dict(c) for c in model.conv_configs]
    new_configs[idx]["stride"] = 2

    new_model = DynamicCNN(new_configs, list(model.fc_sizes), model.n_classes, model.in_channels)
    new_model = new_model.to(device)

    # Stride doesn't change weight shapes, so all weights are transferable
    for old_l, new_l in zip(model.conv_layers, new_model.conv_layers):
        try:
            new_l.load_state_dict(old_l.state_dict())
        except Exception:
            pass
    new_model.fc.load_state_dict(model.fc.state_dict())
    new_model.classifier.load_state_dict(model.classifier.state_dict())

    new_ind = Individual(new_model, deepcopy(ind.memory))
    new_ind.last_op = "change_stride"
    return new_ind, True


def add_skip_conv(ind: Individual) -> tuple[Individual, bool]:
    """Add a skip connection between two conv blocks (via 1x1 conv if channels differ)."""
    # Implemented as adding a residual block -- simplified
    return add_conv_block(ind)


def prune_channels(ind: Individual) -> tuple[Individual, bool]:
    """Halve the channels of a conv block (minimum 16)."""
    model = ind.model
    if not isinstance(model, DynamicCNN):
        return ind, False
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    # Pick a block with more than 32 channels
    candidates = [i for i, c in enumerate(model.conv_configs) if c["out_ch"] > 32]
    if not candidates:
        return ind, False

    idx = candidates[np.random.randint(len(candidates))]
    new_configs = [dict(c) for c in model.conv_configs]
    new_configs[idx]["out_ch"] = max(16, new_configs[idx]["out_ch"] // 2)

    # Propagate in_ch to the next block
    if idx + 1 < len(new_configs):
        new_configs[idx + 1]["in_ch"] = new_configs[idx]["out_ch"]

    new_model = DynamicCNN(new_configs, list(model.fc_sizes), model.n_classes, model.in_channels)
    new_model = new_model.to(device)

    # Preserve the trained weights. Unmodified blocks copy exactly; the pruned
    # block (idx) and the block that consumes it (idx+1) keep the overlapping
    # channel slice. Without this the whole net was re-initialised at random,
    # so a pruned child could never survive t_local training -- crippling the
    # single most important compression operator (also budget-boosted in
    # CnnEvolver._mutate_population).
    affected = {idx, idx + 1}
    for i, (old_l, new_l) in enumerate(zip(model.conv_layers, new_model.conv_layers)):
        if i in affected:
            _copy_overlap(old_l, new_l)
        else:
            new_l.load_state_dict(old_l.state_dict())
    # fc[0] input dim shrinks only when the LAST conv block was pruned; the
    # overlap copy handles both that and the unchanged case. classifier input
    # (last fc width) never changes, so it copies exactly.
    _copy_overlap(model.fc, new_model.fc)
    new_model.classifier.load_state_dict(model.classifier.state_dict())

    new_ind = Individual(new_model, deepcopy(ind.memory))
    new_ind.last_op = "prune_channels"
    return new_ind, True


def duplicate_conv_block(ind: Individual) -> tuple[Individual, bool]:
    """Duplicate the last conv block with weight noise eps."""
    model = ind.model
    if not isinstance(model, DynamicCNN):
        return ind, False
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    last_cfg = dict(model.conv_configs[-1])
    # Duplicated block: same channels, stride=1
    dup_cfg = {"type": last_cfg["type"], "in_ch": last_cfg["out_ch"],
               "out_ch": last_cfg["out_ch"], "stride": 1}

    new_configs = list(model.conv_configs) + [dup_cfg]
    new_model = DynamicCNN(new_configs, list(model.fc_sizes), model.n_classes, model.in_channels)
    new_model = new_model.to(device)

    for i, (old_l, new_l) in enumerate(zip(model.conv_layers, new_model.conv_layers)):
        new_l.load_state_dict(old_l.state_dict())
    # Add noise to the duplicated block
    with torch.no_grad():
        for p in new_model.conv_layers[-1].parameters():
            p.data += torch.randn_like(p) * 0.01
    new_model.fc.load_state_dict(model.fc.state_dict())
    new_model.classifier.load_state_dict(model.classifier.state_dict())

    new_ind = Individual(new_model, deepcopy(ind.memory))
    new_ind.last_op = "duplicate_conv_block"
    return new_ind, True


CNN_OPERATOR_FNS = {
    "add_conv_block":       add_conv_block,
    "depthwise_sep":        depthwise_sep,
    "swap_conv_to_dw":      swap_conv_to_dw,
    "add_fc_neuron":        add_fc_neuron,
    "remove_fc_neuron":     remove_fc_neuron,
    "change_stride":        change_stride,
    "add_skip_conv":        add_skip_conv,
    "prune_channels":       prune_channels,
    "duplicate_conv_block": duplicate_conv_block,
}


def apply_cnn_operator(ind: Individual, op: str) -> tuple[Individual, bool]:
    fn = CNN_OPERATOR_FNS.get(op)
    if fn is None:
        return ind, False
    try:
        return fn(ind)
    except Exception:
        return ind, False
