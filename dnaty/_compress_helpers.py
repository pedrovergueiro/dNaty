"""Private helpers shared across compress functions."""
from __future__ import annotations

import torch
import torch.nn as nn


def _build_pareto_front(evolver, orig_flops: int, eval_data, device: str) -> list[dict]:
    """Extract the non-dominated (accuracy, FLOPs) architectures from the final
    population and return them sorted by FLOPs (smallest first).

    Accuracies are re-measured in eval() mode (BatchNorm running stats — the
    semantics of the deployed model), matching how compress() reports the
    winner's final accuracy. These are NAS-phase, *un-fine-tuned* numbers: the
    front is for choosing an operating point, not a promise of post-fine-tune
    accuracy on every point.
    """
    from dnaty.evolution.selection import fast_non_dominated_sort
    from dnaty.training.local_train import evaluate

    pop = list(getattr(evolver, "population", []))
    if not pop:
        return []

    # Deduplicate by architecture signature so the front is a clean curve.
    seen: dict[tuple, "object"] = {}
    for ind in pop:
        sizes = tuple(getattr(ind.model, "layer_sizes", ()))
        n_cls = getattr(ind.model, "n_classes", None)
        sig = (sizes, n_cls, ind.count_flops())
        if sig not in seen:
            seen[sig] = ind
    uniq = list(seen.values())

    # Honest accuracy for the front: eval-mode re-measurement.
    accs = []
    for ind in uniq:
        try:
            acc, _ = evaluate(ind, eval_data, device, use_train_mode=False)
        except Exception:
            acc = ind.acc
        accs.append(acc)

    # Maximise accuracy, minimise FLOPs -> (acc, -flops) for the maximise-all sorter.
    fitnesses = [(accs[i], -float(uniq[i].count_flops())) for i in range(len(uniq))]
    fronts = fast_non_dominated_sort(fitnesses)
    front_idx = fronts[0] if fronts else list(range(len(uniq)))

    entries = []
    for i in front_idx:
        ind = uniq[i]
        flops = ind.count_flops()
        full_sizes = list(getattr(ind.model, "layer_sizes", []))
        entries.append({
            "arch": full_sizes[1:],  # hidden layer sizes (drop input dim)
            "accuracy": round(float(accs[i]), 4),
            "flops": int(flops),
            "params": int(ind.count_params()),
            "flops_reduction_pct": round((1.0 - flops / max(orig_flops, 1)) * 100, 2),
        })
    entries.sort(key=lambda e: e["flops"])
    return entries


def _infer_layer_sizes(model: nn.Module) -> list[int]:
    """Extract [in, h1, h2, ..., out] from a Linear-based model."""
    sizes: list[int] = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if not sizes:
                sizes.append(m.in_features)
            sizes.append(m.out_features)
    if len(sizes) < 2:
        raise ValueError(
            "Cannot infer architecture: model must contain at least one nn.Linear layer."
        )
    return sizes


def _infer_n_classes_from_head(head: nn.Module) -> int:
    """Get out_features from a classifier that may be Linear or Sequential."""
    if isinstance(head, nn.Linear):
        return head.out_features
    for m in reversed(list(head.modules())):
        if isinstance(m, nn.Linear):
            return m.out_features
    return 10  # safe fallback


def _split_backbone_head(
    backbone: nn.Module,
    device: str,
) -> tuple[nn.Module, int, int]:
    """
    Returns (feature_extractor, feature_dim, n_classes).
    Replaces the classifier head with Identity to expose embeddings.
    """
    import copy

    head = None
    head_attr = None
    for attr in ("fc", "classifier", "head", "heads"):
        if hasattr(backbone, attr):
            head = getattr(backbone, attr)
            head_attr = attr
            break

    if head is None:
        raise ValueError(
            "Cannot locate classifier head. Model must have a 'fc', 'classifier', "
            "'head', or 'heads' attribute. Pass feature_dim and n_classes explicitly."
        )

    n_classes = _infer_n_classes_from_head(head)

    feat_model = copy.deepcopy(backbone).to(device)
    setattr(feat_model, head_attr, nn.Identity())
    feat_model.eval()

    # Probe feature_dim with increasingly larger dummy inputs
    feature_dim = None
    for h in [32, 64, 96, 224]:
        try:
            with torch.no_grad():
                dummy = torch.zeros(1, 3, h, h, device=device)
                out = feat_model(dummy)
                feature_dim = out.view(1, -1).shape[1]
            break
        except Exception:
            continue

    if feature_dim is None:
        raise ValueError(
            "Cannot probe backbone output dimension. Pass feature_dim explicitly."
        )

    return feat_model, feature_dim, n_classes
