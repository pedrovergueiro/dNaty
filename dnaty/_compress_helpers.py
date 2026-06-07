"""Private helpers shared across compress functions."""
from __future__ import annotations

import torch
import torch.nn as nn


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
