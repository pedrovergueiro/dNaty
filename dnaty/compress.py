"""
dNATY compress — public API for model compression via evolutionary NAS.

Usage:
    from dnaty import compress

    result = compress(model, train_data, target_flops=0.5)
    print(f"Compressed {result.flops_reduction_pct:.1f}% FLOPs, acc={result.accuracy:.4f}")
    result.model  # ready-to-use compressed PyTorch model
    result.save("model_compressed.pt")

    # Reload later
    result = dnaty.load("model_compressed.pt")
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import warnings

import torch
import torch.nn as nn


@dataclass
class CompressResult:
    model: nn.Module
    original_flops: int
    compressed_flops: int
    original_params: int
    compressed_params: int
    accuracy: float
    flops_reduction: float      # positive = compressed, negative = model grew
    generations: int
    arch: list[int] = field(default_factory=list)   # hidden layer sizes found

    @property
    def flops_reduction_pct(self) -> float:
        return self.flops_reduction * 100

    @property
    def params_reduction_pct(self) -> float:
        if self.original_params == 0:
            return 0.0
        return (1.0 - self.compressed_params / self.original_params) * 100

    @property
    def model_grew(self) -> bool:
        return self.flops_reduction < 0

    def summary(self) -> str:
        def _fmt(pct: float) -> str:
            sign = "↓" if pct > 0 else "↑" if pct < 0 else "="
            return f"{sign}{abs(pct):.1f}%"
        return (
            f"CompressResult | arch={self.arch} | "
            f"FLOPs {_fmt(self.flops_reduction_pct)} "
            f"({self.original_flops:,} -> {self.compressed_flops:,}) | "
            f"params {_fmt(self.params_reduction_pct)} "
            f"({self.original_params:,} -> {self.compressed_params:,}) | "
            f"acc={self.accuracy:.4f}"
        )

    def save(self, path: str) -> None:
        """Persist the compressed model and all metrics to a .pt file."""
        from dnaty.core.arch import DynamicMLP
        payload = {
            "layer_sizes": list(self.model.layer_sizes) if hasattr(self.model, "layer_sizes") else [],
            "activations": list(self.model.activations) if hasattr(self.model, "activations") else [],
            "n_classes": self.model.n_classes if hasattr(self.model, "n_classes") else None,
            "model_state": self.model.state_dict(),
            "original_flops": self.original_flops,
            "compressed_flops": self.compressed_flops,
            "original_params": self.original_params,
            "compressed_params": self.compressed_params,
            "accuracy": self.accuracy,
            "flops_reduction": self.flops_reduction,
            "generations": self.generations,
            "arch": self.arch,
        }
        torch.save(payload, path)

    def benchmark_latency(
        self,
        input_shape: tuple,
        n_warmup: int = 20,
        n_runs: int = 200,
        batch_size: int = 1,
        device: Optional[str] = None,
    ) -> dict:
        """Measure real inference latency (p50/p95/p99 in milliseconds).

        Designed for edge deployment validation (Raspberry Pi, drones, cameras).
        Uses CPU by default — matches target hardware that has no GPU.

        Args:
            input_shape: Shape of a single sample, e.g. (784,) or (3, 32, 32).
            n_warmup:    Warm-up runs before timing (fills caches, JIT).
            n_runs:      Timed runs for statistics.
            batch_size:  Batch size per inference call (1 = real-time edge mode).
            device:      'cpu' or 'cuda'. Defaults to 'cpu'.

        Returns:
            dict with p50_ms, p95_ms, p99_ms, mean_ms, fps.

        Example:
            result.benchmark_latency((784,))
            # {'p50_ms': 0.12, 'p95_ms': 0.18, 'fps': 5400, ...}
        """
        import time
        dev = device or "cpu"
        model = self.model.to(dev)
        model.eval()
        dummy = torch.zeros(batch_size, *input_shape, device=dev)

        # Warm-up
        with torch.no_grad():
            for _ in range(n_warmup):
                model(dummy)

        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                model(dummy)
                if dev == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)

        times = sorted(times)
        p50 = float(times[int(0.50 * n_runs)])
        p95 = float(times[int(0.95 * n_runs)])
        p99 = float(times[int(0.99 * n_runs)])
        mean = float(sum(times) / n_runs)
        fps = 1000.0 / mean if mean > 0 else float("inf")

        return {
            "p50_ms": round(p50, 3),
            "p95_ms": round(p95, 3),
            "p99_ms": round(p99, 3),
            "mean_ms": round(mean, 3),
            "fps": round(fps, 1),
            "device": dev,
            "batch_size": batch_size,
        }

    def export_onnx(self, path: str, input_shape: tuple) -> None:
        """Export the compressed model to ONNX for CPU deployment (drones, cameras, robots).

        Args:
            path:        Output file path, e.g. "model.onnx".
            input_shape: Shape of a single input sample, e.g. (784,) for MNIST or (3072,) for CIFAR-10.
                         Do NOT include the batch dimension.

        Example:
            result.export_onnx("model.onnx", input_shape=(784,))
        """
        dummy = torch.zeros(1, *input_shape)
        input_names = ["input"]
        output_names = ["output"]
        torch.onnx.export(
            self.model,
            dummy,
            path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
            do_constant_folding=True,
        )


def load(path: str) -> CompressResult:
    """Reload a CompressResult previously saved with result.save().

    Args:
        path: Path to the .pt file created by CompressResult.save().

    Returns:
        CompressResult with the reconstructed model and all compression metrics.

    Example:
        result = dnaty.load("model_compressed.pt")
        print(result.summary())
    """
    from dnaty.core.arch import DynamicMLP
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DynamicMLP(payload["layer_sizes"], payload["activations"], payload["n_classes"])
    model.load_state_dict(payload["model_state"])
    model.eval()
    return CompressResult(
        model=model,
        original_flops=payload["original_flops"],
        compressed_flops=payload["compressed_flops"],
        original_params=payload["original_params"],
        compressed_params=payload["compressed_params"],
        accuracy=payload["accuracy"],
        flops_reduction=payload["flops_reduction"],
        generations=payload["generations"],
        arch=payload["arch"],
    )


def compress(
    model: nn.Module,
    train_data,
    *,
    target_flops: float = 0.5,
    n_generations: int = 30,
    n_pop: int = 15,
    device: Optional[str] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    finetune_epochs: int = 30,
) -> CompressResult:
    """
    Find a smaller, faster architecture for the same task using evolutionary NAS.

    Two-phase process:
      Phase 1 — NAS search: evolutionary search finds the best compressed
                architecture (n_generations × n_pop candidates explored).
      Phase 2 — Fine-tune: the winning architecture is trained from scratch
                for finetune_epochs to maximise accuracy on the full dataset.
                Set finetune_epochs=0 to skip (returns the NAS-phase model).

    Args:
        model:             Any PyTorch nn.Module containing nn.Linear layers.
        train_data:        DataLoader or FastDataset used for both NAS and
                           fine-tuning.
        target_flops:      Target FLOPs as fraction of original (0.5 = 50% less).
        n_generations:     Evolutionary generations (30 default).
        n_pop:             Population size (15 default).
        device:            'cpu' or 'cuda'. Auto-detected when None.
        verbose:           Print generation-by-generation progress.
        seed:              Fix for reproducibility.
        progress_callback: Optional callable(log) called each generation.
        finetune_epochs:   Epochs to train the winning arch from scratch after
                           NAS completes (default 30). Significantly improves
                           accuracy by fully converging the best architecture.

    Returns:
        CompressResult with the best model found and all compression metrics.

    Example:
        >>> from dnaty import compress
        >>> from dnaty.experiments.fast_dataset import FastDataset
        >>> ds = FastDataset("MNIST", device="cpu", train_subset=10_000)
        >>> result = compress(model, ds, target_flops=0.5, n_generations=30)
        >>> print(result.summary())
    """
    import numpy as np
    from dnaty.evolution.evolver import DnatyEvolver
    from dnaty.core.arch import DynamicMLP
    from dnaty.core.individual import Individual
    from dnaty.training.local_train import local_train, evaluate

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    layer_sizes = _infer_layer_sizes(model)
    input_size  = layer_sizes[0]
    n_classes   = layer_sizes[-1]
    init_hidden = layer_sizes[1:-1]

    lambda2 = max(1e-7, 5e-6 * (1.0 - target_flops))

    evolver = DnatyEvolver(
        n_pop=n_pop,
        n_generations=n_generations,
        t_local=3,
        input_size=input_size,
        n_classes=n_classes,
        init_hidden=init_hidden,
        device=device,
        verbose=verbose,
        lambda2=lambda2,
    )

    orig_flops  = sum(2 * layer_sizes[i] * layer_sizes[i + 1] for i in range(len(layer_sizes) - 1))
    orig_params = sum(p.numel() for p in model.parameters())

    # ── Phase 1: NAS search ────────────────────────────────────────
    evolver.run(train_data, train_data, early_stop_patience=n_generations,
                progress_callback=progress_callback)

    # Select best: most-compressed among those above accuracy floor.
    # Fallback: when nothing passes the floor, return the highest-accuracy
    # individual (NOT min FLOPs — that would sacrifice accuracy for nothing).
    acc_floor = 0.90
    candidates = [ind for ind in evolver.population if ind.acc >= acc_floor]
    if candidates:
        best = min(candidates, key=lambda ind: ind.count_flops())
    else:
        best = max(evolver.population, key=lambda ind: ind.acc)

    compressed_flops  = best.count_flops()
    compressed_params = best.count_params()

    full_sizes = list(getattr(best.model, "layer_sizes", [input_size] + init_hidden))
    arch = full_sizes[1:]

    # ── Phase 2: Fine-tune (continue from NAS weights) ────────────
    # Keeps the weights that NAS already tuned and polishes them at a
    # lower learning rate — consistently better than re-initialising
    # from scratch when NAS and fine-tune use the same dataset.
    if finetune_epochs > 0:
        if verbose:
            print(f"\nPhase 2 — fine-tuning {finetune_epochs} epochs (LR 1e-4, no FLOPs pressure)...")
        nas_acc = best.acc
        for _ in range(finetune_epochs):
            local_train(
                best, train_data,
                n_epochs=1, lr=1e-4,          # lower LR polishes without disrupting
                lambda1=0.0, lambda2=0.0,     # pure accuracy, no structural pressure
                device=device, batch_size=512,
            )
        final_acc, _ = evaluate(best, train_data, device)
        best.acc = final_acc
        if verbose:
            delta = final_acc - nas_acc
            print(f"Fine-tune acc: {final_acc:.4f}  (NAS: {nas_acc:.4f}  Δ={delta:+.4f})")
        compressed_flops  = best.count_flops()
        compressed_params = best.count_params()

    flops_reduction = 1.0 - compressed_flops / max(orig_flops, 1)
    result = CompressResult(
        model=best.model,
        original_flops=orig_flops,
        compressed_flops=compressed_flops,
        original_params=orig_params,
        compressed_params=compressed_params,
        accuracy=best.acc,
        flops_reduction=flops_reduction,
        generations=n_generations,
        arch=arch,
    )
    if result.model_grew:
        warnings.warn(
            f"dNATY: compressed model is LARGER than the original "
            f"(FLOPs changed {result.flops_reduction_pct:+.1f}%, "
            f"params changed {result.params_reduction_pct:+.1f}%). "
            f"This usually means the input model was undersized for the task complexity. "
            f"Try passing a larger initial model (more hidden units).",
            UserWarning,
            stacklevel=2,
        )
    return result


def compress_cnn(
    model: nn.Module,
    train_data,
    *,
    target_flops: float = 0.5,
    n_generations: int = 30,
    n_pop: int = 15,
    device: Optional[str] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
) -> CompressResult:
    """
    Find a smaller, faster CNN architecture for image tasks using evolutionary NAS.

    Parallel to compress() but targets Conv2D-based models (CIFAR-10, custom image
    datasets). Uses CnnEvolver with real depthwise-separable and conv block operators.

    NOTE: This is an early-access API. DynamicCNN is stable for CIFAR-10 classification.
    Detection and segmentation are not yet supported.

    Args:
        model:             A DynamicCNN or any nn.Module (architecture is inferred).
        train_data:        DataLoader with (N, C, H, W) image batches.
        target_flops:      Target FLOPs as fraction of original (0.5 = 50% less).
        n_generations:     Evolutionary generations (30 default).
        n_pop:             Population size (15 default).
        device:            'cpu' or 'cuda'. Auto-detected when None.
        verbose:           Print generation-by-generation progress.
        seed:              Fix for reproducibility.
        progress_callback: Optional callable(log) called each generation.

    Returns:
        CompressResult with the best CNN found and compression metrics.

    Example:
        >>> from dnaty import compress_cnn
        >>> from dnaty.core.arch_cnn import DynamicCNN
        >>> model = DynamicCNN()  # or your own DynamicCNN
        >>> result = compress_cnn(model, cifar_loader, target_flops=0.5)
        >>> print(result.summary())
    """
    import numpy as np
    from dnaty.evolution.evolver import CnnEvolver
    from dnaty.core.arch_cnn import DynamicCNN

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Infer n_classes from model
    n_classes = 10
    if hasattr(model, "n_classes"):
        n_classes = model.n_classes
    else:
        for attr in ("fc", "classifier", "head", "heads"):
            if hasattr(model, attr):
                n_classes = _infer_n_classes_from_head(getattr(model, attr))
                break

    lambda2 = max(1e-7, 5e-6 * (1.0 - target_flops))

    # Use DynamicCNN defaults if starting from scratch or non-DynamicCNN model
    if isinstance(model, DynamicCNN):
        conv_configs = list(model.conv_configs)
        fc_sizes = list(model.fc_sizes)
        in_channels = model.in_channels
    else:
        conv_configs = None  # DynamicCNN default: 3 conv blocks for CIFAR-10
        fc_sizes = None
        in_channels = 3

    evolver = CnnEvolver(
        n_pop=n_pop,
        n_generations=n_generations,
        t_local=3,
        n_classes=n_classes,
        device=device,
        verbose=verbose,
        lambda2=lambda2,
    )

    orig_flops  = model.count_flops() if hasattr(model, "count_flops") else 0
    orig_params = sum(p.numel() for p in model.parameters())

    evolver.run(train_data, train_data, early_stop_patience=n_generations,
                progress_callback=progress_callback)

    acc_floor = 0.90
    candidates = [ind for ind in evolver.population if ind.acc >= acc_floor]
    if not candidates:
        candidates = evolver.population
    best = min(candidates, key=lambda ind: ind.count_flops())

    compressed_flops  = best.count_flops()
    compressed_params = best.count_params()

    flops_reduction = 1.0 - compressed_flops / max(orig_flops, 1)
    result = CompressResult(
        model=best.model,
        original_flops=orig_flops,
        compressed_flops=compressed_flops,
        original_params=orig_params,
        compressed_params=compressed_params,
        accuracy=best.acc,
        flops_reduction=flops_reduction,
        generations=n_generations,
        arch=[],
    )
    if result.model_grew:
        warnings.warn(
            f"dNATY: compressed model is LARGER than the original "
            f"(FLOPs changed {result.flops_reduction_pct:+.1f}%, "
            f"params changed {result.params_reduction_pct:+.1f}%). "
            f"This usually means the input model was undersized for the task complexity. "
            f"Try passing a larger initial model (more hidden units).",
            UserWarning,
            stacklevel=2,
        )
    return result


def compress_with_backbone(
    backbone: nn.Module,
    train_data,
    *,
    target_flops: float = 0.5,
    n_generations: int = 30,
    n_pop: int = 15,
    device: Optional[str] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
    finetune_backbone: bool = False,
    finetune_epochs: int = 10,
    feature_dim: Optional[int] = None,
    n_classes: Optional[int] = None,
    batch_size: int = 64,
    progress_callback: Optional[Callable] = None,
) -> CompressResult:
    """
    Compress the classifier head of a CNN backbone using evolutionary NAS.

    dNATY's NAS only optimises nn.Linear layers — it cannot restructure conv layers.
    This function handles CNNs correctly without hiding that constraint:

      1. Freeze the backbone, extract embeddings in one pass (no training).
      2. Run NAS to find a compressed MLP head on those embeddings.
      3. Splice the compressed head back onto the original backbone.
      4. (optional) Fine-tune the full model end-to-end to recover any accuracy gap.

    Supports ResNet (fc), MobileNetV2/EfficientNet (classifier), ViT (head/heads),
    and any backbone where the last classifier is accessible as an attribute.

    Args:
        backbone:           Pretrained CNN (ResNet, MobileNetV2, EfficientNet, etc.).
        train_data:         DataLoader yielding (images, labels) — raw image tensors.
        target_flops:       FLOPs target as fraction of original head FLOPs (0.5 = 50% less).
        n_generations:      NAS search generations.
        n_pop:              Population size per generation.
        device:             'cpu' or 'cuda'. Auto-detected when None.
        verbose:            Print per-generation progress.
        seed:               Reproducibility seed.
        finetune_backbone:  After NAS, fine-tune the full backbone+head end-to-end.
        finetune_epochs:    Epochs for end-to-end fine-tuning (only if finetune_backbone=True).
        feature_dim:        Override auto-detected backbone output dimension.
        n_classes:          Override auto-detected number of classes.
        batch_size:         Batch size for feature extraction and fine-tuning.
        progress_callback:  Optional callable(log) per NAS generation.

    Returns:
        CompressResult where .model is the full backbone + compressed head.
        FLOPs/params metrics cover the compressed head only.

    Example:
        >>> import torchvision.models as tv
        >>> backbone = tv.mobilenet_v2(weights="IMAGENET1K_V1")
        >>> result = compress_with_backbone(backbone, cifar_loader, target_flops=0.4)
        >>> print(result.summary())
        >>> result.model  # full model, ready for inference
    """
    import copy
    from torch.utils.data import DataLoader, TensorDataset

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- 1. Split backbone / head ---------------------------------------------------
    if feature_dim is None or n_classes is None:
        _feat_model, _feat_dim, _n_cls = _split_backbone_head(backbone, device)
        feature_dim = feature_dim or _feat_dim
        n_classes   = n_classes   or _n_cls
        feat_model  = _feat_model
    else:
        feat_model = copy.deepcopy(backbone).to(device)
        for attr in ("fc", "classifier", "head", "heads"):
            if hasattr(feat_model, attr):
                setattr(feat_model, attr, nn.Identity())
                break

    # -- 2. Extract embeddings (frozen backbone, single pass) ----------------------
    if verbose:
        print(f"[compress_with_backbone] Extracting features: dim={feature_dim}, classes={n_classes}")

    feat_model.eval()
    all_X, all_y = [], []
    loader = train_data if hasattr(train_data, "__iter__") else DataLoader(train_data, batch_size=batch_size)
    with torch.no_grad():
        for batch in loader:
            xb, yb = batch[0].to(device), batch[1]
            feats = feat_model(xb)
            if feats.ndim > 2:
                feats = feats.view(feats.size(0), -1)
            all_X.append(feats.cpu())
            all_y.append(yb.cpu() if isinstance(yb, torch.Tensor) else torch.tensor(yb))

    X = torch.cat(all_X)
    y = torch.cat(all_y)
    X = (X - X.mean(0)) / X.std(0).clamp_min(1e-7)  # z-score norm

    emb_loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    if verbose:
        print(f"  -> {len(X):,} embeddings extracted. Running NAS on MLP head...")

    # -- 3. Build MLP head and compress it via NAS ---------------------------------
    if feature_dim >= 1000:
        hidden = [512, 256, 128]
    elif feature_dim >= 300:
        hidden = [256, 128, 64]
    else:
        hidden = [128, 64]

    layers: list[nn.Module] = []
    prev = feature_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    head_model = nn.Sequential(*layers)

    result = compress(
        head_model, emb_loader,
        target_flops=target_flops,
        n_generations=n_generations,
        n_pop=n_pop,
        device=device,
        verbose=verbose,
        seed=seed,
        progress_callback=progress_callback,
    )

    # -- 4. Splice compressed head back onto backbone ------------------------------
    full_model = copy.deepcopy(backbone).to(device)
    for attr in ("fc", "classifier", "head", "heads"):
        if hasattr(full_model, attr):
            original_head = getattr(full_model, attr)
            if isinstance(original_head, nn.Sequential):
                # Preserve Dropout / BN before the final Linear if any
                pre = [m for m in original_head.children() if not isinstance(m, nn.Linear)]
                new_head = nn.Sequential(*pre, *result.model.children()) if pre else result.model
            else:
                new_head = result.model
            setattr(full_model, attr, new_head)
            break

    # -- 5. Optional end-to-end fine-tuning ----------------------------------------
    if finetune_backbone and finetune_epochs > 0:
        if verbose:
            print(f"\nFine-tuning full backbone+head end-to-end ({finetune_epochs} epochs)...")
        full_model.train()
        optimizer = torch.optim.Adam(full_model.parameters(), lr=1e-4, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        raw_loader = train_data if hasattr(train_data, "__iter__") else DataLoader(train_data, batch_size=batch_size)
        for ep in range(finetune_epochs):
            ep_correct = ep_total = 0
            for xb, yb in raw_loader:
                xb = xb.to(device)
                yb = yb.to(device) if isinstance(yb, torch.Tensor) else torch.tensor(yb, device=device)
                optimizer.zero_grad()
                out = full_model(xb)
                nn.CrossEntropyLoss()(out, yb).backward()
                optimizer.step()
                ep_correct += (out.argmax(1) == yb).sum().item()
                ep_total += len(yb)
            if verbose:
                print(f"  Finetune epoch {ep+1}/{finetune_epochs} acc={ep_correct/ep_total:.4f}")
        result.accuracy = ep_correct / ep_total

    result.model = full_model
    return result


def prune_conv_channels(
    model: nn.Module,
    amount: float = 0.3,
    make_permanent: bool = True,
) -> nn.Module:
    """
    Prune the least-important output channels of all Conv2d layers (L1-norm structured pruning).

    dNATY's NAS compresses MLP layers via architecture search.
    This function handles the conv backbone via structured channel pruning
    (torch.nn.utils.prune), which is complementary — not overlapping.

    Typical workflow for CNN compression:
        1. prune_conv_channels(backbone, amount=0.3)  # prune backbone channels
        2. compress_with_backbone(backbone, data)     # NAS-compress the head
        3. result.export_onnx(...)                    # deploy to edge

    Args:
        model:           Any nn.Module with nn.Conv2d layers.
        amount:          Fraction of channels to zero out per layer (0.3 = 30%).
        make_permanent:  If True, remove prune buffers and bake masks into weights.
                         Required before ONNX export or fine-tuning.

    Returns:
        The pruned model (modified in-place, also returned for chaining).

    Example:
        >>> from dnaty import prune_conv_channels
        >>> model = prune_conv_channels(resnet18, amount=0.3)
        >>> result = compress_with_backbone(model, loader, target_flops=0.5)
    """
    from torch.nn.utils import prune

    pruned = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.out_channels > 1:
            n_prune = max(1, int(module.out_channels * amount))
            # Keep at least 1 channel
            if n_prune >= module.out_channels:
                n_prune = module.out_channels - 1
            actual_amount = n_prune / module.out_channels
            prune.ln_structured(module, name="weight", amount=actual_amount, n=1, dim=0)
            if make_permanent:
                prune.remove(module, "weight")
            pruned += 1

    return model


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
