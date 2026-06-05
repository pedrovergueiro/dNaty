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
    flops_reduction: float      # e.g. 0.465 = 46.5% less FLOPs
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

    def summary(self) -> str:
        return (
            f"CompressResult | arch={self.arch} | "
            f"FLOPs -{self.flops_reduction_pct:.1f}% "
            f"({self.original_flops:,} -> {self.compressed_flops:,}) | "
            f"params -{self.params_reduction_pct:.1f}% "
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

    return CompressResult(
        model=best.model,
        original_flops=orig_flops,
        compressed_flops=compressed_flops,
        original_params=orig_params,
        compressed_params=compressed_params,
        accuracy=best.acc,
        flops_reduction=max(0.0, 1.0 - compressed_flops / max(orig_flops, 1)),
        generations=n_generations,
        arch=arch,
    )


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
    elif hasattr(model, "classifier"):
        n_classes = model.classifier.out_features

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

    return CompressResult(
        model=best.model,
        original_flops=orig_flops,
        compressed_flops=compressed_flops,
        original_params=orig_params,
        compressed_params=compressed_params,
        accuracy=best.acc,
        flops_reduction=max(0.0, 1.0 - compressed_flops / max(orig_flops, 1)),
        generations=n_generations,
        arch=[],
    )


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
