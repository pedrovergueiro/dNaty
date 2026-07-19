"""
dNATY compress -- public API for model compression via evolutionary NAS.

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
from typing import Optional, Callable
import warnings

import torch
import torch.nn as nn

# Re-export for backward compat: `from dnaty.compress import CompressResult, load`
from dnaty.result import CompressResult, load  # noqa: F401
from dnaty._compress_helpers import (
    _infer_layer_sizes,
    _infer_n_classes_from_head,
    _split_backbone_head,
)


def _maybe_convert_data(train_data):
    """Convert numpy arrays, pandas DataFrames, or (X, y) tuples to DataLoader."""
    # Already a DataLoader or FastDataset — pass through
    from torch.utils.data import DataLoader
    if isinstance(train_data, DataLoader):
        return train_data
    if hasattr(train_data, "__iter__") and not isinstance(train_data, tuple):
        # FastDataset or any other iterable (not a plain tuple)
        try:
            import numpy as np
            import pandas as pd
            if not isinstance(train_data, (np.ndarray, pd.DataFrame)):
                return train_data
        except ImportError:
            return train_data

    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    # (X, y) tuple
    if isinstance(train_data, tuple) and len(train_data) == 2:
        X, y = train_data
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                X = X.values
        except ImportError:
            pass
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y.astype(np.int64))
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        return DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)

    # pandas DataFrame — last column = label
    try:
        import pandas as pd
        if isinstance(train_data, pd.DataFrame):
            y = torch.from_numpy(train_data.iloc[:, -1].values.astype(np.int64))
            X = torch.from_numpy(train_data.iloc[:, :-1].values.astype(np.float32))
            return DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)
    except ImportError:
        pass

    # numpy 2D array — last column = label
    if isinstance(train_data, np.ndarray) and train_data.ndim == 2:
        X = torch.from_numpy(train_data[:, :-1].astype(np.float32))
        y = torch.from_numpy(train_data[:, -1].astype(np.int64))
        return DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)

    return train_data


def _write_telemetry(result: "CompressResult", hw_target: str, input_size: int) -> None:
    """Append one JSON line to ~/.dnaty/telemetry.jsonl. Never raises."""
    import json
    import time
    from pathlib import Path
    try:
        from dnaty.utils.hw_detect import detect_hw
        from dnaty.utils.latency_bench import measure_latency
        hw = detect_hw()
        # Measure compressed model latency on current CPU
        lat = measure_latency(result.model, input_shape=(input_size,), n_warmup=5, n_runs=30)
        from dnaty import __version__ as _ver
        record = {
            "ts": round(time.time()),
            "dnaty_version": _ver,
            "arch": result.arch,
            "input_size": input_size,
            "original_flops": result.original_flops,
            "compressed_flops": result.compressed_flops,
            "flops_reduction_pct": round(result.flops_reduction_pct, 2),
            "accuracy": round(result.accuracy, 4),
            "hw_arch": hw.get("arch", "unknown"),
            "hw_device_class": hw.get("device_class", "unknown"),
            "hw_cores": hw.get("cores", 1),
            "hw_target": hw_target,
            "latency_p50_ms": lat["p50_ms"],
            "latency_p95_ms": lat["p95_ms"],
            "latency_fps": lat["fps"],
        }
        tele_path = Path.home() / ".dnaty" / "telemetry.jsonl"
        tele_path.parent.mkdir(exist_ok=True)
        with open(tele_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass  # telemetry must never crash the main flow


def _validate_target_flops(target_flops: float) -> None:
    if not (0.0 < target_flops <= 1.0):
        raise ValueError(
            f"target_flops={target_flops} is out of range. It is a fraction of the "
            f"original FLOPs to keep: 0.5 aims for 50% fewer FLOPs. "
            f"Valid range is (0, 1] - e.g. 0.3 aggressive, 0.5 balanced, 0.7 conservative."
        )


def compress(
    model: nn.Module,
    train_data,
    *,
    target_flops: float = 0.5,
    target: str = "flops",
    hw_target: str = "cpu",
    n_generations: int = 30,
    n_pop: int = 15,
    device: Optional[str] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    finetune_epochs: int = 30,
    sparsity: Optional[str] = None,
    quant_aware: bool = False,
    val_data=None,
) -> CompressResult:
    """
    Find a smaller, faster architecture for the same task using evolutionary NAS.

    Two-phase process:
      Phase 1 - NAS search: evolutionary search finds the best compressed
                architecture (n_generations x n_pop candidates explored).
      Phase 2 - Fine-tune: the winning architecture is trained from scratch
                for finetune_epochs to maximise accuracy on the full dataset.
                Set finetune_epochs=0 to skip (returns the NAS-phase model).

    Args:
        model:             Any PyTorch nn.Module containing nn.Linear layers.
        train_data:        DataLoader or FastDataset used for both NAS and
                           fine-tuning.
        target_flops:      Target FLOPs as fraction of original (0.5 = 50% less).
                           Ignored when target="latency".
        target:            "flops" (default) or "latency". When "latency", the
                           search minimises real ONNX Runtime latency on hw_target
                           instead of FLOPs. LatencyEvolver is used automatically.
        hw_target:         Target deployment device for latency optimisation.
                           Used when target="latency". Examples: "cpu" (measures
                           on current machine), "rpi4", "rpi5", "jetson_nano".
                           Non-cpu devices are estimated via calibrated scaling
                           tables (hw_detect.py) until measured on real hardware.
        n_generations:     Evolutionary generations (30 default).
        n_pop:             Population size (15 default).
        device:            'cpu' or 'cuda'. Auto-detected when None.
        verbose:           Print generation-by-generation progress.
        seed:              Fix for reproducibility.
        progress_callback: Optional callable(log) called each generation.
        finetune_epochs:   Epochs to train the winning arch from scratch after
                           NAS completes (default 30). Significantly improves
                           accuracy by fully converging the best architecture.
        sparsity:          Apply N:M structural sparsity after NAS. Pass "2:4"
                           for 50% structured sparsity (compatible with sparse
                           tensor cores). Applied before ONNX export. Default None.
        quant_aware:       When True, uses QuantAwareEvolver — fitness evaluation
                           runs under dynamic INT8 quantization so selected
                           architectures are guaranteed to be quantization-friendly.
                           Adds ~20% overhead per generation. Default False.
        val_data:          Optional held-out DataLoader/FastDataset. When given,
                           NAS selection and the final reported accuracy use it
                           instead of train_data — recommended, since evaluating
                           on the training set inflates the accuracy number.
                           The final accuracy is always measured in eval() mode
                           (BN running stats), matching the exported model.

    Returns:
        CompressResult with the best model found and all compression metrics.

    Example:
        >>> from dnaty import compress
        >>> from dnaty.experiments.fast_dataset import FastDataset
        >>> ds = FastDataset("MNIST", device="cpu", train_subset=10_000)
        >>> # FLOPs-guided (default)
        >>> result = compress(model, ds, target_flops=0.5, n_generations=30)
        >>> # Latency-guided for Raspberry Pi 4
        >>> result = compress(model, ds, target="latency", hw_target="rpi4")
        >>> print(result.summary())
    """
    import numpy as np
    from dnaty.evolution.evolver import DnatyEvolver, LatencyEvolver, QuantAwareEvolver

    train_data = _maybe_convert_data(train_data)
    if val_data is not None:
        val_data = _maybe_convert_data(val_data)

    # Route to LatencyEvolver when target="latency"
    if target == "latency":
        result = _compress_latency(
            model, train_data,
            hw_target=hw_target,
            n_generations=n_generations,
            n_pop=n_pop,
            device=device,
            verbose=verbose,
            seed=seed,
            progress_callback=progress_callback,
            finetune_epochs=finetune_epochs,
            quant_aware=quant_aware,
            val_data=val_data,
        )
        if sparsity:
            result = _apply_sparsity(result, sparsity, verbose)
        return result
    from dnaty.core.arch import DynamicMLP
    from dnaty.core.individual import Individual
    from dnaty.training.local_train import local_train, evaluate

    _validate_target_flops(target_flops)

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

    evolver_cls = QuantAwareEvolver if quant_aware else DnatyEvolver
    if quant_aware and verbose:
        print("[compress] quant_aware=True — using QuantAwareEvolver (INT8 fitness)")
    evolver = evolver_cls(
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

    # Phase 1: NAS search
    eval_data = val_data if val_data is not None else train_data
    evolver.run(train_data, eval_data, early_stop_patience=n_generations,
                progress_callback=progress_callback)

    # Select best: most-compressed among those above accuracy floor.
    # Fallback: when nothing passes the floor, return the highest-accuracy
    # individual (NOT min FLOPs -- that would sacrifice accuracy for nothing).
    # Adaptive floor: on tasks where 90% is unreachable the fixed floor used to
    # disable compression pressure entirely; anchor it to the best achieved
    # accuracy instead (capped at the original 0.90).
    best_pop_acc = max(ind.acc for ind in evolver.population)
    acc_floor = min(0.90, best_pop_acc - 0.02)
    candidates = [ind for ind in evolver.population if ind.acc >= acc_floor]
    if candidates:
        best = min(candidates, key=lambda ind: ind.count_flops())
    else:
        best = max(evolver.population, key=lambda ind: ind.acc)

    compressed_flops  = best.count_flops()
    compressed_params = best.count_params()

    full_sizes = list(getattr(best.model, "layer_sizes", [input_size] + init_hidden))
    arch = full_sizes[1:]

    # Phase 2: Fine-tune (continue from NAS weights)
    # Keeps the weights that NAS already tuned and polishes them at a
    # lower learning rate -- consistently better than re-initialising
    # from scratch when NAS and fine-tune use the same dataset.
    if finetune_epochs > 0:
        if verbose:
            print(f"\nPhase 2 - fine-tuning {finetune_epochs} epochs (LR 1e-4, no FLOPs pressure)...")
        nas_acc = best.acc
        for _ in range(finetune_epochs):
            local_train(
                best, train_data,
                n_epochs=1, lr=1e-4,          # lower LR polishes without disrupting
                lambda1=0.0, lambda2=0.0,     # pure accuracy, no structural pressure
                device=device, batch_size=512,
            )
        final_acc, _ = evaluate(best, eval_data, device)
        if verbose:
            delta = final_acc - nas_acc
            print(f"Fine-tune acc: {final_acc:.4f}  (NAS: {nas_acc:.4f}  delta={delta:+.4f})")
        compressed_flops  = best.count_flops()
        compressed_params = best.count_params()

    # Final reported accuracy: eval() mode (BN running stats) — the semantics of
    # the deployed/exported model, unlike the train-mode batch-stats evaluation
    # used for speed during NAS.
    final_acc, _ = evaluate(best, eval_data, device, use_train_mode=False)
    best.acc = final_acc
    best.model.eval()

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
        if input_size >= 100:
            _suggested = (f"nn.Linear({input_size}, 1024), nn.ReLU(), "
                          f"nn.Linear(1024, 512), nn.ReLU(), "
                          f"nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, {n_classes})")
        elif input_size >= 30:
            _suggested = (f"nn.Linear({input_size}, 512), nn.ReLU(), "
                          f"nn.Linear(512, 256), nn.ReLU(), "
                          f"nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, {n_classes})")
        else:
            _suggested = (f"nn.Linear({input_size}, 256), nn.ReLU(), "
                          f"nn.Linear(256, 128), nn.ReLU(), "
                          f"nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, {n_classes})")
        warnings.warn(
            f"dNATY: the output model is LARGER than the input "
            f"(FLOPs {result.flops_reduction_pct:+.1f}%, params {result.params_reduction_pct:+.1f}%). "
            f"Your baseline is already lean — dNATY needs an oversized starting point to compress. "
            f"Try a larger baseline, for example:\n"
            f"  model = nn.Sequential({_suggested})",
            UserWarning,
            stacklevel=2,
        )
    if sparsity:
        result = _apply_sparsity(result, sparsity, verbose)
    _write_telemetry(result, hw_target=hw_target, input_size=input_size)
    return result


def _apply_sparsity(result: "CompressResult", sparsity: str, verbose: bool) -> "CompressResult":
    """Apply N:M sparsity to result.model and update stats."""
    from dnaty.utils.sparsity import apply_nm_sparsity, sparsity_stats
    try:
        n, m = (int(x) for x in sparsity.split(":"))
    except (ValueError, AttributeError):
        raise ValueError(f"sparsity must be 'N:M' format (e.g. '2:4'), got: {sparsity!r}")
    apply_nm_sparsity(result.model, n=n, m=m)
    stats = sparsity_stats(result.model)
    if verbose:
        print(f"[sparsity {sparsity}] global_sparsity={stats['global_sparsity_pct']:.1f}%  "
              f"zero_weights={stats['zero_weights']:,}/{stats['total_weights']:,}")
    return result


def _compress_latency(
    model: nn.Module,
    train_data,
    *,
    hw_target: str = "cpu",
    n_generations: int = 30,
    n_pop: int = 15,
    device: Optional[str] = None,
    verbose: bool = True,
    seed: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    finetune_epochs: int = 30,
    quant_aware: bool = False,
    val_data=None,
) -> CompressResult:
    """Internal: latency-aware NAS via LatencyEvolver."""
    import numpy as np
    from dnaty.evolution.evolver import LatencyEvolver, QuantLatencyEvolver
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

    if quant_aware and verbose:
        print("[compress] quant_aware=True with target=latency — INT8 fitness + latency objective")
    evolver_cls = QuantLatencyEvolver if quant_aware else LatencyEvolver
    evolver = evolver_cls(
        n_pop=n_pop,
        n_generations=n_generations,
        t_local=3,
        input_size=input_size,
        n_classes=n_classes,
        init_hidden=init_hidden,
        device=device,
        verbose=verbose,
        target_device=hw_target,
    )

    orig_flops  = sum(2 * layer_sizes[i] * layer_sizes[i + 1] for i in range(len(layer_sizes) - 1))
    orig_params = sum(p.numel() for p in model.parameters())

    eval_data = val_data if val_data is not None else train_data
    evolver.run(train_data, eval_data, early_stop_patience=n_generations,
                progress_callback=progress_callback)

    # Best individual: highest accuracy on Pareto front
    best = max(evolver.population, key=lambda ind: ind.acc)

    compressed_flops  = best.count_flops()
    compressed_params = best.count_params()
    full_sizes = list(getattr(best.model, "layer_sizes", [input_size] + init_hidden))
    arch = full_sizes[1:]

    if finetune_epochs > 0:
        if verbose:
            print(f"\nPhase 2 - fine-tuning {finetune_epochs} epochs (LR 1e-4)...")
        for _ in range(finetune_epochs):
            local_train(best, train_data, n_epochs=1, lr=1e-4,
                        lambda1=0.0, lambda2=0.0, device=device, batch_size=512)
        compressed_flops  = best.count_flops()
        compressed_params = best.count_params()

    # Final reported accuracy: eval() mode — matches the deployed/exported model.
    final_acc, _ = evaluate(best, eval_data, device, use_train_mode=False)
    best.acc = final_acc
    best.model.eval()

    best_latency = getattr(best, "latency_ms", None)
    if verbose and best_latency:
        print(f"Best latency ({hw_target}): {best_latency:.2f} ms  "
              f"({1000/best_latency:.0f} FPS)")

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
    _write_telemetry(result, hw_target=hw_target, input_size=input_size)
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

    _validate_target_flops(target_flops)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    n_classes = 10
    if hasattr(model, "n_classes"):
        n_classes = model.n_classes
    else:
        for attr in ("fc", "classifier", "head", "heads"):
            if hasattr(model, attr):
                n_classes = _infer_n_classes_from_head(getattr(model, attr))
                break

    lambda2 = max(1e-7, 5e-6 * (1.0 - target_flops))

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
            f"dNATY: the output model is LARGER than the input "
            f"(FLOPs {result.flops_reduction_pct:+.1f}%, params {result.params_reduction_pct:+.1f}%). "
            f"Your backbone classifier head is already lean — dNATY needs an oversized starting point "
            f"to compress. Try increasing the head size before compressing, e.g. replace the final "
            f"classifier with a wider MLP (512→256→n_classes) before calling compress_with_backbone().",
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

    dNATY's NAS only optimises nn.Linear layers -- it cannot restructure conv layers.
    This function handles CNNs correctly without hiding that constraint:

      1. Freeze the backbone, extract embeddings in one pass (no training).
      2. Run NAS to find a compressed MLP head on those embeddings.
      3. Splice the compressed head back onto the original backbone.
      4. (optional) Fine-tune the full model end-to-end to recover any accuracy gap.

    Supports ResNet (fc), MobileNetV2/EfficientNet (classifier), ViT (head/heads),
    and any backbone where the last classifier is accessible as an attribute.

    Args:
        backbone:           Pretrained CNN (ResNet, MobileNetV2, EfficientNet, etc.).
        train_data:         DataLoader yielding (images, labels) -- raw image tensors.
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

    _validate_target_flops(target_flops)

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
                criterion(out, yb).backward()
                optimizer.step()
                ep_correct += (out.argmax(1) == yb).sum().item()
                ep_total += len(yb)
            if verbose and ep_total > 0:
                print(f"  Finetune epoch {ep+1}/{finetune_epochs} acc={ep_correct/ep_total:.4f}")
        result.accuracy = ep_correct / max(ep_total, 1)

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
    (torch.nn.utils.prune), which is complementary -- not overlapping.

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
            if n_prune >= module.out_channels:
                n_prune = module.out_channels - 1
            actual_amount = n_prune / module.out_channels
            prune.ln_structured(module, name="weight", amount=actual_amount, n=1, dim=0)
            if make_permanent:
                prune.remove(module, "weight")
            pruned += 1

    return model
