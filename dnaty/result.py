"""
CompressResult — return type for all dNATY compression functions.

Carries the compressed model plus all compression metrics.
Use save() / dnaty.load() to persist and reload across sessions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

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

        with torch.no_grad():
            for _ in range(n_warmup):
                model(dummy)

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
        self.model.eval()
        kwargs = dict(
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
            do_constant_folding=True,
        )
        try:
            # torch >= 2.6 defaults to the dynamo exporter, which fails on
            # DynamicMLP's data-dependent skip-connection loop. Force the
            # stable TorchScript exporter.
            torch.onnx.export(self.model, dummy, path, dynamo=False, **kwargs)
        except TypeError:
            # torch < 2.5 has no `dynamo` kwarg — TorchScript is already the default.
            torch.onnx.export(self.model, dummy, path, **kwargs)


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
    # weights_only=True: the payload is tensors + primitives only, and this
    # blocks pickle-based arbitrary code execution from untrusted .pt files.
    payload = torch.load(path, map_location="cpu", weights_only=True)
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
