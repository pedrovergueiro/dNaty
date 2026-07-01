"""
PyTorch Lightning integration for dNATY.

Automatically compresses the trained model at the end of a Lightning training run.

Usage:
    from dnaty.integrations.lightning import DNATYCallback
    from pytorch_lightning import Trainer

    callback = DNATYCallback(target_flops=0.5, n_generations=20)
    trainer = Trainer(max_epochs=30, callbacks=[callback])
    trainer.fit(lit_model, datamodule)

    # After training:
    result = callback.result
    print(result.summary())
    result.export_onnx("compressed.onnx", input_shape=(784,))

Requires:
    pip install pytorch-lightning
"""
from __future__ import annotations

from typing import Optional, Callable, Any

try:
    import pytorch_lightning as pl
    _PL_AVAILABLE = True
except ImportError:
    _PL_AVAILABLE = False


class _DNATYCallbackBase:
    """
    dNATY compression callback for PyTorch Lightning.

    Runs evolutionary NAS compression at the end of trainer.fit(). The
    compressed model and all metrics are stored in self.result.

    Args:
        target_flops:      FLOPs target as fraction of original (0.5 = 50% less).
        n_generations:     NAS search generations (default 20 — faster than full 30).
        n_pop:             Population size (default 10).
        finetune_epochs:   Fine-tune epochs after NAS (default 10). Set 0 to skip.
        verbose:           Print compression progress (default True).
        progress_callback: Optional callable(log) called each NAS generation.
        auto_export_onnx:  If set to a path string (e.g. "model.onnx"), automatically
                           exports ONNX after compression. input_shape must also be set.
        input_shape:       Input shape WITHOUT batch dim, e.g. (784,). Required for
                           auto_export_onnx and benchmark_latency.

    Attributes:
        result: CompressResult — set after on_train_end() completes.

    Example:
        >>> callback = DNATYCallback(target_flops=0.5, auto_export_onnx="out.onnx",
        ...                          input_shape=(784,))
        >>> Trainer(callbacks=[callback]).fit(model, datamodule)
        >>> callback.result.summary()
    """

    def __init__(
        self,
        target_flops: float = 0.5,
        n_generations: int = 20,
        n_pop: int = 10,
        finetune_epochs: int = 10,
        verbose: bool = True,
        progress_callback: Optional[Callable] = None,
        auto_export_onnx: Optional[str] = None,
        input_shape: Optional[tuple] = None,
    ) -> None:
        if not _PL_AVAILABLE:
            raise ImportError(
                "pytorch_lightning is required: pip install pytorch-lightning\n"
                "Or install the Lightning-compatible package: pip install lightning"
            )
        self.target_flops = target_flops
        self.n_generations = n_generations
        self.n_pop = n_pop
        self.finetune_epochs = finetune_epochs
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.auto_export_onnx = auto_export_onnx
        self.input_shape = input_shape
        self.result = None

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        from dnaty import compress

        # Extract the underlying nn.Module
        model = pl_module
        if hasattr(pl_module, "model"):
            model = pl_module.model
        elif hasattr(pl_module, "net"):
            model = pl_module.net

        # Try to get the train dataloader
        train_dl = None
        try:
            train_dl = trainer.train_dataloader
            # Lightning wraps it; unwrap if needed
            if hasattr(train_dl, "loaders"):
                train_dl = train_dl.loaders
        except Exception:
            pass

        if train_dl is None:
            print("[DNATYCallback] Could not extract train_dataloader — skipping compression.")
            return

        if self.verbose:
            print(
                f"\n[DNATYCallback] Starting compression "
                f"(target_flops={self.target_flops}, "
                f"generations={self.n_generations}, "
                f"pop={self.n_pop})"
            )

        try:
            self.result = compress(
                model,
                train_dl,
                target_flops=self.target_flops,
                n_generations=self.n_generations,
                n_pop=self.n_pop,
                finetune_epochs=self.finetune_epochs,
                verbose=self.verbose,
                progress_callback=self.progress_callback,
            )
            if self.verbose:
                print(f"[DNATYCallback] {self.result.summary()}")

            if self.auto_export_onnx and self.input_shape:
                self.result.export_onnx(self.auto_export_onnx, input_shape=self.input_shape)
                if self.verbose:
                    print(f"[DNATYCallback] ONNX exported → {self.auto_export_onnx}")

        except Exception as exc:
            print(f"[DNATYCallback] Compression failed: {exc}")


if _PL_AVAILABLE:
    class DNATYCallback(_DNATYCallbackBase, pl.Callback):
        """dNATY PyTorch Lightning callback. See _DNATYCallbackBase for full docs."""
        pass
else:
    class DNATYCallback(_DNATYCallbackBase):  # type: ignore[no-redef]
        """Stub: pytorch_lightning not installed. Install it to use this callback."""
        pass
