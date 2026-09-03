"""
CompressResult -- return type for all dNATY compression functions.

Carries the compressed model plus all compression metrics.
Use save() / dnaty.load() to persist and reload across sessions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

try:
    from dnaty import __version__
except ImportError:
    __version__ = "unknown"


def _set_onnx_meta(model_proto: "onnx.ModelProto", key: str, value: str) -> None:
    """Upsert a string key/value in ONNX model metadata_props."""
    for prop in model_proto.metadata_props:
        if prop.key == key:
            prop.value = value
            return
    entry = model_proto.metadata_props.add()
    entry.key = key
    entry.value = value


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

    # v2.1.0 --------------------------------------------------------------- #
    # Non-dominated accuracy/FLOPs/params trade-offs discovered during search
    # (the full Pareto front, not just the returned winner). Each entry:
    #   {"arch", "accuracy", "flops", "params", "flops_reduction_pct"}.
    pareto_front: list = field(default_factory=list)
    # Transferable operator prior (dnaty.core.memory.to_prior format). Pass it
    # to a later compress(..., warm_start=...) on a related task.
    operator_priors: dict = field(default_factory=dict)

    # v2.2.0 --------------------------------------------------------------- #
    # Meta-controller policy learned during the search (populated when
    # compress(..., controller=True)). Pass it to a later
    # compress(..., controller=True, controller_policy=...) to transfer *how
    # to search* — complementary to operator_priors (*what worked*).
    controller_policy: dict = field(default_factory=dict)

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
            sign = "-" if pct > 0 else "+" if pct < 0 else "="
            return f"{sign}{abs(pct):.1f}%"
        return (
            f"CompressResult | arch={self.arch} | "
            f"FLOPs {_fmt(self.flops_reduction_pct)} "
            f"({self.original_flops:,} -> {self.compressed_flops:,}) | "
            f"params {_fmt(self.params_reduction_pct)} "
            f"({self.original_params:,} -> {self.compressed_params:,}) | "
            f"acc={self.accuracy:.4f}"
        )

    def pareto_summary(self) -> str:
        """One line per Pareto-optimal architecture found during search.

        Lets you pick the model that fits a specific device budget instead of
        being handed a single winner. Accuracies are NAS-phase (eval-mode)
        validation numbers for the *un-fine-tuned* architectures; the returned
        `.model` is the fine-tuned winner and carries the headline `.accuracy`.
        """
        if not self.pareto_front:
            return "Pareto front unavailable (populated by compress(); not persisted by save())."
        lines = [f"Pareto front — {len(self.pareto_front)} non-dominated architecture(s):"]
        for p in self.pareto_front:
            lines.append(
                f"  arch={p['arch']}  acc={p['accuracy']:.4f}  "
                f"FLOPs={p['flops']:,} (-{p['flops_reduction_pct']:.1f}%)  "
                f"params={p['params']:,}"
            )
        return "\n".join(lines)

    def pareto_front_csv(self, path: str) -> None:
        """Write the Pareto front to a CSV (arch, accuracy, flops, params, ...).

        Handy for plotting the accuracy/FLOPs trade-off curve in a paper or for
        choosing a deployment point per device budget.
        """
        import csv
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["arch", "accuracy", "flops", "params", "flops_reduction_pct"])
            for p in self.pareto_front:
                writer.writerow([
                    "-".join(str(x) for x in p["arch"]),
                    f"{p['accuracy']:.6f}", p["flops"], p["params"],
                    f"{p['flops_reduction_pct']:.4f}",
                ])

    def export_memory(self) -> dict:
        """Return the transferable operator prior learned during this search.

        Pass it straight into a later run on a related task:
            r1 = compress(model_a, data_a)
            r2 = compress(model_b, data_b, warm_start=r1.export_memory())
        """
        return dict(self.operator_priors)

    def save_memory(self, path: str) -> None:
        """Persist the learned operator prior to JSON for future warm-starts.

        Later:  compress(model, data, warm_start="prior.json")
        """
        if not self.operator_priors:
            raise ValueError(
                "No operator prior on this result. save_memory() works on results "
                "returned by compress(); it is not restored by dnaty.load()."
            )
        from dnaty.core.memory import save_prior
        save_prior(self.operator_priors, path)

    def save(self, path: str) -> None:
        """Persist the compressed model and all metrics to a .pt file."""
        from dnaty.core.arch import DynamicMLP
        skip_meta = [
            [src, dst, proj_idx]
            for src, dst, proj_idx in getattr(self.model, "skip_connections", [])
        ]
        proj_dims = [
            [p.in_features, p.out_features]
            for p in getattr(self.model, "skip_projs", [])
        ]
        payload = {
            "layer_sizes": list(self.model.layer_sizes) if hasattr(self.model, "layer_sizes") else [],
            "activations": list(self.model.activations) if hasattr(self.model, "activations") else [],
            "n_classes": self.model.n_classes if hasattr(self.model, "n_classes") else None,
            "skip_connections": skip_meta,
            "skip_proj_dims": proj_dims,
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
        Uses CPU by default -- matches target hardware that has no GPU.

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

    def quantize(self, dtype=None) -> "CompressResult":
        """Apply dynamic INT8 quantization to the compressed model.

        Returns a new CompressResult with the quantized model ready for CPU inference.
        No calibration data needed — uses dynamic quantization (weights INT8,
        activations quantized at runtime).

        Args:
            dtype: torch quantization dtype. Default: torch.qint8.

        Returns:
            New CompressResult with quantized model (all other metrics unchanged).

        Example:
            result = compress(model, data)
            q_result = result.quantize()
            q_result.model  # INT8 PyTorch model, ~2-4x faster on CPU
            q_result.export_onnx("model_int8.onnx", input_shape=(784,))
        """
        import copy
        if dtype is None:
            dtype = torch.qint8
        q_model = torch.quantization.quantize_dynamic(
            copy.deepcopy(self.model).cpu(), {nn.Linear}, dtype=dtype
        )
        q = copy.copy(self)
        q.model = q_model
        return q

    def adapt(
        self,
        new_x,
        new_y,
        epochs: int = 3,
        lr: float = 1e-4,
        batch_size: int = 256,
        replay: bool = True,
        replay_capacity: int = 2048,
        plasticity: bool = True,
        reinit_fraction: float = 0.05,
        device: str = "cpu",
        seed: "int | None" = None,
    ) -> dict:
        """Lifelong on-device adaptation (v2.2.0): learn from drifted data
        without a full recompress and without forgetting the old distribution.

        Three mechanisms, each optional:
          replay      -- a bounded reservoir buffer keeps an unbiased sample of
                         everything adapt() has seen; each adaptation trains on
                         new data *mixed with* replayed old data, so the old
                         distribution keeps applying pressure (anti-forgetting).
          plasticity  -- before training, the lowest-utility hidden units are
                         reborn (continual-backprop-style reinit with zeroed
                         outgoing weights), freeing capacity for the new data.
                         Reborn units start silent, so no random noise is
                         injected; only the dormant units' negligible old
                         contribution is dropped.
          fine-tune   -- a few epochs of Adam at a low LR on the mixed batch.

        Pairs naturally with monitoring:
            if tracker.predict(batch)[1]["alert"]:
                result.adapt(batch_x, batch_y)

        Args:
            new_x, new_y:    the drifted batch (tensor / numpy; y int labels).
            epochs:          fine-tune epochs over the mixed data.
            lr:              fine-tune learning rate.
            replay:          mix in replayed old samples (1:1 with new data).
            replay_capacity: reservoir size (first call only).
            plasticity:      reinit dormant units before fine-tuning.
            reinit_fraction: fraction of units per layer reborn.
            seed:            seeds sampling + reinit for reproducibility.

        Returns:
            dict with acc_before/acc_after (on the new batch),
            retention_before/retention_after (on replayed old data; None on
            the first call), n_reborn, replay_size.
        """
        import numpy as _np
        from dnaty.training.replay import ReplayBuffer
        from dnaty.training.local_train import _bn_eval_if_tiny

        if isinstance(new_x, torch.Tensor):
            new_x = new_x.detach().cpu().float()
        else:
            new_x = torch.as_tensor(_np.asarray(new_x)).float()
        if isinstance(new_y, torch.Tensor):
            new_y = new_y.detach().cpu().long()
        else:
            new_y = torch.as_tensor(_np.asarray(new_y)).long()
        if len(new_x) != len(new_y):
            raise ValueError(f"x and y disagree: {len(new_x)} vs {len(new_y)}")
        if len(new_x) == 0:
            raise ValueError("adapt() needs at least one sample")

        if not hasattr(self, "_replay_buffer") or self._replay_buffer is None:
            self._replay_buffer = ReplayBuffer(capacity=replay_capacity, seed=seed)
        if not hasattr(self, "adapt_history"):
            self.adapt_history = []

        model = self.model.to(device)

        def _acc(x: torch.Tensor, y: torch.Tensor) -> float:
            was_training = model.training
            model.eval()
            correct = 0
            with torch.inference_mode():
                for i in range(0, len(x), 1024):
                    out = model(x[i:i + 1024].to(device))
                    correct += (out.argmax(dim=1) == y[i:i + 1024].to(device)).sum().item()
            model.train(was_training)
            return correct / len(x)

        acc_before = _acc(new_x, new_y)

        # Draw the replay sample BEFORE ingesting the new batch, so it is a
        # sample of the *past* — that is what retention is measured against.
        old_x = old_y = None
        if replay and len(self._replay_buffer) > 0:
            old_x, old_y = self._replay_buffer.sample(len(new_x))
        retention_before = _acc(old_x, old_y) if old_x is not None else None

        n_reborn = 0
        if plasticity and reinit_fraction > 0:
            from dnaty.training.plasticity import NeuronUtilityTracker, reinit_dormant
            try:
                tracker = NeuronUtilityTracker(model)
                was_training = model.training
                model.eval()
                try:
                    for i in range(0, min(len(new_x), 1024), 256):
                        tracker.update(new_x[i:i + 256].to(device))
                finally:
                    model.train(was_training)
                n_reborn = reinit_dormant(
                    model, tracker.dormant_mask(reinit_fraction), seed=seed
                )
            except TypeError:
                pass  # not a DynamicMLP (e.g. quantized/custom) — skip reinit

        # Fine-tune on new data mixed 1:1 with replayed old data.
        if old_x is not None:
            train_x = torch.cat([new_x, old_x])
            train_y = torch.cat([new_y, old_y])
        else:
            train_x, train_y = new_x, new_y

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        for _ in range(epochs):
            perm = torch.randperm(len(train_x), generator=gen)
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                xb = train_x[idx].to(device)
                yb = train_y[idx].to(device)
                optimizer.zero_grad(set_to_none=True)
                with _bn_eval_if_tiny(model, xb.size(0)):
                    loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
        model.eval()

        self._replay_buffer.add_batch(new_x, new_y)

        metrics = {
            "acc_before": round(acc_before, 4),
            "acc_after": round(_acc(new_x, new_y), 4),
            "retention_before": (
                None if retention_before is None else round(retention_before, 4)
            ),
            "retention_after": (
                None if old_x is None else round(_acc(old_x, old_y), 4)
            ),
            "n_reborn": n_reborn,
            "replay_size": len(self._replay_buffer),
        }
        self.adapt_history.append(metrics)
        return metrics

    def export_onnx(self, path: str, input_shape: tuple) -> None:
        """Export the compressed model to ONNX for CPU deployment (drones, cameras, robots).

        When N:M sparsity was applied via compress(..., sparsity="2:4"), the sparsity
        mask metadata is embedded in the ONNX file so edge runtimes and loaders can
        reconstruct the pattern without re-applying it.

        Args:
            path:        Output file path, e.g. "model.onnx".
            input_shape: Shape of a single input sample, e.g. (784,) for MNIST or (3072,) for CIFAR-10.
                         Do NOT include the batch dimension.

        Example:
            result.export_onnx("model.onnx", input_shape=(784,))
        """
        import json
        dummy = torch.zeros(1, *input_shape)
        self.model.eval()
        kwargs = dict(
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17,
            do_constant_folding=True,
        )
        from dnaty.utils.latency_bench import ONNX_EXPORT_LOCK
        try:
            # torch >= 2.6 defaults to the dynamo exporter, which fails on
            # DynamicMLP's data-dependent skip-connection loop. Force the
            # stable TorchScript exporter.
            with ONNX_EXPORT_LOCK:
                torch.onnx.export(self.model, dummy, path, dynamo=False, **kwargs)
        except TypeError:
            # torch < 2.5 has no `dynamo` kwarg -- TorchScript is already the default.
            with ONNX_EXPORT_LOCK:
                torch.onnx.export(self.model, dummy, path, **kwargs)

        # Embed sparsity metadata if the model has sparse weights
        try:
            import onnx
            from dnaty.utils.sparsity import sparsity_stats
            stats = sparsity_stats(self.model)
            if stats["global_sparsity_pct"] > 0.5:  # only if meaningful sparsity
                model_proto = onnx.load(path)
                _set_onnx_meta(model_proto, "dnaty_sparsity", json.dumps(stats))
                _set_onnx_meta(model_proto, "dnaty_version", __version__)
                _set_onnx_meta(model_proto, "dnaty_arch",    json.dumps(self.arch))
                _set_onnx_meta(model_proto, "dnaty_accuracy", str(round(self.accuracy, 6)))
                onnx.save(model_proto, path)
        except ImportError:
            pass  # onnx not installed; export is still valid, just no metadata

    def push_to_hub(
        self,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        input_shape: Optional[tuple] = None,
        commit_message: str = "Upload compressed model via dNATY",
    ) -> str:
        """Push the compressed model to HuggingFace Hub.

        Uploads model_compressed.pt, model.onnx (if input_shape is provided),
        and a structured model card with compression statistics.

        Args:
            repo_id:        HuggingFace repo ID, e.g. "username/my-model-compressed".
            token:          HF API token. Defaults to HF_TOKEN env var if not set.
            private:        Create a private repo (default False).
            input_shape:    Input shape for ONNX export, e.g. (784,). Skips ONNX if None.
            commit_message: Commit message for the Hub upload.

        Returns:
            URL of the uploaded model on HuggingFace Hub.

        Requires:
            pip install huggingface-hub

        Example:
            result.push_to_hub("myuser/mnist-compressed", input_shape=(784,))
            # → "https://huggingface.co/myuser/mnist-compressed"
        """
        import os
        import tempfile
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "huggingface_hub is required: pip install huggingface-hub"
            )

        api = HfApi(token=token)
        model_name = repo_id.split("/")[-1]

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Save compressed model
            pt_path = os.path.join(tmpdir, "model_compressed.pt")
            self.save(pt_path)

            # 2. Export ONNX if input_shape provided
            onnx_note = ""
            if input_shape is not None:
                onnx_path = os.path.join(tmpdir, "model.onnx")
                try:
                    self.export_onnx(onnx_path, input_shape=input_shape)
                    onnx_note = (
                        f"\n```python\n# ONNX export (already included in this repo)\n"
                        f"result.export_onnx('model.onnx', input_shape={input_shape})\n```\n"
                    )
                except Exception:
                    pass

            # 3. Write model card
            sparsity_row = ""
            try:
                from dnaty.utils.sparsity import sparsity_stats
                stats = sparsity_stats(self.model)
                if stats["global_sparsity_pct"] > 0.5:
                    sparsity_row = (
                        f"| Sparsity (N:M) | {stats['global_sparsity_pct']:.1f}% "
                        f"({stats['zero_weights']:,} / {stats['total_weights']:,} weights zeroed) |\n"
                    )
            except Exception:
                pass

            card = f"""---
language:
- en
library_name: dnaty
tags:
- dnaty
- compression
- neural-architecture-search
- edge-ml
- pytorch
---

# {model_name}

Compressed with **[dNATY](https://github.com/pedrovergueiroo/dNATY)** — Evolutionary Neural Architecture Search for CPU-first edge deployment.

## Compression Summary

```
{self.summary()}
```

## Metrics

| Metric | Value |
|--------|-------|
| FLOPs reduction | {self.flops_reduction_pct:.1f}% ({self.original_flops:,} → {self.compressed_flops:,}) |
| Params reduction | {self.params_reduction_pct:.1f}% ({self.original_params:,} → {self.compressed_params:,}) |
| Accuracy | {self.accuracy:.4f} |
| Architecture (hidden) | {self.arch} |
| NAS generations | {self.generations} |
{sparsity_row}
## Usage

```python
import dnaty

result = dnaty.load("model_compressed.pt")
print(result.summary())
result.export_onnx("model.onnx", input_shape={input_shape or (784,)})

# Measure real latency
lat = result.benchmark_latency({input_shape or (784,)})
print(f"p50 latency: {{lat['p50_ms']:.2f}} ms  |  {{lat['fps']:.0f}} FPS")
```
{onnx_note}
## About dNATY

dNATY uses evolutionary NAS with episodic memory to find compact, fast architectures
for CPU-only edge devices (Raspberry Pi, drones, cameras, robots).
No GPU required — no retraining from scratch.
"""
            card_path = os.path.join(tmpdir, "README.md")
            with open(card_path, "w", encoding="utf-8") as f:
                f.write(card)

            # 4. Push to Hub
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                commit_message=commit_message,
            )

        return f"https://huggingface.co/{repo_id}"


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
    # Rebuild skip-connection structure (absent in files saved before v2.0.3)
    # so load_state_dict finds matching skip_projs.* keys.
    import torch.nn as _nn
    for in_f, out_f in payload.get("skip_proj_dims", []):
        model.skip_projs.append(_nn.Linear(in_f, out_f, bias=False))
    model.skip_connections = [
        (src, dst, proj_idx)
        for src, dst, proj_idx in payload.get("skip_connections", [])
    ]
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
