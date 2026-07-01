"""
ONNX Runtime latency microbenchmark.

Measures real inference latency for a PyTorch model by exporting to ONNX
and running warm-up + timed passes through OrtSession.

Returns p50/p95/fps for a given input shape.
"""
from __future__ import annotations

import io
import time
import warnings
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


def measure_latency(
    model: nn.Module,
    input_shape: tuple,
    n_warmup: int = 20,
    n_runs: int = 100,
    batch_size: int = 1,
) -> dict[str, float]:
    """
    Export model to ONNX and measure inference latency via OrtSession.

    Args:
        model: PyTorch model (eval mode, CPU)
        input_shape: single sample shape, e.g. (784,) or (3, 32, 32)
        n_warmup: warm-up passes (not measured)
        n_runs: timed passes
        batch_size: batch size for measurement

    Returns:
        {"p50_ms": float, "p95_ms": float, "fps": float, "mean_ms": float}
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnxruntime required: pip install onnxruntime")

    model = model.eval().cpu()

    # Export to in-memory ONNX buffer
    # Redirect stdout during export: PyTorch 2.x prints emoji (✅) that crash on Windows cp1252
    import contextlib
    dummy = torch.zeros(batch_size, *input_shape)
    buf = io.BytesIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy,
            buf,
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    buf.seek(0)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # single-thread → reproducible latency
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(buf.read(), sess_options=opts, providers=["CPUExecutionProvider"])

    feed = {"input": dummy.numpy()}

    # Warm-up
    for _ in range(n_warmup):
        sess.run(None, feed)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    times = sorted(times)
    p50 = float(np.percentile(times, 50))
    p95 = float(np.percentile(times, 95))
    mean = float(np.mean(times))
    fps = 1000.0 / mean if mean > 0 else 0.0

    return {"p50_ms": round(p50, 3), "p95_ms": round(p95, 3),
            "mean_ms": round(mean, 3), "fps": round(fps, 1)}
