"""
dNATY — Dynamic Neuro-Adaptive sYstem.

Evolutionary Neural Architecture Search with episodic memory.
Finds compact, efficient models via guided evolution — not random search.

Quick start:
    from dnaty import compress
    from dnaty.experiments.fast_dataset import FastDataset

    ds = FastDataset("MNIST", device="cpu", train_subset=10_000)
    result = compress(your_model, ds, target_flops=0.5)
    print(result.summary())
    result.save("compressed.pt")

    # Reload
    result = dnaty.load("compressed.pt")

    # Edge deployment
    result.export_onnx("model.onnx", input_shape=(784,))
    print(result.benchmark_latency((784,)))  # p50/p95/fps

    # CNN compression
    from dnaty import compress_cnn
    result = compress_cnn(cnn_model, cifar_loader, target_flops=0.5)

    # Production monitoring
    from dnaty.monitoring import DriftDetector, ProductionTracker
    detector = DriftDetector().fit(train_x)
    tracker = ProductionTracker(result.model, drift_detector=detector)
    preds, meta = tracker.predict(new_batch)
    if meta["alert"]:
        print(meta["alert"])

    # Accurate FLOPs counting per layer
    from dnaty.utils.flops_counter import count_flops, flops_by_layer
    print(f"Total FLOPs: {count_flops(model, input_shape=(784,)):,}")
"""

__version__ = "1.1.6"

from dnaty.compress import compress, compress_cnn, compress_with_backbone, prune_conv_channels
from dnaty.result import CompressResult, load
from dnaty.evolution.evolver import DnatyEvolver, CnnEvolver
from dnaty.monitoring import DriftDetector, ProductionTracker
from dnaty.utils.flops_counter import count_flops, flops_by_layer

__all__ = [
    # Core API
    "compress",
    "compress_cnn",
    "compress_with_backbone",
    "prune_conv_channels",
    "load",
    "CompressResult",
    # Evolvers
    "DnatyEvolver",
    "CnnEvolver",
    # Monitoring
    "DriftDetector",
    "ProductionTracker",
    # FLOPs
    "count_flops",
    "flops_by_layer",
    # Meta
    "__version__",
]
