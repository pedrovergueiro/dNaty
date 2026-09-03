"""
dNATY -- Dynamic Neuro-Adaptive sYstem.

Evolutionary Neural Architecture Search with episodic memory.
Finds compact, efficient models via guided evolution -- not random search.

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

    # Full accuracy/FLOPs trade-off curve (v2.1.0) — pick per device budget
    print(result.pareto_summary())
    result.pareto_front_csv("front.csv")

    # Transferable memory (v2.1.0) — warm-start a related task from a prior run
    result.save_memory("prior.json")
    result2 = compress(other_model, other_data, warm_start="prior.json")

    # Meta-learned search controller (v2.2.0) — learns *how* to search
    result = compress(model, ds, controller=True)
    result2 = compress(other_model, other_data, controller=True,
                       controller_policy=result.controller_policy)

    # Lifelong on-device adaptation (v2.2.0) — drifted data, no recompress
    metrics = result.adapt(new_x, new_y)   # replay + plasticity built in
    print(metrics["acc_after"], metrics["retention_after"])

    # Federated prior merging (v2.2.0, research preview)
    from dnaty import merge_priors
    consensus = merge_priors([prior_device_a, prior_device_b])
    result = compress(model, ds, warm_start=consensus)
"""

__version__ = "2.2.0"

from dnaty.compress import compress, compress_cnn, compress_with_backbone, prune_conv_channels
from dnaty.result import CompressResult, load
from dnaty.core.memory import save_prior, load_prior, merge_priors
from dnaty.evolution.evolver import DnatyEvolver, CnnEvolver, LatencyEvolver, QuantAwareEvolver
from dnaty.monitoring import DriftDetector, ProductionTracker
from dnaty.utils.flops_counter import count_flops, flops_by_layer
from dnaty.utils.latency_bench import measure_latency
from dnaty.utils.hw_detect import detect_hw, latency_scale, estimate_latency
from dnaty.utils.latency_predictor import LatencyPredictor
from dnaty.utils.latency_tables import lookup_linear_latency, estimate_mlp_latency
from dnaty.utils.sparsity import apply_nm_sparsity, sparsity_stats
from dnaty.utils.proxies import ProxyEnsemble, score_candidate
from dnaty.evolution.controller import MetaController
from dnaty.training.replay import ReplayBuffer
from dnaty.training.plasticity import (
    PlasticityController,
    NeuronUtilityTracker,
    reinit_dormant,
    effective_rank,
)

__all__ = [
    # Core API
    "compress",
    "compress_cnn",
    "compress_with_backbone",
    "prune_conv_channels",
    "load",
    "CompressResult",
    # Transferable operator priors (v2.1.0)
    "save_prior",
    "load_prior",
    # Federated prior merging (v2.2.0, research preview)
    "merge_priors",
    # Meta-learned search controller (v2.2.0)
    "MetaController",
    # Lifelong adaptation (v2.2.0)
    "ReplayBuffer",
    "PlasticityController",
    "NeuronUtilityTracker",
    "reinit_dormant",
    "effective_rank",
    # Evolvers
    "DnatyEvolver",
    "CnnEvolver",
    "LatencyEvolver",
    "QuantAwareEvolver",
    # Monitoring
    "DriftDetector",
    "ProductionTracker",
    # Latency / hardware
    "measure_latency",
    "detect_hw",
    "latency_scale",
    "estimate_latency",
    "LatencyPredictor",
    "lookup_linear_latency",
    "estimate_mlp_latency",
    # Zero-cost proxies
    "ProxyEnsemble",
    "score_candidate",
    # Sparsity
    "apply_nm_sparsity",
    "sparsity_stats",
    # FLOPs
    "count_flops",
    "flops_by_layer",
    # Meta
    "__version__",
]
