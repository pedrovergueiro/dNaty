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
"""

__version__ = "1.0.1"

from dnaty.compress import compress, CompressResult
from dnaty.evolution.evolver import DnatyEvolver

__all__ = ["compress", "CompressResult", "DnatyEvolver", "__version__"]
