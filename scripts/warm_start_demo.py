#!/usr/bin/env python3
"""
dNATY warm_start_demo.py -- measures transferable-memory speedup (v2.1.0).

The idea: dNATY's episodic memory learns *which structural mutations help* on a
task. In v2.1.0 that operator prior can be exported and used to warm-start the
search on a *related* task. If the prior transfers, the warm-started run should
reach a target accuracy in fewer generations than a cold start.

This script measures exactly that on two related tasks:
  - Task A: train, then export the learned operator prior.
  - Task B: run the search twice -- cold vs warm-started with Task A's prior --
            and report generations-to-target for each.

Related tasks (default = MNIST -> FashionMNIST, both 28x28 -> 10 classes; falls
back to two synthetic tabular tasks if torchvision datasets are unavailable).

Usage:
  python scripts/warm_start_demo.py            # ~6-10 min CPU
  python scripts/warm_start_demo.py --quick    # ~2-4 min CPU
  python scripts/warm_start_demo.py --synthetic  # no download, ~1 min

Honest scope: this demonstrates operator-prior transfer between *related* MLP
tasks. It is not a claim about ImageNet-scale conv search. Numbers vary by seed
and hardware; the script prints exactly what it measured.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

import dnaty
from dnaty import compress


def _gens_to_target(progress_logs, target_acc):
    """First generation whose best_acc >= target (else None)."""
    for log in progress_logs:
        if log.best_acc >= target_acc:
            return log.gen
    return None


def _run(model_fn, data, *, warm_start=None, seed=0, n_gen=20, n_pop=12):
    logs = []
    r = compress(
        model_fn(), data,
        target_flops=0.5, n_generations=n_gen, n_pop=n_pop, seed=seed,
        finetune_epochs=0, verbose=False,
        warm_start=warm_start,
        progress_callback=logs.append,
    )
    return r, logs


def _load_vision(name, subset):
    from dnaty.experiments.fast_dataset import FastDataset
    ds = FastDataset(name, device="cpu", train_subset=subset)
    return ds, 784, 10


def _synthetic(seed, d=40, n=2000, n_classes=4):
    rng = np.random.RandomState(seed)
    W = rng.randn(d, n_classes)
    X = rng.randn(n, d).astype("float32")
    logits = X @ W + rng.randn(n, n_classes) * 0.5
    y = logits.argmax(1).astype("int64")
    return (X, y), d, n_classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer gens (~2-4 min)")
    ap.add_argument("--synthetic", action="store_true", help="no dataset download")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warm-weight", type=float, default=2.0)
    args = ap.parse_args()

    n_gen = 12 if args.quick else 25
    n_pop = 10 if args.quick else 14
    subset = 6000 if args.quick else 12000

    use_vision = not args.synthetic
    if use_vision:
        try:
            task_a, in_a, nc_a = _load_vision("MNIST", subset)
            task_b, in_b, nc_b = _load_vision("FashionMNIST", subset)
        except Exception as e:  # noqa: BLE001
            print(f"[warm_start_demo] vision datasets unavailable ({e}); "
                  f"falling back to --synthetic")
            use_vision = False

    if not use_vision:
        task_a, in_a, nc_a = _synthetic(seed=1)
        task_b, in_b, nc_b = _synthetic(seed=2)
        in_b, nc_b = in_a, nc_a  # same shape -> prior is transferable

    def model_a():
        return nn.Sequential(nn.Flatten(), nn.Linear(in_a, 256), nn.ReLU(),
                             nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, nc_a))

    def model_b():
        return nn.Sequential(nn.Flatten(), nn.Linear(in_b, 256), nn.ReLU(),
                             nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, nc_b))

    print("=" * 66)
    print(f"dNATY {dnaty.__version__} - transferable-memory warm-start demo")
    print(f"  mode={'vision (MNIST->FashionMNIST)' if use_vision else 'synthetic'}"
          f"  n_gen={n_gen} n_pop={n_pop}  warm_weight={args.warm_weight}")
    print("=" * 66)

    # 1. Learn a prior on Task A.
    t0 = time.time()
    r_a, _ = _run(model_a, task_a, seed=args.seed, n_gen=n_gen, n_pop=n_pop)
    prior = r_a.export_memory()
    top = sorted(prior.get("scores", {}).items(), key=lambda kv: -kv[1])[:3]
    print(f"\n[Task A] learned prior in {time.time()-t0:.0f}s. "
          f"Top operators: {[k for k, _ in top]}")

    # 2. Task B: cold vs warm.
    r_cold, logs_cold = _run(model_b, task_b, warm_start=None,
                             seed=args.seed, n_gen=n_gen, n_pop=n_pop)
    r_warm, logs_warm = _run(model_b, task_b, warm_start=prior,
                             seed=args.seed, n_gen=n_gen, n_pop=n_pop)

    # Target = 98% of the best cold accuracy (a level both runs can plausibly hit).
    best_cold = max(l.best_acc for l in logs_cold)
    target = 0.98 * best_cold
    g_cold = _gens_to_target(logs_cold, target)
    g_warm = _gens_to_target(logs_warm, target)

    print(f"\n[Task B] target acc = {target:.4f} (98% of cold-best {best_cold:.4f})")
    print(f"  cold  : reached at gen {g_cold}   (final best {max(l.best_acc for l in logs_cold):.4f})")
    print(f"  warm  : reached at gen {g_warm}   (final best {max(l.best_acc for l in logs_warm):.4f})")

    if g_cold and g_warm:
        speedup = g_cold / g_warm
        saved = g_cold - g_warm
        print(f"\n  -> warm-start reached the target {saved:+d} generations "
              f"{'sooner' if saved > 0 else 'later'}  (x{speedup:.2f})")
    print(f"\n  Pareto front (cold): {len(r_cold.pareto_front)} non-dominated architectures")
    print("=" * 66)
    print("Note: single-seed run; average over seeds for a paper-grade number.")


if __name__ == "__main__":
    main()
