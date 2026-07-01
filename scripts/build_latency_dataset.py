"""
Generate synthetic (architecture → latency) dataset and train the GBM surrogate.

Samples N random MLP architectures, measures real ONNX Runtime latency on
the current CPU, then trains LatencyPredictor and saves it to
results/latency_predictor.json + results/latency_predictor.lgb.

Run: python scripts/build_latency_dataset.py
     python scripts/build_latency_dataset.py --n 2000 --input-size 784
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty.utils.latency_bench import measure_latency
from dnaty.utils.flops_counter import count_flops

RESULTS = Path(__file__).parent.parent / "results"


def random_mlp(input_size: int, n_classes: int, rng: np.random.Generator) -> nn.Module:
    n_layers = int(rng.integers(1, 6))
    first = int(rng.choice([64, 128, 256, 512, 1024]))
    widths = [first]
    for _ in range(n_layers - 1):
        w = int(rng.integers(max(16, widths[-1] // 4), widths[-1] + 1))
        widths.append(w)
    layers, prev = [], input_size
    for w in widths:
        layers += [nn.Linear(prev, w), nn.ReLU()]
        prev = w
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


def arch_features(model: nn.Module, input_size: int) -> dict:
    widths = [m.out_features for m in model.modules() if isinstance(m, nn.Linear)][:-1]
    total_params = sum(p.numel() for p in model.parameters())
    total_flops = count_flops(model, input_shape=(input_size,))
    return {
        "n_layers": len(widths),
        "widths": widths,
        "total_params": total_params,
        "total_flops": total_flops,
        "max_width": max(widths) if widths else 0,
        "min_width": min(widths) if widths else 0,
        "mean_width": sum(widths) / len(widths) if widths else 0,
        "input_size": input_size,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000, help="Number of architectures to sample")
    parser.add_argument("--input-size", type=int, default=784, help="Input feature size")
    parser.add_argument("--n-classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    records = []
    failed = 0

    print(f"Generating {args.n} architecture latency measurements...")
    print(f"Input size: {args.input_size}  Classes: {args.n_classes}")
    print()

    t0 = time.time()
    for i in range(args.n):
        model = random_mlp(args.input_size, args.n_classes, rng)
        try:
            lat = measure_latency(model, input_shape=(args.input_size,),
                                  n_warmup=10, n_runs=50)
            feats = arch_features(model, args.input_size)
            records.append({"features": feats, "latency_ms": lat["p50_ms"]})
        except Exception as e:
            failed += 1
            continue

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (args.n - i - 1)
            print(f"  [{i+1}/{args.n}]  ok={len(records)}  failed={failed}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    print(f"\nDone: {len(records)} valid / {args.n} total  ({failed} failed)")

    # Save raw dataset
    RESULTS.mkdir(exist_ok=True)
    raw_path = RESULTS / "latency_dataset.json"
    raw_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Raw dataset: {raw_path}")

    # Train GBM
    try:
        from dnaty.utils.latency_predictor import LatencyPredictor
        pred = LatencyPredictor()
        print("\nTraining GBM surrogate...")
        stats = pred.train(records, save_path=RESULTS / "latency_predictor.json")
        print(f"  MAPE: {stats['mape_pct']:.2f}%  R²: {stats['r2']:.4f}")
        print(f"  Train: {stats['n_train']}  Val: {stats['n_val']}")
        print(f"  Saved: {RESULTS / 'latency_predictor.json'}")
    except ImportError:
        print("\nlightgbm not installed — skipping GBM training.")
        print("  Install: pip install lightgbm")
        print("  Then re-run this script.")


if __name__ == "__main__":
    main()
