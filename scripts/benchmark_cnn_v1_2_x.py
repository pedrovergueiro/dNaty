"""
dNATY v1.2.x CNN Benchmark — real training results.

Measures CnnEvolver on CIFAR-10 with the new FLOPs-guided swap and budget boost.
Compares: (A) no boost  vs  (B) with budget_boost_factor=3.0

Run: python scripts/benchmark_cnn_v1_2_x.py
Results: results/benchmark_cnn_v1_2_x.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty.evolution.evolver import CnnEvolver
from dnaty.experiments.fast_dataset import FastDataset
from dnaty.utils.flops_counter import count_flops

RESULTS_FILE = Path(__file__).parent.parent / "results" / "benchmark_cnn_v1_2_x.json"

W = 60
def banner(t): print(f"\n{'='*W}\n  {t}\n{'='*W}")
def info(t):   print(f"  {t}")


def run_cnn_evolver(label, n_gen, n_pop, target_flops, budget_boost_factor, train_loader, val_loader):
    banner(f"{label}")
    evolver = CnnEvolver(
        n_pop=n_pop,
        n_generations=n_gen,
        t_local=2,
        lr=5e-4,
        target_flops=target_flops,
        budget_boost_factor=budget_boost_factor,
        n_classes=10,
        verbose=True,
    )
    t0 = time.time()
    best, history = evolver.run(train_loader, val_loader)
    elapsed = time.time() - t0

    baseline_flops = evolver._baseline_flops or 1
    best_flops = best.count_flops()
    flops_red = 100 * (1 - best_flops / baseline_flops)

    info(f"")
    info(f"Result — {label}")
    info(f"  Baseline FLOPs : {baseline_flops:>12,.0f}")
    info(f"  Best FLOPs     : {best_flops:>12,.0f}")
    info(f"  FLOPs reduction: {flops_red:+.1f}%")
    info(f"  Val accuracy   : {best.acc:.4f}")
    info(f"  Time           : {elapsed/60:.1f} min")

    # Op counts across all generations (improvements recorded in memory)
    all_op_counts: dict[str, int] = {}
    for h in history:
        for op, cnt in h.op_counts.items():
            all_op_counts[op] = all_op_counts.get(op, 0) + cnt

    info(f"  Operator improvements: {dict(sorted(all_op_counts.items(), key=lambda x: -x[1])[:5])}")

    return {
        "label": label,
        "n_gen": n_gen,
        "n_pop": n_pop,
        "target_flops": target_flops,
        "budget_boost_factor": budget_boost_factor,
        "baseline_flops": int(baseline_flops),
        "best_flops": int(best_flops),
        "flops_reduction_pct": round(flops_red, 2),
        "val_accuracy": round(best.acc, 4),
        "time_minutes": round(elapsed / 60, 1),
        "op_counts": all_op_counts,
        "acc_history": [round(h.best_acc, 4) for h in history],
    }


def main():
    banner("dNATY v1.2.x — CNN Real Benchmark (CIFAR-10)")

    N_GEN   = 20
    N_POP   = 10
    SUBSET  = 8000

    info(f"Dataset: CIFAR-10 (train={SUBSET}, val=10K)")
    info(f"Config:  n_gen={N_GEN}, n_pop={N_POP}, t_local=2")
    info(f"Device:  {'cuda' if torch.cuda.is_available() else 'cpu'}")

    ds = FastDataset("CIFAR10", device="cpu", train_subset=SUBSET)
    train_loader = ds.get_train_loader_compat(batch_size=128)
    val_loader   = ds.get_train_loader_compat(batch_size=256)

    # Baseline model FLOPs (default DynamicCNN architecture)
    from dnaty.core.arch_cnn import DynamicCNN
    baseline_model = DynamicCNN(n_classes=10)
    baseline_flops = count_flops(baseline_model, input_shape=(3, 32, 32))
    info(f"\nDefault DynamicCNN baseline FLOPs: {baseline_flops:,}")

    results = {
        "dataset": "CIFAR-10",
        "train_subset": SUBSET,
        "n_gen": N_GEN,
        "n_pop": N_POP,
        "baseline_model_flops": int(baseline_flops),
    }

    # Run A: no budget boost (budget_boost_factor=1.0 = no effect)
    r_a = run_cnn_evolver(
        "A — no budget boost (baseline)",
        N_GEN, N_POP, target_flops=0.5,
        budget_boost_factor=1.0,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    results["no_boost"] = r_a

    # Run B: with budget boost x3
    r_b = run_cnn_evolver(
        "B — budget boost x3 (v1.2.x)",
        N_GEN, N_POP, target_flops=0.5,
        budget_boost_factor=3.0,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    results["with_boost"] = r_b

    # Summary
    banner("SUMMARY")
    for label, r in [("A — no boost", r_a), ("B — boost x3", r_b)]:
        info(f"{label}:")
        info(f"  acc={r['val_accuracy']:.4f}  FLOPs {r['flops_reduction_pct']:+.1f}%  ({r['time_minutes']:.1f} min)")

    flops_diff = r_b["flops_reduction_pct"] - r_a["flops_reduction_pct"]
    info(f"\nBudget boost effect: {flops_diff:+.1f}pp more FLOPs reduction")

    results["boost_delta_flops_pp"] = round(flops_diff, 2)

    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    info(f"\nSaved: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
