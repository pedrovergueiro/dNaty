"""
Validation script for dNATY v1.2.x features — real data, real results.

Tests:
  1. swap_conv_to_dw FLOPs-guided selection vs random baseline (CIFAR-10 CNN)
  2. budget_boost in CnnEvolver — short evolution, FLOPs trajectory (CIFAR-10)
  3. auto_retrigger — MNIST MLP + synthetic distribution shift
  4. export_failure_report — MNIST MLP failures + PCA export

Run: python scripts/validate_v1_2_x.py
Results saved to: results/benchmark_v1_2_x.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty.core.arch_cnn import DynamicCNN
from dnaty.core.individual import Individual
from dnaty.core.memory import EpisodicMemory
from dnaty.experiments.fast_dataset import FastDataset
from dnaty.monitoring.drift import DriftDetector
from dnaty.monitoring.tracker import ProductionTracker
from dnaty.operators.mutations_cnn import swap_conv_to_dw
from dnaty.utils.flops_counter import count_flops, flops_by_layer

RESULTS_FILE = Path(__file__).parent.parent / "results" / "benchmark_v1_2_x.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def _flops_by_layer_summary(model, input_shape):
    detail = flops_by_layer(model, input_shape=input_shape)
    return {k: v for k, v in detail.items() if v > 0}


# ---------------------------------------------------------------------------
# 1. swap_conv_to_dw: FLOPs-guided vs random
# ---------------------------------------------------------------------------

def validate_flops_guided_swap():
    _section("Item 4 — swap_conv_to_dw: FLOPs-guided vs random")

    # CNN with 3 conv layers of deliberately different sizes
    # Layer 0: 3→16  (in*out = 48)
    # Layer 1: 16→64 (in*out = 1024)   <-- highest, should always be picked
    # Layer 2: 64→32 (in*out = 2048)   <-- actually highest! 64*32=2048 vs 16*64=1024
    # Correction: Layer 2 is 64*32=2048, Layer 1 is 16*64=1024, Layer 0 is 3*16=48
    # So Layer 2 should always be picked.
    configs = [
        {"type": "conv", "in_ch": 3,  "out_ch": 16, "stride": 1},
        {"type": "conv", "in_ch": 16, "out_ch": 64, "stride": 1},
        {"type": "conv", "in_ch": 64, "out_ch": 32, "stride": 1},
    ]
    model = DynamicCNN(configs, fc_sizes=[128], n_classes=10, in_channels=3)
    ind = Individual(model, EpisodicMemory())

    # Compute FLOPs per layer before swap
    input_shape = (3, 32, 32)
    layer_flops_before = _flops_by_layer_summary(model, input_shape)
    total_before = count_flops(model, input_shape=input_shape)

    print(f"\nBaseline model — total FLOPs: {total_before:,}")
    for name, flops in layer_flops_before.items():
        print(f"  {name}: {flops:,}")

    # Which layer has highest in_ch * out_ch?
    products = [(i, configs[i]["in_ch"] * configs[i]["out_ch"]) for i in range(len(configs))]
    expected_idx = max(products, key=lambda x: x[1])[0]
    print(f"\nExpected swap target: layer {expected_idx} "
          f"(in_ch={configs[expected_idx]['in_ch']} * out_ch={configs[expected_idx]['out_ch']} "
          f"= {configs[expected_idx]['in_ch']*configs[expected_idx]['out_ch']})")

    # Run swap 20 times, record which layer was picked
    picks = []
    flops_reductions = []
    for trial in range(20):
        ind_copy = Individual(
            DynamicCNN(configs, fc_sizes=[128], n_classes=10, in_channels=3),
            EpisodicMemory()
        )
        new_ind, ok = swap_conv_to_dw(ind_copy)
        if ok:
            for i, c in enumerate(new_ind.model.conv_configs):
                if c["type"] == "depthwise":
                    picks.append(i)
                    total_after = count_flops(new_ind.model, input_shape=input_shape)
                    flops_reductions.append(100 * (1 - total_after / total_before))
                    break

    always_correct = all(p == expected_idx for p in picks)
    mean_reduction = np.mean(flops_reductions) if flops_reductions else 0.0

    print(f"\nResults over 20 trials:")
    print(f"  Layer picks: {picks}")
    print(f"  Always picks expected layer {expected_idx}: {always_correct}")
    print(f"  FLOPs reduction (mean): {mean_reduction:.1f}%")

    # Compare to what random would give (uniform over eligible layers)
    eligible = [i for i, c in enumerate(configs) if c["type"] == "conv" and c["out_ch"] >= 8]
    random_flops = []
    for idx in eligible:
        new_configs = [dict(c) for c in configs]
        new_configs[idx] = {"type": "depthwise", "in_ch": configs[idx]["in_ch"],
                             "out_ch": configs[idx]["out_ch"], "stride": 1}
        m = DynamicCNN(new_configs, fc_sizes=[128], n_classes=10, in_channels=3)
        f = count_flops(m, input_shape=input_shape)
        random_flops.append(100 * (1 - f / total_before))

    random_mean = np.mean(random_flops)
    improvement_over_random = mean_reduction - random_mean

    print(f"\n  Random baseline (uniform pick): {random_mean:.1f}% reduction")
    print(f"  FLOPs-guided improvement over random: +{improvement_over_random:.1f}pp")

    return {
        "test": "flops_guided_swap",
        "n_trials": 20,
        "expected_layer": expected_idx,
        "always_correct": always_correct,
        "flops_reduction_pct": round(mean_reduction, 2),
        "random_baseline_pct": round(random_mean, 2),
        "improvement_over_random_pp": round(improvement_over_random, 2),
    }


# ---------------------------------------------------------------------------
# 2. CnnEvolver budget-aware boost — FLOPs trajectory (short run)
# ---------------------------------------------------------------------------

def validate_budget_aware_evolver():
    _section("Item 4b — CnnEvolver budget-aware boost (CIFAR-10, mechanics check)")
    from dnaty.evolution.evolver import CnnEvolver
    from dnaty.operators.mutations_cnn import CNN_OPERATORS, apply_cnn_operator

    print("Loading CIFAR-10 (subset)...")
    ds = FastDataset("CIFAR10", device="cpu", train_subset=2000)

    # Patch _mutate_population to log which ops are selected when boost is active
    selections_boosted   = []
    selections_unboosted = []
    orig_mutate = CnnEvolver._mutate_population

    def patched_mutate(self, population):
        boost_active = (
            self._baseline_flops is not None
            and population
            and max(ind.count_flops() for ind in population) > self.target_flops * self._baseline_flops
        )
        mutated = orig_mutate(self, population)
        for ind in mutated:
            op = ind.last_op
            if boost_active:
                selections_boosted.append(op)
            else:
                selections_unboosted.append(op)
        return mutated

    CnnEvolver._mutate_population = patched_mutate

    t0 = time.time()
    evolver = CnnEvolver(
        n_pop=6, n_generations=4, t_local=1, lr=1e-3,
        target_flops=0.5, budget_boost_factor=3.0, n_classes=10, verbose=False,
    )
    train_loader = ds.get_train_loader_compat(batch_size=128)
    val_loader   = ds.get_train_loader_compat(batch_size=256)
    best, history = evolver.run(train_loader, val_loader)
    CnnEvolver._mutate_population = orig_mutate  # restore
    elapsed = time.time() - t0

    baseline_flops = evolver._baseline_flops
    best_flops = best.count_flops()
    budget_exceeded = best_flops > 0.5 * (baseline_flops or 1)

    # Boost-active selection rate for compression ops
    boost_compression = sum(1 for op in selections_boosted if op in ("swap_conv_to_dw", "prune_channels", "no_op") and op != "no_op")
    boost_total       = len([op for op in selections_boosted if op != "no_op"]) or 1
    unboosted_compression = sum(1 for op in selections_unboosted if op in ("swap_conv_to_dw", "prune_channels"))
    unboosted_total   = len([op for op in selections_unboosted if op != "no_op"]) or 1

    boost_rate    = boost_compression / boost_total
    unboosted_rate = unboosted_compression / unboosted_total

    print(f"\nBaseline FLOPs: {baseline_flops:,.0f}  |  Best FLOPs: {best_flops:,.0f}")
    print(f"Budget exceeded (>{0.5:.0%}): {budget_exceeded}")
    print(f"Generations with boost active: {len(selections_boosted)//6} / {len(history)}")
    print(f"\nCompression-op selection rate:")
    print(f"  When budget exceeded (boosted): {boost_rate:.1%}  ({boost_compression}/{boost_total})")
    print(f"  When within budget (unboosted): {unboosted_rate:.1%} ({unboosted_compression}/{unboosted_total})")
    print(f"\nBest acc: {best.acc:.4f}  |  Time: {elapsed:.1f}s")

    return {
        "test": "budget_aware_evolver",
        "dataset": "CIFAR-10 (subset 2000)",
        "baseline_flops": int(baseline_flops) if baseline_flops else None,
        "best_flops": int(best_flops),
        "budget_exceeded": budget_exceeded,
        "compression_rate_when_boosted_pct": round(boost_rate * 100, 1),
        "compression_rate_when_unboosted_pct": round(unboosted_rate * 100, 1),
        "best_val_accuracy": round(best.acc, 4),
        "time_seconds": round(elapsed, 1),
    }


# ---------------------------------------------------------------------------
# 3. auto_retrigger — real compression + distribution shift
# ---------------------------------------------------------------------------

def validate_auto_retrigger():
    _section("Item 5 — auto_retrigger: MNIST MLP + distribution shift")

    print("Loading MNIST...")
    ds = FastDataset("MNIST", device="cpu", train_subset=5000)
    train_x, train_y = ds.train_x, ds.train_y
    train_np = train_x.numpy()

    # Compress a real MLP
    print("Compressing MNIST MLP...")
    import dnaty
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = dnaty.compress(
            nn.Sequential(
                nn.Linear(784, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64),  nn.ReLU(),
                nn.Linear(64, 10),
            ),
            ds.get_train_loader_compat(batch_size=256),
            n_generations=10,
            n_pop=8,
            device="cpu",
            verbose=False,
        )

    model = result.model
    flops_reduction = result.flops_reduction_pct
    print(f"  Compressed: {flops_reduction:.1f}% FLOPs reduction, {result.accuracy:.4f} acc")

    # Setup tracker: small buffer (200) so shifted batches dominate quickly
    det = DriftDetector(psi_threshold=0.15)
    det.fit(train_np[:2000])

    tracker = ProductionTracker(
        model,
        drift_detector=det,
        drift_every=100,     # check every 100 preds
        max_history=200,     # small buffer → shifted data dominates after 1-2 batches
        device="cpu",
    )

    trigger_calls = []

    def compress_again(data):
        trigger_calls.append(len(data))
        return model

    # Phase 1: one small normal batch (50 samples) — should NOT trigger
    print("\nPhase 1: 50 normal samples — no drift expected...")
    normal_preds, meta1 = tracker.predict(train_np[:50])
    t1 = tracker.auto_retrigger(compress_again, train_np[:2000], consecutive_drifts=3)
    print(f"  Consecutive drifts: {tracker._consecutive_drift_count} | Triggered: {t1}")

    # Phase 2: feed 4 batches of pure shifted data (+5.0) — buffer fills with shifted only
    print("\nPhase 2: 4 batches x 100 pure shifted samples (should fire drift each batch)...")
    shifted_x = train_np[2000:2400] + 5.0
    psi_scores = []
    for batch_i in range(4):
        batch = shifted_x[batch_i*100:(batch_i+1)*100]
        preds, meta = tracker.predict(batch)
        psi = meta.get("drift_score") or 0.0
        psi_scores.append(psi)
        print(f"  Batch {batch_i+1}: PSI={psi:.3f}  alert={meta['alert'] or 'none'}")

    consecutive_before = tracker._consecutive_drift_count
    triggered = tracker.auto_retrigger(compress_again, train_np[:2000], consecutive_drifts=3)
    print(f"\n  Consecutive drifts reached: {consecutive_before}")
    print(f"  auto_retrigger fired: {triggered}")
    print(f"  compress_fn called {len(trigger_calls)}x")

    return {
        "test": "auto_retrigger",
        "dataset": "MNIST (subset 5000)",
        "compression_flops_reduction_pct": round(flops_reduction, 2),
        "compression_accuracy": round(result.accuracy, 4),
        "psi_scores_shifted_batches": [round(p, 3) for p in psi_scores],
        "consecutive_drifts_at_trigger": int(consecutive_before),
        "triggered": triggered,
        "compress_fn_calls": len(trigger_calls),
    }


# ---------------------------------------------------------------------------
# 4. export_failure_report — real model, real failures
# ---------------------------------------------------------------------------

def validate_failure_report():
    _section("Item 6 — export_failure_report: MNIST MLP failures + PCA")

    print("Loading MNIST (val set)...")
    ds = FastDataset("MNIST", device="cpu", train_subset=5000)
    val_x, val_y = ds.val_x, ds.val_y

    # Compress a real MLP
    print("Compressing MNIST MLP...")
    import dnaty
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = dnaty.compress(
            nn.Sequential(
                nn.Linear(784, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 10),
            ),
            ds.get_train_loader_compat(batch_size=256),
            n_generations=8,
            n_pop=8,
            device="cpu",
            verbose=False,
        )

    model = result.model
    print(f"  Compressed: {result.flops_reduction_pct:.1f}% FLOPs, acc={result.accuracy:.4f}")

    det = DriftDetector()
    det.fit(ds.train_x.numpy()[:2000])
    tracker = ProductionTracker(model, drift_detector=det, device="cpu")

    # Run inference on val set in batches, collect failures
    print("\nRunning inference on val set (10,000 samples)...")
    val_np = val_x.numpy()
    val_gt = val_y.numpy()
    batch_size = 500
    all_preds = []

    for i in range(0, len(val_np), batch_size):
        xb = val_np[i:i+batch_size]
        yb = val_gt[i:i+batch_size]
        preds, _ = tracker.predict(xb)
        tracker.record_outcome(preds, yb, inputs=xb)
        all_preds.append(preds)

    all_preds_np = np.concatenate(all_preds)
    accuracy = (all_preds_np == val_gt).mean()
    n_failures = len(tracker._failure_buffer)
    failure_rate = n_failures / len(val_gt)

    print(f"  Accuracy on val: {accuracy:.4f}")
    print(f"  Failures stored: {n_failures} / {len(val_gt)} ({failure_rate:.1%})")

    # Export report
    out_path = str(Path(__file__).parent.parent / "results" / "failure_report_mnist.json")
    report = tracker.export_failure_report(out_path, n_components=2)

    # Validate PCA coordinates exist and have variance
    pca_arr = np.array(report["pca_coords"])
    pca_variance = float(pca_arr.var(axis=0).sum()) if len(pca_arr) > 0 else 0.0

    print(f"\n  Failure report saved: {out_path}")
    print(f"  Class breakdown (top 5): {dict(list(report['class_breakdown'].items())[:5])}")
    print(f"  PCA coordinates shape: {pca_arr.shape}")
    print(f"  PCA total variance: {pca_variance:.4f}")

    # Also test SQLite export
    db_path = str(Path(__file__).parent.parent / "results" / "failures_mnist.db")
    tracker.export_failure_report(out_path, db_uri=db_path, n_components=2)
    print(f"  SQLite DB written: {db_path}")

    return {
        "test": "export_failure_report",
        "dataset": "MNIST val (10,000 samples)",
        "model_accuracy": round(float(accuracy), 4),
        "n_failures": n_failures,
        "failure_rate_pct": round(failure_rate * 100, 2),
        "class_breakdown": report["class_breakdown"],
        "pca_total_variance": round(pca_variance, 4),
        "report_path": out_path,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("dNATY v1.2.x — Real-data validation")
    print(f"Torch: {torch.__version__} | Device: CPU")

    all_results = {}

    # 1. FLOPs-guided swap (fast, no training)
    r1 = validate_flops_guided_swap()
    all_results["flops_guided_swap"] = r1

    # 2. Budget-aware evolver (CIFAR-10, 5 gens)
    r2 = validate_budget_aware_evolver()
    all_results["budget_aware_evolver"] = r2

    # 3. auto_retrigger (MNIST)
    r3 = validate_auto_retrigger()
    all_results["auto_retrigger"] = r3

    # 4. export_failure_report (MNIST)
    r4 = validate_failure_report()
    all_results["failure_report"] = r4

    # Save results
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    _section("SUMMARY")
    print(f"\n1. swap_conv_to_dw always correct: {r1['always_correct']}")
    print(f"   FLOPs-guided: {r1['flops_reduction_pct']:.1f}% vs random: {r1['random_baseline_pct']:.1f}%  (+{r1['improvement_over_random_pp']:.1f}pp)")
    print(f"\n2. CnnEvolver budget boost: budget exceeded={r2['budget_exceeded']}, acc={r2['best_val_accuracy']:.4f}")
    print(f"   compression-op rate: boosted={r2['compression_rate_when_boosted_pct']:.1f}% vs unboosted={r2['compression_rate_when_unboosted_pct']:.1f}%")
    print(f"\n3. auto_retrigger: fired={r3['triggered']} after {r3['consecutive_drifts_at_trigger']} consecutive drifts")
    print(f"\n4. failure_report: {r4['n_failures']} failures ({r4['failure_rate_pct']:.1f}%) | PCA variance={r4['pca_total_variance']:.4f}")

    print(f"\nResults saved to: {RESULTS_FILE}")
