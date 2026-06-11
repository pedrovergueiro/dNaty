#!/usr/bin/env python
"""Run only the e-commerce benchmark with corrected multiclass labels."""
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from scripts.benchmark_market_real import benchmark_synthetic

result = benchmark_synthetic(
    name='E-commerce Purchase Propensity',
    domain='retail / online commerce',
    n_samples=80_000,
    n_features=48,
    n_classes=4,
    target_flops=0.45,
    n_gen=15,
    n_pop=12,
)

print(f"\n{'='*70}")
print("E-COMMERCE BENCHMARK RESULT (CORRECTED)")
print('='*70)
print(f"Accuracy:        {result['accuracy']:.4f}")
print(f"FLOPs reduction: {result['flops_reduction']:.1f}%")
print(f"Search time:     {result['time_seconds']/60:.2f} min")
print('='*70)

# Save to JSON
results_file = Path(__file__).parent / "results" / "benchmark_ecommerce_fixed.json"
with open(results_file, 'w') as f:
    json.dump([result], f, indent=2)
print(f"\nSaved to {results_file.name}")
