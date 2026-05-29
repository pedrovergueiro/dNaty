"""
Visualiza resultados de CPU latency benchmark
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
with open("results_cpu_latency/cpu_latency_benchmark.json") as f:
    data = json.load(f)

results = data['results']
names = list(results.keys())
params = [results[n]['params']/1e6 for n in names]
latency_b1 = [results[n]['batch_sizes']['1']['latency_ms'] for n in names]
batch_sizes_str = list(results[names[0]]['batch_sizes'].keys())
throughput_max = [max(results[n]['batch_sizes'][bs]['throughput_imgs_per_sec'] for bs in batch_sizes_str) for n in names]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

# Plot 1: Latency vs Parameters
axes[0].scatter(params, latency_b1, s=600, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
for i, name in enumerate(names):
    axes[0].annotate(name, (params[i], latency_b1[i]), xytext=(8, 8), textcoords='offset points', fontsize=10, fontweight='bold')
axes[0].set_xlabel('Parameters (Millions)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Latency per Image (ms)', fontsize=11, fontweight='bold')
axes[0].set_title('Latency vs Model Size', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Plot 2: Latency comparison (bar chart)
axes[1].barh(names, latency_b1, color=colors, edgecolor='black', linewidth=1.5)
for i, (name, val) in enumerate(zip(names, latency_b1)):
    axes[1].text(val + 1, i, f'{val:.1f}ms', va='center', fontweight='bold')
axes[1].set_xlabel('Latency (ms) - Batch=1', fontsize=11, fontweight='bold')
axes[1].set_title('CPU Inference Latency', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')

# Plot 3: Throughput comparison
axes[2].barh(names, throughput_max, color=colors, edgecolor='black', linewidth=1.5)
for i, (name, val) in enumerate(zip(names, throughput_max)):
    axes[2].text(val + 2, i, f'{val:.0f}', va='center', fontweight='bold')
axes[2].set_xlabel('Max Throughput (images/sec)', fontsize=11, fontweight='bold')
axes[2].set_title('CPU Throughput Performance', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('results_cpu_latency/cpu_latency_comparison.png', dpi=150, bbox_inches='tight')
print("[OK] Gráfico salvo em: results_cpu_latency/cpu_latency_comparison.png")
plt.show()

# Print speedup
print("\n" + "="*70)
print("SPEEDUP COMPARADO A RESNET-50")
print("="*70)
resnet_latency = latency_b1[0]
for i, name in enumerate(names[1:], 1):
    speedup = resnet_latency / latency_b1[i]
    print(f"{name:20} | {speedup:.1f}x mais rápido")
