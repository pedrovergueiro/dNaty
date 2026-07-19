<div align="center">

<img src="https://raw.githubusercontent.com/pedrovergueiro/dNaty/main/assets/logo.png" alt="dNATY" width="180" />

# dNATY

### Evolutionary AI Model Compression

**8–83% fewer FLOPs (median −56% across 17 real datasets) · accuracy kept · no GPU required**

[![PyPI version](https://img.shields.io/pypi/v/dnaty?color=green&cacheSeconds=300)](https://pypi.org/project/dnaty/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: BSL-1.1](https://img.shields.io/badge/License-BSL--1.1-orange.svg)](https://github.com/pedrovergueiro/dNaty/blob/main/LICENSE)

Compress any PyTorch model with one function call.
dNATY uses multi-objective evolutionary search (NSGA-II) guided by episodic memory to find smaller, faster architectures — automatically, on a standard CPU.

```bash
pip install dnaty
```

[Website](https://dnaty.org) · [Docs](https://dnaty.org/docs) · [Benchmarks](https://dnaty.org/benchmarks) · [Changelog](https://dnaty.org/changelog)

</div>

---

## Quickstart

```python
import torch.nn as nn
import dnaty
from dnaty import compress
from dnaty.experiments.fast_dataset import FastDataset

# 1. Your model — any nn.Module with Linear layers
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512), nn.ReLU(),
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 10)
)

# 2. Load dataset (cached in RAM — zero I/O across generations)
ds = FastDataset("MNIST", device="cpu", train_subset=10_000)

# 3. Compress
result = compress(model, ds, target_flops=0.5, n_generations=30)

print(result.summary())
# CompressResult | arch=[...] | FLOPs -44% (1,133,056 -> ~633K) | acc=0.977
# (exact numbers vary by seed/subset; run scripts/prove_it.py to reproduce)
```

The compressed model is a regular `nn.Module` — drop it into your existing pipeline:

```python
result.model                  # nn.Module, ready for inference
result.accuracy               # 0.977  (example; varies by seed/subset)
result.flops_reduction_pct    # 44.1   (example)
result.arch                   # [301, 226, 32, 128]  ← hidden layer sizes found

# Save / reload
result.save("compressed.pt")
result = dnaty.load("compressed.pt")

# Export to ONNX for edge deployment (no PyTorch needed on the device)
result.export_onnx("model.onnx", input_shape=(784,))

# Measure real CPU latency on your machine
print(result.benchmark_latency((784,)))   # p50/p95/p99 ms + fps
```

---

## Why dNATY?

**The problem:** most models ship larger than they need to be. That means slower inference, higher cloud bills, and models too heavy for edge devices (cameras, drones, robots, industrial boxes). Shrinking them by hand is days of trial-and-error with no guarantee you found the best size/accuracy trade-off.

**What you get with dNATY:**

- **Smaller, cheaper models** — 8–83% fewer FLOPs across 18 real datasets, accuracy kept
- **No GPU** — the search runs on CPU in minutes, so it works in CI and on the hardware you already have
- **No manual architecture design** — point it at a model + dataset, get a deployable `nn.Module` back
- **One function call** — `compress(model, dataset)`; export to `.pt` / `.onnx`

### "Why not just TensorRT or TFLite?" — wrong layer.

Runtimes optimize *execution* of a fixed architecture. dNATY optimizes *the architecture itself*, upstream of any runtime. You don't choose between them — you chain them: `compress()` → `export_onnx()` → load into TensorRT / TFLite / ONNX Runtime. The savings stack.

### Versus other compression techniques

| Method | What it does | Catch |
|---|---|---|
| **Quantization** | Lower-precision weights (fp32→int8) | Same architecture & op count. **Stack it on top of dNATY.** |
| **Pruning** | Zeroes individual weights | Needs sparse runtimes to actually run faster; manual tuning |
| **Distillation** | Trains a small student model | You design the student + write the training loop |
| **DARTS** | Gradient-based architecture search | Needs a GPU + hours of config |
| **Random NAS** | Random architecture sampling | No memory — re-tries bad ideas |
| **dNATY** | Evolves a smaller architecture, memory-guided | CPU-only, one call |

The engine is **episodic memory-guided evolutionary search**: operators that helped in past generations get sampled more often, so it consistently finds better compression than random NAS at the same generation budget — no gradients, no GPU.

---

## Measured results

All numbers measured on a standard desktop CPU, validation accuracy on a held-out 20% split, reproducible from scripts in this repo. Full tables, configs, and caveats: [dnaty.org/benchmarks](https://dnaty.org/benchmarks).

**13 public datasets** (n_generations=30, n_pop=15) — top rows:

| Dataset | Samples | FLOPs ↓ | Val acc | Domain |
|---|---|---|---|---|
| Electrical Fault Detect | 12,001 | **−83.0%** | 99.25% | smart grid sensors |
| Dry Bean Quality | 13,611 | −83.4% | 92.43% | agricultural IoT |
| Predictive Maint. (AI4I) | 10,000 | −83.1% | 96.70% | factory IoT |
| Breast Cancer (UCI) | 569 | −72.6% | 99.56% | clinical tabular |
| Credit Card Fraud (full) | 284,807 | −64.0% | 99.96% | financial anomaly |
| Network Intrusion (NSL-KDD) | 31,490 | −56.3% | 99.46% | edge security |
| HAR Sensors (UCI) | 10,299 | −46.8% | 99.17% | wearables · robotics |
| MNIST (full 70K) | 70,000 | −41.8% | 98.68% | vision · digits |

**5 public Kaggle datasets** — different sizes, domains, and feature counts. Reproduce: `python scripts/benchmark_market_real.py` (downloads from Kaggle, ~2 h on CPU).

| Dataset | Rows | Features | FLOPs ↓ | Val acc | Domain |
|---|---|---|---|---|---|
| IBM HR Employee Attrition | 1,470 | 51 | **−75.5%** | 99.3% | HR / corporate |
| Adult Census Income | 32,561 | 104 | **−74.4%** | 90.4% | social / financial |
| Air Quality (UCI) | 7,674 | 12 | **−45.0%** | 91.2% | environmental sensors |
| Diabetes 130-US Hospitals | 101,766 | 119 | **−8.0%** | 89.3% | clinical / hospital |
| Telco Customer Churn | 7,043 | 45 | +20% ⚠ | 93.2% | telecom |

4 of 5 compressed (median −45%). The Telco case: NSGA-II explored deeper rather than narrower from the `[512, 256, 128]` baseline — `model_grew=True` was raised automatically. Passing a wider baseline (`[1024, 512, 256]`) or increasing `target_flops` resolves it. This is expected Pareto behavior, not a silent failure.

Compression scales with how oversized the baseline is — dNATY finds the right size, it doesn't force a fixed cut. Lean models get small cuts (correct Pareto behavior); the library warns explicitly when nothing was cut.

**Continual learning — 3 benchmarks, 3 seeds each**

Split-MNIST (5 tasks, digit pairs) — proof of concept:

| Method | BWT (↑ better) | |
|---|---|---|
| **dNATY (balanced replay)** | **−0.204** | **~5× less forgetting vs EWC** |
| EWC (λ=400) | −0.998 | near-total forgetting |
| MLP (no CL) | −0.998 | baseline |

Permuted-MNIST (10 tasks, domain-incremental) and Split-CIFAR-10 (5 tasks, class-incremental) are harder benchmarks with results in `results/exp4_*` and `results/exp5_*`. Note: Split-MNIST is a weak benchmark — it is included for comparability with prior work. The harder benchmarks are the primary CL evidence. Full methodology and comparisons to ER-ACE/MAML: [METHODOLOGY.md](METHODOLOGY.md).

<img src="https://raw.githubusercontent.com/pedrovergueiro/dNaty/main/results/cpu_latency/cpu_latency_comparison.png" alt="CPU latency comparison" width="640" />

Reproduce: `python scripts/prove_it.py` (NAS vs random) · `python -m dnaty.experiments.exp3_cl` (Split-MNIST) · `python -m dnaty.experiments.exp4_permuted_mnist` (Permuted-MNIST) · `python -m dnaty.experiments.exp5_split_cifar10` (Split-CIFAR-10)

---

## API at a glance

| You want to… | Use |
|---|---|
| Compress a tabular/sensor MLP | `compress(model, data, target_flops=0.5)` |
| Compress a small CNN trained from scratch | `compress_cnn(model, loader)` *(early access — CIFAR-scale classification)* |
| Compress the head of a pretrained backbone | `compress_with_backbone(resnet, loader, finetune_backbone=True)` |
| Thin out conv layers too | `prune_conv_channels(model, amount=0.3)` |
| Deploy without PyTorch on the device | `result.export_onnx("m.onnx", input_shape=...)` |
| Save / reload | `result.save("m.pt")` / `dnaty.load("m.pt")` |
| Detect data drift in production | `DriftDetector().fit(X_train)` + `ProductionTracker(model, detector)` |
| Profile compute before deciding | `count_flops(model, input_shape)` / `flops_by_layer(...)` |

Supported backbones for `compress_with_backbone`: ResNet, MobileNetV2/V3, EfficientNet, VGG, DenseNet, ViT, and custom models with an `fc`/`classifier`/`head` attribute.

Full reference with copy-paste recipes: [dnaty.org/docs](https://dnaty.org/docs)

### Example — pretrained backbone for edge deployment

```python
import torchvision.models as tv
import dnaty

backbone = tv.mobilenet_v2(weights="IMAGENET1K_V1")
dnaty.prune_conv_channels(backbone, amount=0.2)          # optional: thin convs first

result = dnaty.compress_with_backbone(
    backbone, train_loader,
    target_flops=0.4,
    finetune_backbone=True, finetune_epochs=10,
)
result.export_onnx("mobilenet_edge.onnx", input_shape=(3, 224, 224))
```

### Deterministic results

```python
result = compress(model, ds, target_flops=0.5, n_generations=30, seed=42)
# Same seed → identical result. The pytest suite gates every release on this.
```

---

## Scope, stated plainly

**Strong:** MLPs on tabular/sensor data; classifier heads on frozen CNN/ViT backbones; CPU-only environments.
**Not yet:** full convolutional NAS end-to-end (under development — convs are handled by structural pruning today); transformer/LLM compression; models that are already minimal (no fat → little or no cut, and the library warns you when the model would need to *grow*).

No comparison against OFA or MnasNet is claimed — those target full conv search spaces on GPUs; dNATY targets CPU-only workflows on a different problem slice.

---

## Installation

```bash
pip install dnaty                # stable (recommended)
pip install dnaty==2.0.3         # pin to this release
pip install git+https://github.com/pedrovergueiro/dNaty  # latest from source
```

**Requirements:** Python 3.10+, PyTorch 2.0+, NumPy 1.24+

```bash
pip install dnaty[dev]   # adds pytest, matplotlib, jupyter
```

---

## Project structure

```
dNaty/
├── dnaty/
│   ├── compress.py              # public API: compress, compress_cnn,
│   │                            #   compress_with_backbone, prune_conv_channels
│   ├── result.py                # CompressResult + load() — save/export/latency
│   ├── evolution/evolver.py     # DnatyEvolver / CnnEvolver — NSGA-II search
│   ├── core/                    # DynamicMLP, DynamicCNN, Individual, episodic memory
│   ├── operators/               # structural mutation operators (dense + conv)
│   ├── training/local_train.py  # fast local trainer
│   ├── monitoring/              # DriftDetector, ProductionTracker
│   ├── utils/flops_counter.py   # count_flops, flops_by_layer
│   └── experiments/fast_dataset.py  # zero-I/O MNIST/FashionMNIST/CIFAR10 loader
├── scripts/                     # prove_it.py, benchmark_market_real.py, ...
└── tests/                       # pytest suite (119 tests) — gates every release
```

---

## Hosted version

Prefer not to run it locally? [dnaty.org](https://dnaty.org) hosts the same engine with a web UI and REST API — upload a CSV, get a compressed model back. Free tier: 1 training a day, no card.

---

## Citation

```bibtex
@software{vergueiro_dnaty_2026,
  author  = {Vergueiro, Pedro},
  title   = {dNaty: Dynamic Neuro-Adaptive sYstem with evoluTionarY Learning},
  year    = {2026},
  url     = {https://github.com/pedrovergueiro/dNaty},
  version = {2.0.3},
  license = {BSL-1.1}
}
```

---

## License

[Business Source License 1.1](https://github.com/pedrovergueiro/dNaty/blob/main/LICENSE) — free for research, academic work, and personal projects.
Commercial use requires a license: [dnaty.org/commercial](https://dnaty.org/commercial) · [pedrol.vergueiro@gmail.com](mailto:pedrol.vergueiro@gmail.com)
