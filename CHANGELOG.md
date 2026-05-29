# Changelog

All notable changes to dNATY are documented here.

## [1.0.1] - 2026-05-29 — License metadata fix

### Fixed
- PyPI package metadata: license was incorrectly listed as MIT; corrected to BSL-1.1

---

## [1.0.0] - 2026-05-29 — First Public Release

First stable release of dNATY on PyPI (`pip install dnaty`).

### What's included

- **`compress()` public API** — compress any PyTorch `nn.Module` with one function call
- **Evolutionary NAS** — multi-objective NSGA-II search (maximize accuracy, minimize FLOPs)
- **Episodic memory** — operators that worked before are tried more often; search improves over generations
- **FastDataset** — loads entire dataset into RAM once; zero I/O overhead across all generations
- **CompressResult** — structured result with `.model`, `.accuracy`, `.flops_reduction_pct`, `.arch`, `.summary()`
- **`DnatyEvolver`** — lower-level API for custom search loops
- **DataLoader support** — `compress()` works with `FastDataset` or any standard `torch.utils.data.DataLoader`
- **Web UI** — React + TypeScript frontend with dashboard, benchmarks, pricing pages
- **Production API** — FastAPI backend with auth, job queue, Stripe billing, Prometheus metrics

### Proven results

| Metric | Value |
|---|---|
| FLOPs reduction (MNIST NAS) | **−46.5%** |
| Accuracy retained | **98.59%** |
| Convergence speedup vs RandomNAS | **1.6×** |
| Less forgetting vs EWC (CL) | **6.9×** |

### Bug fixes in this release

- Fixed `compress()` crashing with `AttributeError: count_flops` when passing a generic `nn.Module` (was only working with internal `DynamicMLP`)

### Roadmap

- **v1.1**: CNN support (`arch_cnn` operators)
- **v1.2**: Mixed precision distillation
- **v2.0**: Multi-GPU training + custom architecture search
