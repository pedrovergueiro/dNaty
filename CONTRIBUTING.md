# Contributing to dNATY

Thanks for considering a contribution. This document covers the workflow and the project's conventions.

## Getting started

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR-USERNAME/dNaty.git`
3. **Create a branch**: `git checkout -b feature/your-feature`
4. **Make changes** and commit with clear messages
5. **Push** to your fork and open a **Pull Request**

## Project layout

```
dNaty/
├── dnaty/
│   ├── compress.py              # public API: compress, compress_cnn,
│   │                            #   compress_with_backbone, prune_conv_channels
│   ├── result.py                # CompressResult + load()
│   ├── evolution/evolver.py     # DnatyEvolver / CnnEvolver (NSGA-II)
│   ├── core/                    # DynamicMLP, DynamicCNN, Individual, episodic memory
│   ├── operators/               # structural mutation operators
│   ├── training/local_train.py  # fast local trainer
│   ├── monitoring/              # DriftDetector, ProductionTracker
│   └── utils/flops_counter.py   # count_flops, flops_by_layer
├── scripts/                     # benchmarks + stress_adversarial.py
└── tests/                       # pytest suite
```

## Conventions

- **Python**: type hints, docstrings on public functions, Black-compatible formatting (88 cols)
- **Commits**: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`
- **No fabricated numbers**: every benchmark figure must come from a script in `scripts/` that reproduces it

## Testing

```bash
pytest tests/ -q                          # full local suite
pytest tests/test_sanity.py -q            # fast smoke (what CI runs first)
python scripts/stress_adversarial.py      # 38 adversarial scenarios
```

CI runs `test_sanity`, `test_edge_cases`, `test_reproducibility` and
`test_benchmark_market` on Python 3.10/3.11. Heavy NAS suites run locally.
A PR should keep the full local suite green.

## Reporting issues

Use the GitHub issue templates:
- **Bug report** — what happened, steps to reproduce, expected result, `dnaty.__version__` and torch version
- **Feature request** — the idea, the use case, why it belongs in scope (CPU-only NAS for MLP/CNN-head compression)

## Code of conduct

Be respectful. Zero tolerance for harassment.
