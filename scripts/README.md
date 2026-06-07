# scripts/

Utility scripts for development, benchmarking, and data collection.
These are NOT part of the dNATY library — they run standalone.

## Benchmarking

| Script | Purpose |
|--------|---------|
| `benchmark_iot.py` | Primary IoT benchmark suite — 13 real-world datasets |
| `benchmark_extra.py` | Extended benchmarks (Electrical Fault Classify fix, Dry Bean Quality) |
| `benchmark_heavy.py` | Long-running benchmarks (NSL-KDD, large tabular datasets) |
| `benchmark_heavy_part2.py` | Continuation of heavy benchmarks |
| `benchmark_kaggle.py` | Kaggle dataset benchmarks (requires Kaggle API token) |
| `real_benchmarks.py` | General real-dataset benchmarks |
| `real_benchmarks_uci.py` | UCI ML repository benchmarks |
| `bench_params.py` | Parameter sweep benchmarks |

## Examples and Demos

| Script | Purpose |
|--------|---------|
| `demo_compress.py` | Interactive demo: compress a model and inspect results |
| `example_compress.py` | Minimal runnable example (good starting point for new users) |
| `prove_it.py` | End-to-end proof: download dataset → compress → report FLOPs |

## Training

| Script | Purpose |
|--------|---------|
| `train.py` | Train a baseline model before compression |
| `train_mobilenet_cifar100.py` | Train MobileNetV2 on CIFAR-100 (backbone example) |

## Data

| Script | Purpose |
|--------|---------|
| `fetch_real_datasets.py` | Download and cache UCI/Kaggle datasets locally |

## Testing

| Script | Purpose |
|--------|---------|
| `test_compress_real.py` | Real-dataset compression smoke test |
| `test_all_formats.py` | Test save/load across .pt, ONNX, and scripted formats |
| `test_datasets.py` | Validate dataset loading and preprocessing |
| `test_zip_structures.py` | Test ZIP/archive handling for dataset loading |
| `time_compress.py` | Measure compress() wall-clock time across datasets |

## Infrastructure

| Script | Purpose |
|--------|---------|
| `github_setup.py` | One-time GitHub repo configuration |
| `generate_paper.py` | Generate benchmark tables for academic writeups |

## Running

```bash
# Run a benchmark
python scripts/benchmark_iot.py

# Run the demo
python scripts/demo_compress.py

# Time compression on MNIST
python scripts/time_compress.py
```

All scripts expect to be run from the repo root:
```bash
cd /path/to/dNATY
python scripts/<script>.py
```
