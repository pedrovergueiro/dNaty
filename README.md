<div align="center">

# dNATY

### Evolutionary Model Compression — find smaller, faster models automatically

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: BSL 1.1](https://img.shields.io/badge/License-BSL_1.1-blue.svg)](LICENSE)

**dNATY is an open compression layer for PyTorch models.**  
Point it at any model + dataset and it evolves a smaller, faster architecture using guided evolutionary search — no manual tuning, no GPU required.

</div>

---

## Why dNATY

Most models trained today are too big to run on edge devices, phones, or cheap servers.
Existing compression tools (pruning, quantization) require manual configuration and deep ML knowledge.

dNATY does it differently:

- **Evolutionary NAS** — searches architectures guided by episodic memory, not random chance
- **Proven -46.5% FLOPs** vs. random search baseline on MNIST benchmarks
- **Works on CPU** — no GPU required for compression
- **One function call** — drop it into any existing PyTorch project

---

## Quick Start

```bash
pip install dnaty
```

```python
from dnaty import compress
from dnaty.experiments.fast_dataset import FastDataset

# Your existing model (any PyTorch model with Linear layers)
model = your_trained_model

# Your data
ds = FastDataset("MNIST", device="cpu", train_subset=10_000)

# Compress
result = compress(model, ds, target_flops=0.5, n_generations=30)

print(result.summary())
# CompressResult | arch=[128, 64] | FLOPs -46.5% (327680 -> 175104) |
#   params -52.3% (328K -> 156K) | acc=0.9821
```

---

## How It Works

dNATY runs a population of candidate architectures through an evolution loop:

1. **Mutate** — apply structural operators (add/remove neurons, merge layers, etc.)
2. **Train** — locally train each candidate for a few epochs
3. **Select** — NSGA-II Pareto selection: maximize accuracy, minimize FLOPs
4. **Remember** — episodic memory records which operators helped most; they get picked more often next round

The memory mechanism is dNATY's core innovation. Over generations, the search becomes smarter — not random.

---

## API

### `compress(model, train_data, **kwargs) -> CompressResult`

| Parameter | Default | Description |
|---|---|---|
| `model` | required | Any `nn.Module` with Linear layers |
| `train_data` | required | `DataLoader` or `FastDataset` |
| `target_flops` | `0.5` | Target fraction of original FLOPs (0.5 = 50% less) |
| `n_generations` | `30` | Evolutionary generations |
| `n_pop` | `15` | Population size |
| `device` | auto | `'cpu'` or `'cuda'` |
| `seed` | `None` | Fix for reproducibility |

### `CompressResult`

```python
result.model              # compressed nn.Module, ready to use
result.accuracy           # validation accuracy
result.flops_reduction    # e.g. 0.465 = 46.5% fewer FLOPs
result.flops_reduction_pct  # same as percentage
result.params_reduction_pct
result.arch               # hidden layer sizes found  [128, 64]
result.summary()          # one-line human-readable summary
```

---

## SaaS API

dNATY ships with a production-ready FastAPI backend.

```bash
cd dnaty_saas
cp .env.example .env   # fill DATABASE_URL, JWT_SECRET, ANTHROPIC_API_KEY
uvicorn main:app --reload
```

### `POST /api/v1/compress`

```json
{
  "description": "classifica defeitos em pecas, precisa rodar no Raspberry Pi",
  "dataset": "MNIST",
  "target_flops": 0.5,
  "n_generations": 30
}
```

Response `202`:
```json
{ "job_id": "a3f2c1b0", "status": "queued", "message": "..." }
```

### `GET /api/v1/compress/{job_id}`

```json
{
  "status": "completed",
  "result": {
    "accuracy": 0.9821,
    "flops_reduction": 0.465,
    "arch": [128, 64],
    "explanation": "...",        // Claude-generated explanation
    "deployment_code": "..."     // ready-to-use Python code
  }
}
```

> Set `ANTHROPIC_API_KEY` in `.env` to enable Claude explanations.  
> Without it, the endpoint still works — returns template text instead.

---

## Demo

```bash
python scripts/demo_compress.py           # 20 gens, MNIST (~5 min CPU)
python scripts/demo_compress.py --full    # 30 gens, more accurate
python scripts/demo_compress.py --dataset FashionMNIST
```

---

## Benchmarks

| Metric | Value |
|---|---|
| FLOPs reduction vs. initial arch | -46.5% |
| FLOPs reduction vs. RandomNAS | better in Pareto front |
| Speedup to target accuracy | 1.6x fewer generations |
| CL: BWT vs. EWC | 6.9x less forgetting |

All numbers reproducible with `python scripts/prove_it.py`.

---

## License

[BSL 1.1](LICENSE) — free for non-commercial use; contact pedrol.vergueiro@gmail.com for commercial licensing.
