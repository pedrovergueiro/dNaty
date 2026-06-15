# Changelog

All notable changes to dNATY are documented here.

## [1.1.7] - 2026-06-15 — Cross-platform console output + English-only messages

### Fixed
- **Windows console crash (`UnicodeEncodeError`)** — runtime output used non-ASCII glyphs (`Δ`, `δ`, `↓`, em-dash) that crash on the default Windows code page (cp1252). `compress()`, `result.summary()`, and the per-generation progress line now emit pure ASCII (`delta=`, `d_grad`/`d_mem`, `-`), so the README quickstart runs out-of-the-box on Windows as it already did on Linux/macOS.
- **Portuguese leaking into user-facing messages** — the early-stop notice (`evolver`) and the unsupported-dataset error (`fast_dataset`) still printed Portuguese; both are now English.

No API, behavior, or benchmark changes — output formatting only.

---

## [1.1.6] - 2026-06-11 — Adversarial stress-suite fixes

6 bugs found by `scripts/stress_adversarial.py` (38 adversarial scenarios across 7 categories).

### Fixed
- **`export_onnx()` broken on torch >= 2.6** — the dynamo exporter cannot trace `DynamicMLP`; now forces the TorchScript path (`dynamo=False`, with fallback for torch < 2.5). Numerical parity torch×onnxruntime verified (diff < 1e-4)
- **float64 input crash** — data coming from pandas/numpy (`.values`) crashed with "mat1 and mat2 must have the same dtype"; `DynamicMLP.forward` now casts to float32
- **Pickle RCE in `dnaty.load()`** — was using `weights_only=False`, allowing arbitrary code execution when loading untrusted `.pt` files; now `weights_only=True` with no behavior change for legitimate files
- **`target_flops` range validation** — values outside (0, 1] now raise `ValueError` with an explanation, in `compress()`, `compress_cnn()` and `compress_with_backbone()` (previously accepted 1.5, -0.5, 0.0 silently)
- **`DriftDetector` contract** — accepts `threshold=` as alias of `psi_threshold`; `score()` exposes the documented keys `psi` and `kl_divergence`
- **`ProductionTracker` honors a pre-fitted detector** — the documented flow never triggered drift checks; `meta` now includes `psi` and `n_samples`

### Added
- `scripts/stress_adversarial.py`: reusable 38-scenario adversarial suite (degenerate data, API misuse, ONNX parity, file corruption). Status: 38/38 passing; official suite 72 passed with no regression

---

## [1.1.5] - 2026-06-07 — Internal refactor + CI expansion

### Changed
- `compress.py` split (774 → 519 lines): `CompressResult` + `load()` extracted to `dnaty/result.py`; private helpers to `dnaty/_compress_helpers.py`. Backward compat preserved via re-exports
- CI expanded from 1 to 3 test suites (`test_sanity`, `test_edge_cases`, `test_reproducibility`) on Python 3.10 and 3.11; `test_regression` removed from CI (NAS is stochastic, baseline comparison too noisy)
- `requirements.txt` split into `requirements-lib.txt` / `requirements-saas.txt` / `requirements-dev.txt`

### Added
- `Makefile`: `make test / test-fast / test-all / lint / format / build / clean / dev`
- `scripts/README.md` documenting all 21 benchmark/demo/training/infra scripts

---

## [1.1.4] - 2026-06-07 — Honest metrics when the model grows

### Fixed
- `flops_reduction` no longer clamped with `max(0.0, ...)` — stores the real value; negative means the model grew. Silent failure eliminated
- `summary()` shows ↑/↓/= instead of always "-"
- Benchmark `build_model()` scales hidden layers by `n_classes` (`scale = 1 + 0.2*(n_classes-2)`, cap 3×) so the NAS receives a model with adequate capacity

### Added
- `model_grew: bool` on `CompressResult` for programmatic checks
- Automatic `UserWarning` when the compressed model is larger than the original, explaining the cause (undersized base model) and the fix

---

## [1.1.3] - 2026-06-06 — PyTorch compatibility

### Fixed
- `local_train.py`: `torch.cuda.amp.GradScaler` → `torch.amp.GradScaler("cuda")` (old API deprecated, will be removed in a future PyTorch)

---

## [1.1.2] - 2026-06-05 — Full CNN support via backbone splitting

### Added
- **`compress_with_backbone(backbone, loader)`** — supports ResNet, MobileNetV2, EfficientNet, ViT: extracts frozen-backbone features, runs NAS on the MLP head, splices back. `finetune_backbone=True` fine-tunes end-to-end after NAS
- **`prune_conv_channels(model, amount=0.3)`** — structural L1 channel pruning for Conv2d layers, complementary to the NAS
- `_split_backbone_head()`: auto-detects `feature_dim` via dummy forward (32→224px inputs); supports `fc`, `classifier`, `head`, `heads` attributes
- Public `/docs` page with the full Python API reference

### Fixed
- `compress_cnn` crash on models with `Sequential` heads (MobileNetV2, EfficientNet) — `classifier.out_features` replaced by `_infer_n_classes_from_head()`

---

## [1.1.1] - 2026-06-05 — Transfer learning + video upload (SaaS)

### Added
- Pretrained MobileNetV2 as frozen feature extractor for all image datasets (image → 1280 semantic features → MLP NAS). Accuracy on small image datasets up from 5–50% to 60–95%+
- Native video upload: MP4, AVI, MOV, MKV, WebM, M4V, FLV — extracts up to 300 frames; ZIP with class-folder videos supported
- Adaptive `init_hidden` by input size; stratified per-class split; adaptive val fraction for small datasets

### Changed
- `n_pop` 8 → 12; fine-tune epochs 30 → 50; 2-phase training threshold 8k → 5k samples

---

## [1.1.0] - 2026-05-28 — Public API completeness + persistence

### Fixed
- **`target_flops` was ignored** — `lambda2` was hardcoded at 3e-6 since v1.0.0; now maps `target_flops` to real structural pressure
- `delta_grad` always reported as 0

### Added
- `CompressResult.save(path)` + `dnaty.load(path)` — persists architecture + weights + metrics
- `result.export_onnx(path, input_shape)` — ONNX export for CPU-only edge deploy
- `result.benchmark_latency(input_shape)` — p50/p95/FPS latency on CPU
- `progress_callback` exposed on the public `compress()`
- 2-phase training: NAS on a fast subset, winner fine-tuned on the full data (datasets > 8k samples)
- Production monitoring: `DriftDetector` (PSI/KL) and `ProductionTracker`
- Per-operation FLOPs counter (`nn.Linear`, `nn.Conv2d`, depthwise separable)
- Real Kaggle benchmarks: HAR Sensors −62.8% FLOPs @ 100% acc; Predictive Maintenance (AI4I) −76.2% @ 98.98%; Breast Cancer −72.6%; Credit Card Fraud −75.3%

---

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
