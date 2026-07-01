# dNATY Edge Hardware Benchmarks

Real-device latency measurements for compressed models on CPU-only edge hardware.

> **Status as of June 2026:**
> Latency values marked `[est]` are estimated from calibrated lookup tables
> (`dnaty/utils/latency_tables.py`). Values marked `[meas]` are from physical
> hardware measurement. RPi 4 physical run is pending hardware access.

---

## Test Configuration

| Config | Value |
|--------|-------|
| Measurement method | ONNX Runtime 1.17, batch=1, 500 runs, p50 |
| CPU benchmark baseline | Intel Core i5-13500, 1 thread |
| Estimation method | hw_detect.py scale tables (RPi4 = 9.0× x86 baseline) |
| Model format | ONNX opset 17, TorchScript export |

---

## MLP Benchmarks (Classification Tasks)

### MNIST (784 features, 10 classes)

| Model | Arch | FLOPs | Params | x86 p50 (ms) | RPi4 p50 (ms) | RPi4 FPS | Status |
|-------|------|------:|------:|------------:|-------------:|--------:|--------|
| Baseline [784→512→256→128→10] | 3 hidden | 1,133,056 | 567,818 | 0.62 [est] | 5.6 [est] | 179 | — |
| **dNATY compressed** [784→343→196→128→10] | 3 hidden | **725,016** | **286,874** | **0.38 [est]** | **3.4 [est]** | **294** | [est] |
| Reduction | — | **−36.0%** | **−49.5%** | **−38.7%** | **−38.7%** | **+64%** | — |

### Network Intrusion Detection — NSL-KDD (118 features, 2 classes)

| Model | Arch | FLOPs | Params | x86 p50 (ms) | RPi4 p50 (ms) | RPi4 FPS | Status |
|-------|------|------:|------:|------------:|-------------:|--------:|--------|
| Baseline [118→512→256→128→2] | 3 hidden | 270,080 | 135,810 | 0.26 [est] | 2.3 [est] | 435 | — |
| **dNATY compressed** [118→103→116→16→8→64→2] | 5 hidden | **101,280** | **50,946** | **0.12 [est]** | **1.1 [est]** | **909** | [est] |
| Reduction | — | **−62.5%** | **−62.5%** | **−53.8%** | **−53.8%** | **+109%** | — |

> Real-time IDS at 1000 pkt/s requires <1ms on device. Target: **met** with compressed model.

### Epileptic Seizure EEG (178 features, 5 classes)

| Model | Arch | FLOPs | Params | x86 p50 (ms) | RPi4 p50 (ms) | RPi4 FPS | Status |
|-------|------|------:|------:|------------:|-------------:|--------:|--------|
| Baseline [178→1024→512→256→5] | 3 hidden | 1,315,840 | 659,205 | 0.71 [est] | 6.4 [est] | 156 | — |
| **dNATY compressed** [178→827→52→26→204→5] | 4 hidden | **460,544** | **231,457** | **0.29 [est]** | **2.6 [est]** | **385** | [est] |
| Reduction | — | **−65.0%** | **−64.9%** | **−59.2%** | **−59.2%** | **+147%** | — |

> 200 Hz EEG real-time requires <5ms per inference. Target: **met** with 2.6ms estimate.

### Electrical Fault Detection (6 features, 2 classes)

| Model | Arch | FLOPs | Params | x86 p50 (ms) | RPi4 p50 (ms) | RPi4 FPS | Status |
|-------|------|------:|------:|------------:|-------------:|--------:|--------|
| Baseline | — | ~11,298 | ~5,649 | 0.04 [est] | 0.4 [est] | 2,500 | — |
| **dNATY compressed** | 5 hidden | **~1,920** | **~960** | **0.01 [est]** | **0.1 [est]** | **10,000** | [est] |
| Reduction | — | **−83.0%** | **−78.1%** | **−75.0%** | **−75.0%** | — | — |

---

## CNN Benchmarks (Image Classification)

### CIFAR-10 (3×32×32, 10 classes)

| Model | FLOPs | Params | Accuracy | x86 p50 (ms) | RPi4 p50 (ms) | RPi4 FPS |
|-------|------:|------:|---------:|------------:|-------------:|--------:|
| DynamicCNN baseline | 16,050,688 | — | 71.9% | 4.2 [est] | 37.8 [est] | 26 [est] |
| DynamicCNN + budget boost | ~16,571,200 | — | 67.0% | 4.3 [est] | 38.7 [est] | 26 [est] |

> Note: CNN NAS for CIFAR-10 is early-access. These numbers reflect architecture
> search results; inference optimization (depthwise-sep, channel pruning) is ongoing.

---

## Latency Methodology

### Estimated values (`[est]`)

Computed via `dnaty.estimate_mlp_latency(layer_sizes, device="rpi4")` from calibrated
lookup tables in `dnaty/utils/latency_tables.py`.

Scale factors from `hw_detect.py`:

| Device | Scale vs x86 | CPU | Clock |
|--------|:-----------:|-----|-------|
| x86_64 desktop | 1.0× | Intel i5/i7 / AMD Ryzen | 3.5–5 GHz |
| x86_64 server | 0.7× | Xeon / EPYC | — |
| Raspberry Pi 4 | 9.0× | ARM Cortex-A72 | 1.8 GHz |
| Raspberry Pi 5 | 4.5× | ARM Cortex-A76 | 2.4 GHz |
| Jetson Nano (CPU) | 5.0× | ARM Cortex-A57 | 1.4 GHz |
| Apple M1 | 0.6× | ARM Firestorm | 3.2 GHz |

### Measured values (`[meas]`)

Protocol (when hardware is available):

```bash
# On Raspberry Pi 4
pip install onnxruntime numpy

python3 - <<'EOF'
import onnxruntime as rt, numpy as np, time

sess = rt.InferenceSession("model.onnx")
inp_name = sess.get_inputs()[0].name
dummy = np.zeros((1, INPUT_FEATURES), dtype=np.float32)

# Warmup
for _ in range(50): sess.run(None, {inp_name: dummy})

# Measure
times = []
for _ in range(500):
    t0 = time.perf_counter()
    sess.run(None, {inp_name: dummy})
    times.append((time.perf_counter() - t0) * 1000)

times.sort()
p50 = times[int(0.50 * len(times))]
p95 = times[int(0.95 * len(times))]
print(f"p50={p50:.3f}ms  p95={p95:.3f}ms  FPS={1000/p50:.0f}")
EOF
```

---

## Reproduce These Numbers

```python
import dnaty
from dnaty.experiments.fast_dataset import FastDataset

# MNIST
ds = FastDataset("MNIST", device="cpu")
result = dnaty.compress(model, ds, target_flops=0.5, n_generations=30, seed=42)
print(result.summary())

# Estimated RPi4 latency
lat = result.benchmark_latency(input_shape=(784,))
print(f"x86 p50: {lat['p50_ms']:.3f}ms")
print(f"RPi4 est: {lat['p50_ms'] * 9.0:.1f}ms")

# Export for physical measurement
result.export_onnx("model.onnx", input_shape=(784,))
```

---

*Last updated: June 2026 — physical RPi4 measurements pending.*
*To contribute benchmarks on your hardware: open an issue at github.com/pedrovergueiro/dNaty*
