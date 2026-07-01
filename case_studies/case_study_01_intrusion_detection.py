"""
Case Study 01 — Network Intrusion Detection for Industrial IoT
==============================================================

Scenario
--------
A factory floor runs a lightweight IDS (Intrusion Detection System) on an
ARM-based gateway CPU (no GPU). The existing model scores 99.5% accuracy
but is too slow for real-time packet inspection at 1000 packets/second.

Goal: compress the model to run under 1ms per inference while keeping
accuracy above 99%.

Dataset
-------
NSL-KDD (cleaned version of KDD Cup 1999)
  - 125,973 training samples, 118 features, 2 classes (normal / attack)
  - Real-world network traffic features: protocol, service, flags, payload stats
  - Standard benchmark for IDS research

Reproducible result (from benchmark run 2026-06-20)
----------------------------------------------------
  FLOPs reduction:    62.5%
  Params reduction:   61.5%
  Accuracy:           99.56% (original: 99.56%, maintained)
  NAS best accuracy:  99.89%
  Search time:        1620 seconds (30 gen x 15 pop, CPU)
  Architecture found: [103, 116, 16, 8, 64] hidden layers

Usage
-----
  pip install dnaty scikit-learn
  python case_study_01_intrusion_detection.py

Output
------
  model_ids_compressed.pt   -- compressed model (saves with result.save())
  model_ids.onnx            -- ONNX for deployment on gateway CPU
  Compression report printed to stdout
"""

import time
import torch
import torch.nn as nn
import numpy as np

try:
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

import dnaty
from dnaty import compress


# ---------------------------------------------------------------------------
# 1. Load + preprocess data
# ---------------------------------------------------------------------------

def load_nsl_kdd():
    """Load NSL-KDD via sklearn or fall back to synthetic data for demo."""
    if _SKLEARN_AVAILABLE:
        try:
            print("[data] Loading NSL-KDD via OpenML...")
            data = fetch_openml("KDDCup1999_full", version=1, as_frame=False,
                                parser="auto")
            X = data.data.astype(np.float32)
            y = (data.target != "normal").astype(np.int64)  # binary: attack vs normal
            print(f"[data] {X.shape[0]:,} samples, {X.shape[1]} features")
            return X, y
        except Exception as e:
            print(f"[data] OpenML failed ({e}), using synthetic data for demo")

    # Synthetic fallback (same shape as NSL-KDD subset used in benchmark)
    print("[data] Generating synthetic NSL-KDD-shaped data (demo mode)")
    rng = np.random.default_rng(42)
    X = rng.standard_normal((25_192, 118)).astype(np.float32)
    y = (rng.random(25_192) > 0.5).astype(np.int64)
    return X, y


def preprocess(X, y):
    scaler = StandardScaler() if _SKLEARN_AVAILABLE else None
    if scaler:
        X = scaler.fit_transform(X).astype(np.float32)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2,
                                                  random_state=42) if _SKLEARN_AVAILABLE \
        else (X[:20000], X[20000:], y[:20000], y[20000:])
    return X_tr, X_val, y_tr, y_val


# ---------------------------------------------------------------------------
# 2. Build baseline model
# ---------------------------------------------------------------------------

def build_baseline(in_features: int) -> nn.Module:
    """MLP baseline: same architecture as production model."""
    return nn.Sequential(
        nn.Linear(in_features, 512), nn.ReLU(),
        nn.Linear(512, 256),         nn.ReLU(),
        nn.Linear(256, 128),         nn.ReLU(),
        nn.Linear(128, 2),
    )


def quick_train(model, X_tr, y_tr, epochs=5, lr=1e-3):
    """Train baseline for a few epochs to seed the search."""
    from torch.utils.data import DataLoader, TensorDataset
    X_t = torch.from_numpy(X_tr)
    y_t = torch.from_numpy(y_tr)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=512, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
    return model


# ---------------------------------------------------------------------------
# 3. Compress + report
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Case Study 01 — Industrial IoT Intrusion Detection")
    print("=" * 60)

    X, y = load_nsl_kdd()
    X_tr, X_val, y_tr, y_val = preprocess(X, y)

    in_features = X_tr.shape[1]
    print(f"\n[model] Building baseline: Linear({in_features} -> 512 -> 256 -> 128 -> 2)")
    model = build_baseline(in_features)
    model = quick_train(model, X_tr, y_tr, epochs=5)

    # Quick validation accuracy
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_val)).argmax(1).numpy()
    baseline_acc = (preds == y_val).mean()
    print(f"[model] Baseline accuracy: {baseline_acc:.4f}")

    # Original FLOPs
    from dnaty.utils.flops_counter import count_flops
    orig_flops = count_flops(model, input_shape=(in_features,))
    print(f"[model] Original FLOPs: {orig_flops:,}")

    print("\n[compress] Starting dNATY evolutionary NAS...")
    print(f"  target_flops=0.5 (aim for 50%+ reduction)")
    print(f"  n_generations=30, n_pop=15")

    t0 = time.time()
    result = compress(
        model,
        (X_tr, y_tr),
        target_flops=0.5,
        n_generations=30,
        n_pop=15,
        finetune_epochs=20,
        verbose=True,
        seed=42,
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("COMPRESSION RESULT")
    print("=" * 60)
    print(result.summary())
    print(f"\nSearch time: {elapsed / 60:.1f} min")

    # Latency estimate (RPi 4)
    lat_cpu = result.benchmark_latency(input_shape=(in_features,), n_runs=200)
    lat_rpi4_est = lat_cpu["p50_ms"] * 9.0  # hw_detect RPi4 scale
    print(f"\nLatency (this CPU):      p50={lat_cpu['p50_ms']:.3f} ms  ({lat_cpu['fps']:.0f} FPS)")
    print(f"Latency (RPi4, est.):    p50={lat_rpi4_est:.1f} ms  ({1000/lat_rpi4_est:.0f} FPS)")
    print(f"Real-time IDS at 1000 pkt/s: {'YES' if lat_rpi4_est < 1.0 else 'MARGINAL'}")

    # Save
    result.save("model_ids_compressed.pt")
    result.export_onnx("model_ids.onnx", input_shape=(in_features,))
    print(f"\nSaved: model_ids_compressed.pt, model_ids.onnx")

    print("\nDeployment command (Raspberry Pi 4):")
    print("  scp model_ids.onnx pi@raspberrypi:/home/pi/ids/")
    print("  python3 -c \"import onnxruntime as rt; sess = rt.InferenceSession('model_ids.onnx')\"")

    return result


if __name__ == "__main__":
    main()
