"""
dNATY Real-World Market Benchmarks -- 5 public Kaggle datasets.

Datasets (different sizes and domains):
  1. IBM HR Employee Attrition  (1,470 rows, 35 cols) — HR / corporate
  2. Telco Customer Churn       (7,043 rows, 21 cols) — telecom
  3. Air Quality UCI            (9,471 rows, 15 cols) — environmental sensors
  4. Adult Census Income        (32,561 rows, 15 cols) — social / financial
  5. Diabetes 130-US Hospitals  (101,766 rows, 50 cols) — medical

Data lives in: data/market_real/{ibm_hr,telco_churn,air_quality,adult_income,diabetes_130}/

Download with:
  kaggle datasets download -d pavansubhasht/ibm-hr-analytics-attrition-dataset --unzip -p data/market_real/ibm_hr
  kaggle datasets download -d blastchar/telco-customer-churn --unzip -p data/market_real/telco_churn
  kaggle datasets download -d fedesoriano/air-quality-data-set --unzip -p data/market_real/air_quality
  kaggle datasets download -d uciml/adult-census-income --unzip -p data/market_real/adult_income
  kaggle datasets download -d brandao/diabetes --unzip -p data/market_real/diabetes_130
"""
from __future__ import annotations
import sys, time, json, warnings
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty import compress

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "market_real"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Generic preprocessing
# ---------------------------------------------------------------------------

def _encode_and_normalize(df: pd.DataFrame, target_col: str,
                          drop_cols: list[str] | None = None,
                          cat_nunique_max: int = 50) -> tuple[torch.Tensor, torch.Tensor, int]:
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Encode target
    y_raw = df[target_col].astype(str).str.strip()
    classes = sorted(y_raw.unique())
    y = torch.tensor(
        pd.Categorical(y_raw, categories=classes).codes.astype(np.int64), dtype=torch.long
    )
    df = df.drop(columns=[target_col])

    # Split numeric / categorical
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category", "bool"]).columns
        if df[c].nunique() <= cat_nunique_max
    ]

    parts: list[pd.DataFrame] = []
    if num_cols:
        Xn = df[num_cols].fillna(0).astype(float)
        mu, sigma = Xn.mean(), Xn.std().replace(0, 1)
        parts.append((Xn - mu) / sigma)
    if cat_cols:
        parts.append(pd.get_dummies(df[cat_cols], drop_first=False).astype(float))

    if not parts:
        raise ValueError("No usable features found after preprocessing.")

    X = torch.tensor(
        pd.concat(parts, axis=1).fillna(0).values.astype(np.float32), dtype=torch.float32
    )
    return X, y, len(classes)


def _make_loaders(X: torch.Tensor, y: torch.Tensor,
                  batch: int = 512, split: float = 0.8, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)
    n_tr = int(len(X) * split)
    tr = DataLoader(TensorDataset(X[idx[:n_tr]], y[idx[:n_tr]]), batch_size=batch, shuffle=True)
    vl = DataLoader(TensorDataset(X[idx[n_tr:]], y[idx[n_tr:]]), batch_size=batch, shuffle=False)
    return tr, vl


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------

def load_ibm_hr() -> tuple[torch.Tensor, torch.Tensor, int]:
    """IBM HR Employee Attrition — 1,470 rows, binary classification."""
    path = DATA_ROOT / "ibm_hr" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    df = pd.read_csv(path)
    # These columns are constants in this dataset
    drop = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    return _encode_and_normalize(df, target_col="Attrition", drop_cols=drop)


def load_telco_churn() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Telco Customer Churn — 7,043 rows, binary classification."""
    path = DATA_ROOT / "telco_churn" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return _encode_and_normalize(df, target_col="Churn", drop_cols=["customerID"])


def load_air_quality() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Air Quality UCI — ~9,000 hourly sensor readings, 3-class CO level."""
    path = DATA_ROOT / "air_quality" / "AirQuality.csv"
    df = pd.read_csv(path, sep=";", decimal=",")
    # Drop empty trailing columns and Date/Time
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
    df = df.drop(columns=["Date", "Time"], errors="ignore")
    # CO(GT) is the ground-truth CO concentration; use it as target (binned into 3 classes)
    df = df.replace(-200, np.nan)  # -200 is the dataset's missing value sentinel
    df = df.dropna(subset=["CO(GT)"])
    co = df["CO(GT)"].astype(float)
    df["co_class"] = pd.qcut(co, q=3, labels=[0, 1, 2]).astype(int)
    drop = ["CO(GT)"]   # don't leak the raw value we're predicting
    return _encode_and_normalize(df, target_col="co_class", drop_cols=drop)


def load_adult_income() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Adult Census Income — 32,561 rows, binary classification (income >50K)."""
    path = DATA_ROOT / "adult_income" / "adult.csv"
    df = pd.read_csv(path)
    # Replace '?' with NaN so get_dummies handles them as a separate category
    df = df.replace("?", np.nan)
    return _encode_and_normalize(df, target_col="income", drop_cols=["fnlwgt"])


def load_diabetes_130() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Diabetes 130-US Hospitals — 101,766 rows, 3-class readmission prediction."""
    path = DATA_ROOT / "diabetes_130" / "diabetic_data.csv"
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)
    drop = [
        "encounter_id", "patient_nbr",
        # High-cardinality free-text / ID columns
        "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3",
    ]
    return _encode_and_normalize(df, target_col="readmitted", drop_cols=drop)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_real(name: str, domain: str, loader_fn, target_flops: float = 0.5,
                   n_gen: int = 30, n_pop: int = 15) -> dict:
    print(f"\n{'='*70}")
    print(f"{name}")
    print(f"Domain: {domain}")
    print(f"{'='*70}")

    print("  Loading data...", end=" ", flush=True)
    X, y, n_classes = loader_fn()
    n_samples, n_features = X.shape
    print(f"{n_samples:,} rows × {n_features} features → {n_classes} classes")

    train_loader, val_loader = _make_loaders(X, y)

    # Baseline must be clearly oversized so the algorithm has fat to cut.
    # These sizes are intentionally large — dNATY will compress down from here.
    if n_features >= 100:
        hidden = [1024, 512, 256]
    elif n_features >= 30:
        hidden = [512, 256, 128]
    else:
        hidden = [256, 128, 64]

    model = nn.Sequential(
        nn.Linear(n_features, hidden[0]), nn.ReLU(),
        *[layer for h_in, h_out in zip(hidden, hidden[1:])
          for layer in (nn.Linear(h_in, h_out), nn.ReLU())],
        nn.Linear(hidden[-1], n_classes),
    )

    arch_str = [n_features] + hidden + [n_classes]
    print(f"  Baseline architecture: {arch_str}")
    print(f"  Compressing (n_gen={n_gen}, n_pop={n_pop}, target_flops={target_flops})...")

    t0 = time.time()
    result = compress(
        model=model,
        train_data=train_loader,
        n_generations=n_gen,
        n_pop=n_pop,
        target_flops=target_flops,
        verbose=False,
        seed=42,
        finetune_epochs=10,
    )
    elapsed = time.time() - t0

    print(f"\n  Results:")
    print(f"    Val accuracy:    {result.accuracy:.4f}")
    print(f"    FLOPs reduction: {result.flops_reduction_pct:.1f}%")
    print(f"    Compressed arch: {result.arch}")
    print(f"    Search time:     {elapsed/60:.2f} min")

    return {
        "name": name,
        "domain": domain,
        "source": "kaggle (real)",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_classes": n_classes,
        "accuracy": float(result.accuracy),
        "flops_reduction_pct": float(result.flops_reduction_pct),
        "compressed_arch": result.arch,
        "baseline_arch": arch_str,
        "time_seconds": elapsed,
        "n_generations": n_gen,
        "n_pop": n_pop,
        "target_flops": target_flops,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASETS = [
    {
        "name": "IBM HR Employee Attrition",
        "domain": "HR / corporate retention",
        "loader": load_ibm_hr,
        "target_flops": 0.5,
        "n_gen": 30,
        "n_pop": 15,
    },
    {
        "name": "Telco Customer Churn",
        "domain": "telecommunications / subscription churn",
        "loader": load_telco_churn,
        "target_flops": 0.5,
        "n_gen": 30,
        "n_pop": 15,
    },
    {
        "name": "Air Quality (UCI) — CO level",
        "domain": "environmental sensors / smart city",
        "loader": load_air_quality,
        "target_flops": 0.5,
        "n_gen": 30,
        "n_pop": 15,
    },
    {
        "name": "Adult Census Income",
        "domain": "social / financial — income prediction",
        "loader": load_adult_income,
        "target_flops": 0.5,
        "n_gen": 30,
        "n_pop": 15,
    },
    {
        "name": "Diabetes 130-US Hospitals",
        "domain": "clinical / hospital readmission prediction",
        "loader": load_diabetes_130,
        "target_flops": 0.5,
        "n_gen": 30,
        "n_pop": 15,
    },
]


def main():
    print("dNATY Real-World Benchmarks — 5 Kaggle Datasets")
    print("=" * 70)

    results = []
    for i, ds in enumerate(DATASETS, 1):
        print(f"\n[{i}/{len(DATASETS)}]", end=" ", flush=True)
        try:
            r = benchmark_real(
                name=ds["name"],
                domain=ds["domain"],
                loader_fn=ds["loader"],
                target_flops=ds["target_flops"],
                n_gen=ds["n_gen"],
                n_pop=ds["n_pop"],
            )
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    out_file = RESULTS_DIR / "benchmark_market_real.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Saved {len(results)}/{len(DATASETS)} results to {out_file.name}")
    print("=" * 70)

    if results:
        print(f"\n{'Dataset':<40} {'Rows':>8} {'Feats':>6} {'FLOPs↓':>7} {'Acc':>6} {'Time':>7}")
        print("-" * 80)
        for r in results:
            print(
                f"{r['name']:<40} {r['n_samples']:>8,} {r['n_features']:>6} "
                f"{r['flops_reduction_pct']:>6.1f}% {r['accuracy']:>6.3f} {r['time_seconds']/60:>6.1f}m"
            )
        flops_vals = [r["flops_reduction_pct"] for r in results]
        print(f"\n  Mean FLOPs reduction: {np.mean(flops_vals):.1f}%  |  Median: {np.median(flops_vals):.1f}%")


if __name__ == "__main__":
    main()
