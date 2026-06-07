"""
dNATY — Benchmark pesado em datasets reais grandes do Kaggle.

Datasets: 50K–284K amostras, muitas features, multiclass.
Parametros mais pesados: n_generations=30, n_pop=15.
Modelo base escalado ao input_size para compressao significativa.
"""
from __future__ import annotations
import sys, time, json, warnings
from pathlib import Path

# Windows: garante saida UTF-8 mesmo sem PYTHONIOENCODING=utf-8
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
from dnaty.experiments.fast_dataset import FastDataset


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prep_csv(path: str | Path, label_col: str | None = None,
              max_rows: int | None = None, drop_cols: list[str] | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
    df = pd.read_csv(path)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if label_col is None:
        label_col = df.columns[-1]
    if label_col not in df.columns:
        raise ValueError(f"Label '{label_col}' not in {list(df.columns[:5])}")

    y_raw = df[label_col].astype(str)
    classes = sorted(y_raw.unique())
    y = torch.tensor(pd.Categorical(y_raw, categories=classes).codes, dtype=torch.long)
    df = df.drop(columns=[label_col])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if df[c].nunique() <= 50]

    parts: list[pd.DataFrame] = []
    if num_cols:
        X_num = df[num_cols].fillna(0)
        mu, sigma = X_num.mean(), X_num.std().replace(0, 1)
        parts.append((X_num - mu) / sigma)
    if cat_cols:
        parts.append(pd.get_dummies(df[cat_cols], drop_first=False).astype(float))

    X_df = pd.concat(parts, axis=1).fillna(0)
    X = torch.tensor(X_df.values, dtype=torch.float32)

    if max_rows and len(X) > max_rows:
        g = torch.Generator().manual_seed(42)
        idx = torch.randperm(len(X), generator=g)[:max_rows]
        X, y = X[idx], y[idx]

    return X, y, len(classes)


def make_loaders(X, y, batch=512, split=0.8, seed=42):
    n = len(X)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_train = int(n * split)
    tr, te = idx[:n_train], idx[n_train:]
    return (
        DataLoader(TensorDataset(X[tr], y[tr]), batch_size=batch, shuffle=True),
        DataLoader(TensorDataset(X[te], y[te]), batch_size=batch, shuffle=False),
    )


def build_model(input_size: int, n_classes: int) -> nn.Module:
    """Modelo base escalado ao input_size — tem gordura real para o NAS comprimir."""
    if input_size >= 500:
        hidden = [1024, 512, 256]
    elif input_size >= 150:
        hidden = [512, 256, 128]
    elif input_size >= 50:
        hidden = [256, 128, 64]
    else:
        hidden = [128, 64, 32]
    layers: list[nn.Module] = []
    prev = input_size
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


def val_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += len(yb)
    return correct / total if total else 0.0


def run_one(name: str, X, y, n_classes: int,
            target_flops: float = 0.5,
            n_generations: int = 30,
            n_pop: int = 15,
            seed: int = 42) -> dict:
    train_loader, val_loader = make_loaders(X, y)
    model = build_model(X.shape[1], n_classes)

    n_params_orig = sum(p.numel() for p in model.parameters())

    print(f"\n{'═'*65}")
    print(f"  {name}")
    print(f"  samples={len(X):,}  features={X.shape[1]}  classes={n_classes}  params={n_params_orig:,}")
    print(f"  target_flops={target_flops}  gens={n_generations}  pop={n_pop}")
    print(f"{'═'*65}")

    t0 = time.time()
    result = compress(
        model, train_loader,
        target_flops=target_flops,
        n_generations=n_generations,
        n_pop=n_pop,
        verbose=True,
        seed=seed,
    )
    elapsed = time.time() - t0

    # Acuracia no val set com o modelo comprimido
    val_acc = val_accuracy(result.model, val_loader)

    print(f"\n  RESULTADO: {result.summary()}")
    print(f"  Val acc: {val_acc:.4f}  |  elapsed: {elapsed:.0f}s  |  gens: {result.generations}")

    return {
        "dataset":              name,
        "samples":              len(X),
        "features":             int(X.shape[1]),
        "classes":              n_classes,
        "params_orig":          n_params_orig,
        "target_flops":         target_flops,
        "flops_reduction_pct":  round(result.flops_reduction_pct, 1),
        "params_reduction_pct": round(result.params_reduction_pct, 1),
        "val_accuracy":         round(val_acc, 4),
        "nas_best_accuracy":    round(result.accuracy, 4),
        "elapsed_s":            round(elapsed, 1),
        "arch":                 result.arch,
        "generations":          result.generations,
    }


# ── Datasets ──────────────────────────────────────────────────────────────────

def run_all():
    import kagglehub
    results = []
    failed = []

    # ── 1. Credit Card Fraud COMPLETO — 284K rows, 30 features, binary ──────
    try:
        print("\n[1/8] Credit Card Fraud (284K — COMPLETO)...")
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        csv = next(Path(path).rglob("*.csv"))
        X, y, n_cls = _prep_csv(csv, label_col="Class")
        print(f"  -> {len(X):,} amostras carregadas (dataset completo)")
        results.append(run_one("Credit Fraud FULL (284K)", X, y, n_cls,
                               target_flops=0.4, n_generations=30, n_pop=15))
    except Exception as e:
        print(f"  SKIP: {e}"); failed.append(("Credit Fraud", str(e)))

    # ── 2. Adult Income / Census — 48K rows, 14 features (cat+num), binary ──
    try:
        print("\n[2/8] Adult Income Census (48K)...")
        path = kagglehub.dataset_download("wenruliu/adult-income-dataset")
        csv = next(Path(path).rglob("*.csv"))
        X, y, n_cls = _prep_csv(csv, label_col="income")
        print(f"  -> {len(X):,} amostras, {X.shape[1]} features apos encoding")
        results.append(run_one("Adult Income (48K)", X, y, n_cls,
                               target_flops=0.4, n_generations=30, n_pop=15))
    except Exception as e:
        print(f"  SKIP: {e}"); failed.append(("Adult Income", str(e)))

    # ── 3. Stellar Classification SDSS17 — 100K rows, 17 features, 3 classes ─
    # galaxy / star / quasar — caso real de classificacao multiclass grande
    try:
        print("\n[3/8] Stellar Classification SDSS17 (100K, 3 classes)...")
        path = kagglehub.dataset_download("fedesoriano/stellar-classification-dataset-sdss17")
        csv = next(Path(path).rglob("*.csv"))
        X, y, n_cls = _prep_csv(csv, label_col="class",
                                 drop_cols=["obj_ID", "run_ID", "rerun_ID", "cam_col",
                                            "field_ID", "spec_obj_ID", "fiber_ID", "MJD", "plate"])
        print(f"  -> {len(X):,} amostras, {X.shape[1]} features, {n_cls} classes")
        results.append(run_one("Stellar Class. (100K)", X, y, n_cls,
                               target_flops=0.5, n_generations=30, n_pop=15))
    except Exception as e:
        print(f"  SKIP Stellar: {e}"); failed.append(("Stellar", str(e)))

    # ── 4. Covertype Forest — 581K rows, 54 features, 7 classes → 100K ──────
    try:
        print("\n[4/8] Forest Covertype (100K subset de 581K, 7 classes)...")
        path = kagglehub.dataset_download("uciml/forest-cover-type-dataset")
        csv = next(Path(path).rglob("*.csv"))
        X, y, n_cls = _prep_csv(csv, max_rows=100_000)
        print(f"  -> {len(X):,} amostras (subset), {X.shape[1]} features, {n_cls} classes")
        results.append(run_one("Covertype Forest (100K)", X, y, n_cls,
                               target_flops=0.5, n_generations=30, n_pop=15))
    except Exception as e:
        print(f"  SKIP Covertype: {e}"); failed.append(("Covertype", str(e)))

    # ── 5. HAR Sensors — 10K rows, 561 features, 6 classes ──────────────────
    try:
        print("\n[5/8] HAR Sensors (10K, 561 features)...")
        path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")
        csvs = list(Path(path).rglob("*.csv"))
        # Combina train + test se existirem
        dfs = []
        for c in csvs:
            try:
                df = pd.read_csv(c)
                if "Activity" in df.columns or "subject" in df.columns.str.lower().tolist():
                    dfs.append(df)
            except Exception:
                pass
        if len(dfs) > 1:
            combined = pd.concat(dfs, ignore_index=True)
            tmp = Path("results/_har_combined.csv")
            combined.to_csv(tmp, index=False)
            X, y, n_cls = _prep_csv(tmp, label_col="Activity")
            tmp.unlink(missing_ok=True)
        else:
            csv = max(csvs, key=lambda f: f.stat().st_size)
            X, y, n_cls = _prep_csv(csv, label_col="Activity")
        print(f"  -> {len(X):,} amostras, {X.shape[1]} features, {n_cls} classes")
        results.append(run_one("HAR Sensors (10K)", X, y, n_cls,
                               target_flops=0.4, n_generations=30, n_pop=15))
    except Exception as e:
        print(f"  SKIP HAR: {e}"); failed.append(("HAR", str(e)))

    # ── 6. Telco Churn — 7K rows, 44 features (cat+num), binary ────────────
    try:
        print("\n[6/8] Telco Customer Churn (7K, mixed features)...")
        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        csv = next(Path(path).rglob("*.csv"))
        X, y, n_cls = _prep_csv(csv, label_col="Churn",
                                 drop_cols=["customerID"])
        print(f"  -> {len(X):,} amostras, {X.shape[1]} features apos one-hot")
        results.append(run_one("Telco Churn (7K)", X, y, n_cls,
                               target_flops=0.4, n_generations=30, n_pop=15))
    except Exception as e:
        print(f"  SKIP Telco: {e}"); failed.append(("Telco", str(e)))

    # ── 7. MNIST via FastDataset — 70K rows, 784 features, 10 classes ───────
    try:
        print("\n[7/8] MNIST (70K, 784 features, 10 classes)...")
        ds = FastDataset("MNIST", device="cpu", train_subset=60_000)
        n_feat = 784
        model_mnist = nn.Sequential(
            nn.Linear(784, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256),  nn.ReLU(),
            nn.Linear(256, 10),
        )
        t0 = time.time()
        result = compress(model_mnist, ds, target_flops=0.5,
                          n_generations=30, n_pop=15, verbose=True, seed=42)
        elapsed = time.time() - t0
        print(f"\n  RESULTADO: {result.summary()}")
        print(f"  elapsed: {elapsed:.0f}s")
        results.append({
            "dataset":              "MNIST (70K)",
            "samples":              70000,
            "features":             784,
            "classes":              10,
            "params_orig":          sum(p.numel() for p in model_mnist.parameters()),
            "target_flops":         0.5,
            "flops_reduction_pct":  round(result.flops_reduction_pct, 1),
            "params_reduction_pct": round(result.params_reduction_pct, 1),
            "val_accuracy":         round(result.accuracy, 4),
            "nas_best_accuracy":    round(result.accuracy, 4),
            "elapsed_s":            round(elapsed, 1),
            "arch":                 result.arch,
            "generations":          result.generations,
        })
    except Exception as e:
        print(f"  SKIP MNIST: {e}"); failed.append(("MNIST", str(e)))

    # ── 8. Predictive Maintenance — 10K rows, 8 features, binary ────────────
    try:
        print("\n[8/8] Predictive Maintenance AI4I (10K)...")
        path = kagglehub.dataset_download("shivamb/machine-predictive-maintenance-classification")
        csv = next(Path(path).rglob("*.csv"))
        X, y, n_cls = _prep_csv(csv, label_col="Target",
                                 drop_cols=["UDI", "Product ID", "Failure Type"])
        print(f"  -> {len(X):,} amostras, {X.shape[1]} features")
        results.append(run_one("Predictive Maint. (10K)", X, y, n_cls,
                               target_flops=0.4, n_generations=30, n_pop=15))
    except Exception as e:
        print(f"  SKIP Predictive Maint: {e}"); failed.append(("Pred Maint", str(e)))

    return results, failed


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  dNATY v1.1.1 — Heavy Benchmark (datasets reais grandes)")
    print("  Gens=30  Pop=15  Modelo base escalado  Val accuracy separada")
    print("=" * 65)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results, failed = run_all()

    print("\n\n" + "=" * 65)
    print("  RESUMO FINAL")
    print("=" * 65)
    header = f"{'Dataset':<28} {'Amostras':>9} {'Feats':>6} {'Classes':>8} {'FLOPs↓':>8} {'Params↓':>8} {'Val Acc':>8} {'Tempo':>7}"
    print(header)
    print("─" * len(header))
    for r in results:
        print(
            f"{r['dataset']:<28} "
            f"{r['samples']:>9,} "
            f"{r['features']:>6} "
            f"{r['classes']:>8} "
            f"{r['flops_reduction_pct']:>7.1f}% "
            f"{r['params_reduction_pct']:>7.1f}% "
            f"{r['val_accuracy']:>8.4f} "
            f"{r['elapsed_s']:>6.0f}s"
        )

    if failed:
        print(f"\n  Skipped ({len(failed)}): {', '.join(n for n, _ in failed)}")

    out = results_dir / "benchmark_heavy.json"
    out.write_text(json.dumps({"results": results, "failed": failed}, indent=2))
    print(f"\n  Resultados salvos em {out}")
    print(f"  Total: {len(results)} datasets concluidos")
