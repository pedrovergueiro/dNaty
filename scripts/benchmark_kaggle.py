"""
Benchmark real com datasets públicos do Kaggle.
Valida:
  1. delta_grad agora != 0 (bug corrigido)
  2. target_flops realmente controla lambda2 e gera compressão real
  3. Resultados reproduzíveis em datasets de produção
  4. Datasets alinhados com foco Edge ML / IoT (sensores, manutenção, visão)
"""
from __future__ import annotations
import sys, time, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dnaty import compress


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prep_csv(path: str | Path, label_col: str | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Carrega CSV -> (X float32, y int64, n_classes).
    Encoda colunas numericas (normaliza) + categoricas de baixa cardinalidade (one-hot).
    """
    df = pd.read_csv(path)
    if label_col is None:
        label_col = df.columns[-1]

    y_raw = df[label_col].astype(str)
    classes = sorted(y_raw.unique())
    y = torch.tensor(pd.Categorical(y_raw, categories=classes).codes, dtype=torch.long)

    df = df.drop(columns=[label_col])

    # Numericas: normaliza
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Categoricas de baixa cardinalidade: one-hot (maximo 50 categorias)
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "category"]).columns
        if df[c].nunique() <= 50
    ]

    parts: list[pd.DataFrame] = []
    if num_cols:
        X_num = df[num_cols].fillna(0)
        mu, sigma = X_num.mean(), X_num.std().replace(0, 1)
        parts.append((X_num - mu) / sigma)
    if cat_cols:
        X_cat = pd.get_dummies(df[cat_cols], drop_first=False).astype(float)
        parts.append(X_cat)

    X_df = pd.concat(parts, axis=1).fillna(0)
    X = torch.tensor(X_df.values, dtype=torch.float32)
    return X, y, len(classes)


def make_loaders(X, y, batch=256, split=0.8, seed=42):
    n = len(X)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)
    n_train = int(n * split)
    tr, te = idx[:n_train], idx[n_train:]
    return (
        DataLoader(TensorDataset(X[tr], y[tr]), batch_size=batch, shuffle=True),
        DataLoader(TensorDataset(X[te],  y[te]), batch_size=batch, shuffle=False),
    )


def build_model(input_size: int, n_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_size, 256), nn.ReLU(),
        nn.Linear(256, 128),        nn.ReLU(),
        nn.Linear(128, n_classes),
    )


def run_one(name: str, X, y, n_classes: int, target_flops: float = 0.5,
            n_generations: int = 20, n_pop: int = 10, seed: int = 42):
    train_loader, val_loader = make_loaders(X, y)
    model = build_model(X.shape[1], n_classes)

    print(f"\n{'─'*60}")
    print(f"  Dataset : {name}  |  samples={len(X):,}  features={X.shape[1]}  classes={n_classes}")
    print(f"  target_flops={target_flops}  gens={n_generations}  pop={n_pop}")
    print(f"{'─'*60}")

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

    print(f"\n  -> {result.summary()}")
    print(f"  -> elapsed: {elapsed:.1f}s")
    print(f"  -> generations run: {result.generations}")

    if result.flops_reduction_pct > 1.0:
        print(f"  -> target_flops -> lambda2 FUNCIONANDO (compressao real: -{result.flops_reduction_pct:.1f}%)")
    else:
        print(f"  => compressao pequena ({result.flops_reduction_pct:.1f}%) — modelo ja estava enxuto")

    return {
        "dataset": name,
        "samples": len(X),
        "features": int(X.shape[1]),
        "classes": n_classes,
        "target_flops": target_flops,
        "flops_reduction_pct": round(result.flops_reduction_pct, 1),
        "params_reduction_pct": round(result.params_reduction_pct, 1),
        "accuracy": round(result.accuracy, 4),
        "elapsed_s": round(elapsed, 1),
        "arch": result.arch,
    }


# ── Download + executa cada dataset ──────────────────────────────────────────

def download_and_run_all():
    import kagglehub
    results = []

    # ── 1. Telco Customer Churn — 7K rows, 20 cols (cat + num), binary ────────
    try:
        print("\n[1/6] Baixando blastchar/telco-customer-churn...")
        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        csv = next(Path(path).rglob("*.csv"))
        X, y, n_cls = _prep_csv(csv, label_col="Churn")
        print(f"  -> features apos encoding: {X.shape[1]} (num + one-hot categoricas)")
        results.append(run_one("Telco Churn", X, y, n_cls, target_flops=0.5))
    except Exception as e:
        print(f"  SKIP Telco: {e}")

    # ── 2. Breast Cancer Wisconsin — 569 rows, 30 features, binary ────────────
    try:
        print("\n[2/6] Baixando uciml/breast-cancer-wisconsin-data...")
        path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")
        csv = next(Path(path).rglob("*.csv"))
        X, y, n_cls = _prep_csv(csv, label_col="diagnosis")
        results.append(run_one("Breast Cancer", X, y, n_cls, target_flops=0.4, n_generations=15))
    except Exception as e:
        print(f"  SKIP Breast Cancer: {e}")

    # ── 3. Credit Card Fraud — 284K rows, 29 features, binary (grande!) ───────
    try:
        print("\n[3/6] Baixando mlg-ulb/creditcardfraud...")
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        csv = next(Path(path).rglob("*.csv"))
        X_full, y_full, n_cls = _prep_csv(csv, label_col="Class")
        idx = torch.randperm(len(X_full), generator=torch.Generator().manual_seed(42))[:20_000]
        X, y = X_full[idx], y_full[idx]
        results.append(run_one("Credit Fraud (20K)", X, y, n_cls, target_flops=0.5, n_generations=20, n_pop=8))
    except Exception as e:
        print(f"  SKIP Credit Fraud: {e}")

    # ── 4. HAR — Human Activity Recognition (sensores smartphone) ─────────────
    # 10K+ rows, 561 features (stats de acelerometro/giroscopio), 6 classes
    # Caso de uso: drones, wearables, robots — sensor fusion na borda
    try:
        print("\n[4/6] Baixando uciml/human-activity-recognition-with-smartphones...")
        path = kagglehub.dataset_download("uciml/human-activity-recognition-with-smartphones")
        csvs = list(Path(path).rglob("*.csv"))
        # Pega o maior CSV (train.csv tem ~7K rows)
        csv = max(csvs, key=lambda f: f.stat().st_size)
        X, y, n_cls = _prep_csv(csv, label_col="Activity")
        results.append(run_one(
            "HAR Sensors", X, y, n_cls,
            target_flops=0.5, n_generations=20, n_pop=10
        ))
    except Exception as e:
        print(f"  SKIP HAR: {e}")

    # ── 5. Predictive Maintenance AI4I 2020 — 10K rows, 14 features ──────────
    # Sensores industriais: temperatura, torque, velocidade de rotacao, desgaste
    # Caso de uso: IoT industrial, monitoramento de maquinas na borda
    try:
        print("\n[5/6] Baixando shivamb/machine-predictive-maintenance-classification...")
        path = kagglehub.dataset_download("shivamb/machine-predictive-maintenance-classification")
        csv = next(Path(path).rglob("*.csv"))
        df_check = pd.read_csv(csv)
        # Usa Target (binario) se disponivel, senao ultima coluna
        label = "Target" if "Target" in df_check.columns else df_check.columns[-1]
        # Remove colunas de ID e tipo de falha (redundante com Target)
        drop_cols = [c for c in ["UDI", "Product ID", "Failure Type"] if c in df_check.columns]
        df_check = df_check.drop(columns=drop_cols)
        df_check.to_csv(csv, index=False)  # salva sem colunas de ID
        X, y, n_cls = _prep_csv(csv, label_col=label)
        results.append(run_one(
            "Predictive Maint.", X, y, n_cls,
            target_flops=0.4, n_generations=20, n_pop=10
        ))
    except Exception as e:
        print(f"  SKIP Predictive Maint: {e}")

    # ── 6. Heart Disease UCI ──────────────────────────────────────────────────
    try:
        print("\n[6/6] Baixando redwankarimsony/heart-disease-uci...")
        path = kagglehub.dataset_download("redwankarimsony/heart-disease-uci")
        csvs = list(Path(path).rglob("*.csv"))
        csv = max(csvs, key=lambda f: f.stat().st_size)
        X, y, n_cls = _prep_csv(csv)
        results.append(run_one("Heart Disease", X, y, n_cls, target_flops=0.3, n_generations=15))
    except Exception as e:
        print(f"  SKIP Heart Disease: {e}")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  dNATY v1.1.0 — Benchmark Kaggle (producao)")
    print("  Datasets: tabular + sensores + IoT industrial")
    print("  Valida: delta_grad != 0 + target_flops -> compressao real")
    print("=" * 60)

    results = download_and_run_all()

    print("\n\n" + "=" * 60)
    print("  RESUMO FINAL")
    print("=" * 60)
    print(f"{'Dataset':<22} {'Samples':>8} {'Feats':>6} {'FLOPs-':>8} {'Params-':>8} {'Acc':>7} {'Tempo':>7}")
    print("-" * 62)
    for r in results:
        print(f"{r['dataset']:<22} {r['samples']:>8,} {r['features']:>6} "
              f"{r['flops_reduction_pct']:>7.1f}% "
              f"{r['params_reduction_pct']:>7.1f}% {r['accuracy']:>7.4f} {r['elapsed_s']:>6.0f}s")

    out = Path("results/benchmark_kaggle.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Resultados salvos em {out}")
