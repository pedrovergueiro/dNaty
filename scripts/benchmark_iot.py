"""
dNATY IoT-focused benchmark — datasets relevantes ao mercado Edge ML/IoT.
Todos reais, do Kaggle, sem simulacao.

Datasets:
  1. Epileptic Seizure Detection  (11.5K, 178 EEG features, 5 classes) — wearables medicos
  2. Network Intrusion NSL-KDD    (125K,  ~50 features,    multi-class) — edge security / firewall IoT
  3. Electrical Fault Detection   (12K,   6 features,      binary)      — smart grid / subestacao
  4. Electrical Fault Classify    (7.8K,  6 features,      multi-class) — tipo de falha eletrica
  5. Occupancy Detection (bonus)  (20K,   5 sensor feat,   binary)      — smart home CO2/luz/temp

Configuracao: n_gen=30, pop=15, CPU only.
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


def _prep_df(df: pd.DataFrame, label_col: str, drop_cols: list[str] | None = None,
             max_rows: int | None = None, cat_nunique_max: int = 80):
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    y_raw = df[label_col].astype(str)
    classes = sorted(y_raw.unique())
    y = torch.tensor(pd.Categorical(y_raw, categories=classes).codes, dtype=torch.long)
    df = df.drop(columns=[label_col])

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns
                if df[c].nunique() <= cat_nunique_max]

    parts = []
    if num_cols:
        X_num = df[num_cols].fillna(0)
        mu, sigma = X_num.mean(), X_num.std().replace(0, 1)
        parts.append((X_num - mu) / sigma)
    if cat_cols:
        parts.append(pd.get_dummies(df[cat_cols], drop_first=False).astype(float))

    X = torch.tensor(pd.concat(parts, axis=1).fillna(0).values, dtype=torch.float32)

    if max_rows and len(X) > max_rows:
        idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(42))[:max_rows]
        X, y = X[idx], y[idx]

    return X, y, len(classes)


def make_loaders(X, y, batch=512, split=0.8, seed=42):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X), generator=g)
    n_tr = int(len(X) * split)
    tr = DataLoader(TensorDataset(X[idx[:n_tr]], y[idx[:n_tr]]), batch_size=batch, shuffle=True)
    vl = DataLoader(TensorDataset(X[idx[n_tr:]], y[idx[n_tr:]]), batch_size=batch, shuffle=False)
    return tr, vl


def build_model(input_size, n_classes):
    if input_size >= 500:
        hidden = [1024, 512, 256]
    elif input_size >= 150:
        hidden = [512, 256, 128]
    elif input_size >= 50:
        hidden = [256, 128, 64]
    else:
        hidden = [128, 64, 32]
    # Scale up capacity for harder multi-class problems so NAS has real fat to cut
    if n_classes > 2:
        scale = min(3.0, 1.0 + 0.2 * (n_classes - 2))
        hidden = [max(16, int(h * scale)) for h in hidden]
    layers, prev = [], input_size
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


def val_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / total if total else 0.0


def run_one(name, X, y, n_classes, target_flops=0.5, n_generations=30, n_pop=15, seed=42):
    train_loader, val_loader = make_loaders(X, y)
    model = build_model(X.shape[1], n_classes)
    n_params_orig = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"  samples={len(X):,}  feat={X.shape[1]}  classes={n_classes}  params={n_params_orig:,}")
    print(f"  target_flops={target_flops}  gens={n_generations}  pop={n_pop}")
    print(f"{'='*65}")

    t0 = time.time()
    result = compress(model, train_loader,
                      target_flops=target_flops,
                      n_generations=n_generations,
                      n_pop=n_pop,
                      verbose=True,
                      seed=seed)
    elapsed = time.time() - t0
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


def run_all():
    import kagglehub
    results, failed = [], []

    # ------------------------------------------------------------------
    # 1. Epileptic Seizure Detection — EEG wearable medical sensor
    # ------------------------------------------------------------------
    try:
        print("\n[1/5] Epileptic Seizure Detection (EEG, 11.5K, 178 feat)...")
        path = kagglehub.dataset_download("harunshimanto/epileptic-seizure-recognition")
        csv = next(Path(path).rglob("*.csv"))
        df = pd.read_csv(csv)
        # Drop row-ID column (Unnamed: 0 / first unnamed col)
        drop = [c for c in df.columns if c.lower().startswith("unnamed")]
        X, y, n_cls = _prep_df(df, label_col="y", drop_cols=drop)
        print(f"  -> {len(X):,} amostras, {X.shape[1]} features, {n_cls} classes")
        results.append(run_one("Epileptic Seizure (11.5K)", X, y, n_cls,
                               target_flops=0.4, n_generations=30, n_pop=15))
    except Exception as e:
        print(f"  SKIP Epileptic Seizure: {e}")
        failed.append(("Epileptic Seizure", str(e)))

    # ------------------------------------------------------------------
    # 2. Network Intrusion Detection — NSL-KDD, edge firewall / IDS
    # ------------------------------------------------------------------
    try:
        print("\n[2/5] Network Intrusion NSL-KDD (125K, edge security)...")
        path = kagglehub.dataset_download("sampadab17/network-intrusion-detection")
        # Use only Train_data.csv (has label column 'class')
        train_csv = next(p for p in Path(path).rglob("*.csv") if "train" in p.name.lower())
        df = pd.read_csv(train_csv)
        # 'class' col: normal, neptune, warezclient, etc. — keep multi-class
        X, y, n_cls = _prep_df(df, label_col="class", max_rows=100_000)
        print(f"  -> {len(X):,} amostras, {X.shape[1]} features, {n_cls} classes")
        results.append(run_one("Network Intrusion NSL-KDD (125K)", X, y, n_cls,
                               target_flops=0.5, n_generations=30, n_pop=15))
    except Exception as e:
        print(f"  SKIP NSL-KDD: {e}")
        failed.append(("NSL-KDD", str(e)))

    # ------------------------------------------------------------------
    # 3. Electrical Fault Detection — smart grid binary sensor
    # ------------------------------------------------------------------
    try:
        print("\n[3/5] Electrical Fault Detection (smart grid, 12K, 6 feat)...")
        path = kagglehub.dataset_download("esathyaprakash/electrical-fault-detection-and-classification")
        detect_csv = next(p for p in Path(path).rglob("*.csv") if "detect" in p.name.lower())
        df = pd.read_csv(detect_csv)
        # Drop unnamed cols; label = 'Output (S)'
        drop = [c for c in df.columns if c.lower().startswith("unnamed")]
        label = next(c for c in df.columns if "output" in c.lower())
        X, y, n_cls = _prep_df(df, label_col=label, drop_cols=drop)
        print(f"  -> {len(X):,} amostras, {X.shape[1]} features, {n_cls} classes")
        results.append(run_one("Electrical Fault Detect (12K)", X, y, n_cls,
                               target_flops=0.4, n_generations=30, n_pop=15))
    except Exception as e:
        print(f"  SKIP Elec Fault Detect: {e}")
        failed.append(("Elec Fault Detect", str(e)))

    # ------------------------------------------------------------------
    # 4. Electrical Fault Classification — tipo de falha (multi-class)
    # ------------------------------------------------------------------
    try:
        print("\n[4/5] Electrical Fault Classification (smart grid, 7.8K, multi-class)...")
        path = kagglehub.dataset_download("esathyaprakash/electrical-fault-detection-and-classification")
        class_csv = next(p for p in Path(path).rglob("*.csv") if "class" in p.name.lower())
        df = pd.read_csv(class_csv)
        # G, C, B, A = binary fault-type flags; Ia,Ib,Ic,Va,Vb,Vc = signal features
        fault_cols = [c for c in ["G", "C", "B", "A"] if c in df.columns]
        feat_cols  = [c for c in df.columns if c not in fault_cols]
        # Combine fault flags into single class label string
        df["fault_class"] = df[fault_cols].astype(int).apply(
            lambda row: "_".join(str(v) for v in row), axis=1
        )
        X, y, n_cls = _prep_df(df[feat_cols + ["fault_class"]], label_col="fault_class")
        print(f"  -> {len(X):,} amostras, {X.shape[1]} features, {n_cls} classes")
        results.append(run_one("Electrical Fault Classify (7.8K)", X, y, n_cls,
                               target_flops=0.4, n_generations=30, n_pop=15))
    except Exception as e:
        print(f"  SKIP Elec Fault Classify: {e}")
        failed.append(("Elec Fault Classify", str(e)))

    # ------------------------------------------------------------------
    # 5. Room Occupancy Detection — smart home CO2/luz/temperatura/PIR
    # ------------------------------------------------------------------
    try:
        print("\n[5/5] Room Occupancy Detection (smart home sensors)...")
        path = kagglehub.dataset_download("uciml/occupancy-detection-data")
        csvs = sorted(Path(path).rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
        dfs = []
        for csv in csvs:
            try:
                df = pd.read_csv(csv)
                if "Occupancy" in df.columns:
                    dfs.append(df)
            except Exception:
                pass
        if not dfs:
            raise ValueError("CSV com coluna Occupancy nao encontrado")
        df = pd.concat(dfs, ignore_index=True).drop_duplicates()
        drop = [c for c in df.columns if c.lower().startswith("unnamed") or c.lower() == "date"]
        X, y, n_cls = _prep_df(df, label_col="Occupancy", drop_cols=drop)
        print(f"  -> {len(X):,} amostras, {X.shape[1]} features, {n_cls} classes")
        results.append(run_one("Room Occupancy (smart home)", X, y, n_cls,
                               target_flops=0.4, n_generations=30, n_pop=15))
    except Exception as e:
        print(f"  SKIP Room Occupancy: {e}")
        failed.append(("Room Occupancy", str(e)))

    return results, failed


if __name__ == "__main__":
    print("=" * 65)
    print("  dNATY v1.1.3 — IoT Market Benchmark (5 datasets)")
    print("  Edge ML / IoT focus: wearables, security, smart grid, smart home")
    print("  Gens=30  Pop=15  CPU only  val_accuracy = held-out 20%")
    print("=" * 65)

    Path("results").mkdir(exist_ok=True)
    results, failed = run_all()

    print("\n\n" + "=" * 65)
    print("  RESUMO FINAL")
    print("=" * 65)
    header = f"{'Dataset':<33} {'N':>7} {'F':>5} {'C':>4} {'FLOPs-':>7} {'ValAcc':>7} {'Tempo':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        t = r["elapsed_s"]
        t_str = f"{int(t//60)}m{int(t%60):02d}s" if t >= 60 else f"{t:.0f}s"
        print(
            f"{r['dataset']:<33} "
            f"{r['samples']:>7,} "
            f"{r['features']:>5} "
            f"{r['classes']:>4} "
            f"{r['flops_reduction_pct']:>6.1f}% "
            f"{r['val_accuracy']:>7.4f} "
            f"{t_str:>7}"
        )

    if failed:
        print(f"\n  Skipped: {', '.join(n for n, _ in failed)}")

    out = Path("results/benchmark_iot.json")
    out.write_text(json.dumps({"results": results, "failed": failed}, indent=2), encoding="utf-8")
    print(f"\n  Salvo em {out}")
    print(f"  Total: {len(results)}/{len(results)+len(failed)} datasets concluidos")
