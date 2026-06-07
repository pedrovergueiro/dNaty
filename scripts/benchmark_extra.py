"""
dNATY v1.1.4 — Extra benchmark batch.

Objetivos:
  1. Re-run Electrical Fault Classify com build_model corrigido (n_classes scale)
     -> antes dava 0% FLOPs / modelo crescia; agora deve comprimir de verdade
  2. Novos datasets reais focados em Edge ML / IoT / industria

Novos datasets:
  A. Steel Plates Faults (UCI)  1.9K  33 feat  7 classes  — inspeção de qualidade fabril
  B. Gas Sensor Array Drift      13K  128 feat  6 classes  — sensor de gas quimico IoT
  C. Dry Bean Quality            13K   16 feat  7 classes  — controle qualidade agricola IoT
  D. Occupancy Estimation        10K   16 feat  4 classes  — smart building sensores CO2/luz/PIR

Configuracao: n_gen=30, pop=15, CPU only, val=held-out 20%.
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _prep_df(df: pd.DataFrame, label_col: str,
             drop_cols: list[str] | None = None,
             max_rows: int | None = None,
             cat_nunique_max: int = 80):
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

    if not parts:
        raise ValueError("Nenhuma coluna utilizavel encontrada apos pre-processamento")

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
    vl = DataLoader(TensorDataset(X[idx[n_tr:]], y[idx[n_tr:]]), batch_size=batch)
    return tr, vl


def build_model(input_size: int, n_classes: int) -> nn.Sequential:
    if input_size >= 500:
        hidden = [1024, 512, 256]
    elif input_size >= 150:
        hidden = [512, 256, 128]
    elif input_size >= 50:
        hidden = [256, 128, 64]
    else:
        hidden = [128, 64, 32]
    # Scale up for harder multi-class so NAS has real fat to cut
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
    if result.model_grew:
        print(f"  [AVISO] modelo cresceu {-result.flops_reduction_pct:.1f}% — base model ainda pequeno")

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
        "model_grew":           result.model_grew,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_elec_fault_classify(kagglehub):
    """Re-run com build_model corrigido para verificar fix do model_grew."""
    path = kagglehub.dataset_download("esathyaprakash/electrical-fault-detection-and-classification")
    class_csv = next(p for p in Path(path).rglob("*.csv") if "class" in p.name.lower())
    df = pd.read_csv(class_csv)
    fault_cols = [c for c in ["G", "C", "B", "A"] if c in df.columns]
    feat_cols  = [c for c in df.columns if c not in fault_cols]
    df["fault_class"] = df[fault_cols].astype(int).apply(
        lambda row: "_".join(str(v) for v in row), axis=1
    )
    return _prep_df(df[feat_cols + ["fault_class"]], label_col="fault_class")


def load_steel_plates(kagglehub):
    """Steel Plates Faults — UCI. 1941 amostras, 27 features, 7 tipos de falha fabril."""
    path = kagglehub.dataset_download("uciml/steel-plates-faults")
    csv = next(Path(path).rglob("*.csv"))
    df = pd.read_csv(csv)
    print(f"  Colunas: {list(df.columns)}")
    # UCI format: ultimas 7 cols sao flags binarias de classe (Pastry, Z_Scratch, etc.)
    # Converter flags em label unico
    fault_types = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
    fault_cols = [c for c in fault_types if c in df.columns]
    if fault_cols:
        feat_cols = [c for c in df.columns if c not in fault_cols]
        df["fault_class"] = df[fault_cols].idxmax(axis=1)
        return _prep_df(df[feat_cols + ["fault_class"]], label_col="fault_class")
    # Fallback: ultima coluna como label
    label_col = df.columns[-1]
    return _prep_df(df, label_col=label_col)


def load_gas_sensor(kagglehub):
    """Gas Sensor Array Drift — 128 sensores quimicos, 6 gases, ~13K amostras."""
    path = kagglehub.dataset_download("uciml/gas-sensor-array-drift")
    # Formato: arquivo .dat com "class_label feat_idx:value ..."
    # Tentar csv primeiro
    csvs = list(Path(path).rglob("*.csv"))
    if csvs:
        df = pd.read_csv(csvs[0])
        label_col = df.columns[0] if df.dtypes.iloc[0] == object else df.columns[-1]
        return _prep_df(df, label_col=label_col)
    # Formato libsvm / .dat
    dats = list(Path(path).rglob("*.dat"))
    if not dats:
        raise ValueError("Formato desconhecido — nem CSV nem .dat encontrado")
    rows, labels = [], []
    for dat in dats:
        for line in dat.read_text(errors="replace").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                label = int(float(parts[0]))
            except ValueError:
                continue
            feats = {}
            for kv in parts[1:]:
                if ":" in kv:
                    idx, val = kv.split(":", 1)
                    feats[int(idx)] = float(val)
            rows.append(feats)
            labels.append(label)
    df_feat = pd.DataFrame(rows).fillna(0).sort_index(axis=1)
    df_feat["gas_label"] = labels
    return _prep_df(df_feat, label_col="gas_label")


def load_dry_bean(kagglehub):
    """Dry Bean Dataset — 13K amostras, 16 features morfologicas, 7 variedades (controle qualidade)."""
    # Tenta varios slugs conhecidos
    slugs = [
        "sansuthi/dry-bean-dataset",
        "muratkoklu/dry-bean-dataset",
        "gauravduttakiit/multiclass-dataset-for-seed-type-determination",
    ]
    path = None
    for slug in slugs:
        try:
            path = kagglehub.dataset_download(slug)
            print(f"  Baixou: {slug}")
            break
        except Exception:
            continue
    if path is None:
        raise ValueError("Nenhum slug do Dry Bean Dataset acessivel")
    # Pode ser xlsx ou csv
    xlsxs = list(Path(path).rglob("*.xlsx"))
    csvs  = list(Path(path).rglob("*.csv"))
    if xlsxs:
        df = pd.read_excel(xlsxs[0])
    elif csvs:
        df = pd.read_csv(csvs[0])
    else:
        raise ValueError("Nenhum arquivo xlsx/csv encontrado")
    print(f"  Colunas: {list(df.columns)}")
    # Label esperado: 'Class' ou ultima coluna string
    label_col = next((c for c in df.columns if c.lower() == "class"), None)
    if label_col is None:
        label_col = df.select_dtypes(include=["object"]).columns[-1]
    return _prep_df(df, label_col=label_col)


def load_occupancy_estimation(kagglehub):
    """Occupancy Estimation — smart building, 16 sensores (CO2, luz, PIR, acelerometro), 4 zones."""
    slugs = [
        "gauravduttakiit/occupancy-estimation",
        "ananthr1/room-occupancy-estimation",
        "claytonmiller/occupancy-estimation-for-smart-buildings",
    ]
    path = None
    for slug in slugs:
        try:
            path = kagglehub.dataset_download(slug)
            print(f"  Baixou: {slug}")
            break
        except Exception:
            continue
    if path is None:
        raise ValueError("Nenhum slug de Occupancy Estimation acessivel")
    csvs = sorted(Path(path).rglob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        raise ValueError("Nenhum CSV encontrado")
    df = pd.read_csv(csvs[0])
    print(f"  Colunas: {list(df.columns)}")
    # Encontra coluna de label: Room_Occupancy_Count, Occupancy, etc.
    label_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["occupancy", "room", "label", "class", "zone"])),
        df.columns[-1]
    )
    drop = [c for c in df.columns if c.lower() in ("date", "time", "datetime") or c.lower().startswith("unnamed")]
    return _prep_df(df, label_col=label_col, drop_cols=drop)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run_all():
    import kagglehub
    results, failed = [], []

    tasks = [
        (
            "0. RE-RUN — Electrical Fault Classify (7.8K, build_model fix)",
            "Electrical Fault Classify (7.8K) [FIXED]",
            load_elec_fault_classify,
            {"target_flops": 0.4, "n_generations": 30, "n_pop": 15},
        ),
        (
            "1. Steel Plates Faults (1.9K, manufacturing quality)",
            "Steel Plates Faults (1.9K)",
            load_steel_plates,
            {"target_flops": 0.4, "n_generations": 30, "n_pop": 15},
        ),
        (
            "2. Gas Sensor Array Drift (13K, chemical IoT sensor)",
            "Gas Sensor Drift (13K)",
            load_gas_sensor,
            {"target_flops": 0.4, "n_generations": 30, "n_pop": 15},
        ),
        (
            "3. Dry Bean Quality (13K, agricultural IoT quality control)",
            "Dry Bean Quality (13K)",
            load_dry_bean,
            {"target_flops": 0.4, "n_generations": 30, "n_pop": 15},
        ),
        (
            "4. Occupancy Estimation (smart building multi-sensor)",
            "Occupancy Estimation (smart building)",
            load_occupancy_estimation,
            {"target_flops": 0.4, "n_generations": 30, "n_pop": 15},
        ),
    ]

    for i, (header, result_name, loader_fn, kwargs) in enumerate(tasks):
        try:
            print(f"\n[{i+1}/{len(tasks)}] {header}...")
            X, y, n_cls = loader_fn(kagglehub)
            print(f"  -> {len(X):,} amostras, {X.shape[1]} features, {n_cls} classes")
            results.append(run_one(result_name, X, y, n_cls, **kwargs))
        except Exception as e:
            print(f"  SKIP: {e}")
            failed.append((result_name, str(e)))

    return results, failed


if __name__ == "__main__":
    print("=" * 65)
    print("  dNATY v1.1.4 — Extra Benchmark (5 datasets)")
    print("  Re-run Elec Fault Classify + 4 novos datasets IoT/industria")
    print("  Gens=30  Pop=15  CPU only  val=held-out 20%")
    print("=" * 65)

    Path("results").mkdir(exist_ok=True)
    results, failed = run_all()

    print("\n\n" + "=" * 65)
    print("  RESUMO FINAL")
    print("=" * 65)
    header = f"{'Dataset':<38} {'N':>7} {'F':>5} {'C':>3} {'FLOPs-':>7} {'ValAcc':>7} {'Grew':>5}"
    print(header)
    print("-" * len(header))
    for r in results:
        grew = "YES" if r.get("model_grew") else "-"
        print(
            f"{r['dataset']:<38} "
            f"{r['samples']:>7,} "
            f"{r['features']:>5} "
            f"{r['classes']:>3} "
            f"{r['flops_reduction_pct']:>6.1f}% "
            f"{r['val_accuracy']:>7.4f} "
            f"{grew:>5}"
        )

    if failed:
        print(f"\n  Skipped ({len(failed)}): {', '.join(n for n, _ in failed)}")

    out = Path("results/benchmark_extra.json")
    out.write_text(json.dumps({"results": results, "failed": failed}, indent=2), encoding="utf-8")
    print(f"\n  Salvo em {out}")
    print(f"  Total: {len(results)}/{len(results)+len(failed)} datasets concluidos")
