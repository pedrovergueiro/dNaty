"""
dNATY — Teste em datasets reais de negócio (local CPU)
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from dnaty.evolution.evolver import DnatyEvolver

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess(df, target_col):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    y = df[target_col].values.astype(int)
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    X = StandardScaler().fit_transform(X)
    return X, y


def make_loaders(X, y, batch_size=512):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    def loader(Xd, yd, shuffle):
        ds = TensorDataset(torch.tensor(Xd), torch.tensor(yd, dtype=torch.long))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader(X_tr, y_tr, True), loader(X_te, y_te, False), X_tr.shape[1], len(np.unique(y))


def run_dnaty(name, train_loader, test_loader, input_size, n_classes):
    print(f"\n{'='*60}")
    print(f"dNATY x {name}")
    print(f"Input: {input_size} features | Classes: {n_classes} | Device: {DEVICE.upper()}")
    print(f"{'='*60}")

    evolver = DnatyEvolver(
        n_pop=10, n_generations=15,
        t_local=3, lr=1e-3, lambda1=1e-4, lambda2=1e-3,
        device=DEVICE, input_size=input_size, n_classes=n_classes,
        init_hidden=[128, 64], batch_size=512, verbose=True,
    )

    t0 = time.time()
    best, history = evolver.run(train_loader, test_loader)
    duration = time.time() - t0

    print(f"\n--- RESULTADO: {name} ---")
    print(f"Acuracia:    {best.acc*100:.2f}%")
    print(f"Parametros:  {best.count_params():,}")
    print(f"Tempo:       {duration:.1f}s ({duration/len(history):.1f}s/gen)")
    print(f"Arquitetura: {best.model.layer_sizes}")
    print(f"Ativacoes:   {best.model.activations}")
    print(f"Progressao:")
    for log in history:
        bar = "#" * int(log.best_acc * 30)
        print(f"  Gen {log.gen:02d}: {log.best_acc*100:.2f}% |{bar}|")
    return best.acc, duration, len(history)


# ── Dataset 1: Adult Income ────────────────────────────────────────────────────
print("Baixando Adult Income (OpenML #1590)...")
data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
df = data.frame.copy()
df["y"] = (df["class"].astype(str).str.strip() == ">50K").astype(int)
df = df.drop(columns=["class"])
print(f"Linhas: {len(df):,} | Renda alta: {df.y.sum():,} ({df.y.mean()*100:.1f}%)")

X1, y1 = preprocess(df, "y")
tr1, te1, inp1, nc1 = make_loaders(X1, y1)
acc1, dur1, g1 = run_dnaty("Adult Income — Predicao de renda (>$50K)", tr1, te1, inp1, nc1)


# ── Dataset 2: Bank Marketing ──────────────────────────────────────────────────
print("\n\nBaixando Bank Marketing (OpenML #1461)...")
data2 = fetch_openml(data_id=1461, as_frame=True, parser="auto")
df2 = data2.frame.copy()
df2["y"] = (df2["Class"].astype(str) == "2").astype(int)
df2 = df2.drop(columns=["Class"])
print(f"Linhas: {len(df2):,} | Assinou deposito: {df2.y.sum():,} ({df2.y.mean()*100:.1f}%)")

X2, y2 = preprocess(df2, "y")
tr2, te2, inp2, nc2 = make_loaders(X2, y2)
acc2, dur2, g2 = run_dnaty("Bank Marketing — Conversao de clientes", tr2, te2, inp2, nc2)


# ── Resumo ─────────────────────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("RESUMO FINAL — dNATY em Dados Reais de Negocio")
print(f"{'='*60}")
print(f"{'Dataset':<40} {'Acuracia':>10} {'Tempo':>8} {'Gens':>5}")
print(f"{'-'*60}")
print(f"{'Adult Income (48k pessoas)':<40} {acc1*100:>9.2f}% {dur1:>7.1f}s {g1:>5}")
print(f"{'Bank Marketing (45k clientes)':<40} {acc2*100:>9.2f}% {dur2:>7.1f}s {g2:>5}")
print(f"{'='*60}")
print(f"Device: {DEVICE.upper()}")
print(f"Tempo total: {(dur1+dur2)/60:.1f} minutos")
