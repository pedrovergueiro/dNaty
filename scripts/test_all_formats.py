"""
Testa TODOS os formatos que o dNATY diz aceitar (ACCEPTED_EXTS).
Para cada formato: gera dataset de exemplo -> _parse_tabular -> treino rapido.
Reporta OK / ERRO por formato.
"""
import sys, time, json, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dnaty_saas"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

from routes.train import _parse_tabular, ACCEPTED_EXTS
from dnaty.evolution.evolver import DnatyEvolver
from dnaty.training.local_train import evaluate

OUT = Path(tempfile.mkdtemp(prefix="dnaty_fmt_"))

# Base dataset: 300 linhas, 4 features numericas + 1 categorica + label (3 classes)
rng = np.random.default_rng(42)
N = 300
df = pd.DataFrame({
    "f1": rng.normal(size=N),
    "f2": rng.normal(size=N),
    "f3": rng.integers(0, 100, size=N),
    "f4": rng.normal(size=N),
    "cat": rng.choice(["red", "green", "blue"], size=N),
    "label": rng.integers(0, 3, size=N),
})

def make_file(ext: str) -> Path:
    """Cria um arquivo no formato pedido. Retorna o path, ou levanta para 'nao gerei'."""
    p = OUT / f"data{ext}"
    if ext in (".csv", ".txt", ".md"):
        df.to_csv(p, index=False)
    elif ext == ".tsv":
        df.to_csv(p, sep="\t", index=False)
    elif ext in (".xlsx", ".xls"):
        df.to_excel(p, index=False)
    elif ext == ".json":
        df.to_json(p, orient="records")
    elif ext == ".parquet":
        df.to_parquet(p)
    elif ext == ".npy":
        # numerico apenas (sem cat) + label como ultima coluna
        arr = np.column_stack([df[["f1","f2","f3","f4"]].values, df["label"].values])
        np.save(p, arr)
    elif ext == ".npz":
        np.savez(p, X=df[["f1","f2","f3","f4"]].values, y=df["label"].values)
    elif ext in (".h5", ".hdf5"):
        df.to_hdf(p, key="data", mode="w")
    elif ext in (".yaml", ".yml"):
        import yaml
        with open(p, "w") as f:
            yaml.safe_dump(df.to_dict(orient="records"), f)
    elif ext == ".xml":
        df.to_xml(p, index=False)
    else:
        raise ValueError(f"gerador nao implementado para {ext}")
    return p

def quick_train(X, y, classes):
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    n = len(Xt); idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    ntr = max(1, int(n*0.8))
    tr = DataLoader(TensorDataset(Xt[idx[:ntr]], yt[idx[:ntr]]), batch_size=64, shuffle=True)
    te = DataLoader(TensorDataset(Xt[idx[ntr:]], yt[idx[ntr:]]), batch_size=64)
    torch.manual_seed(0)
    ev = DnatyEvolver(n_pop=6, n_generations=3, t_local=1, lr=1e-3, lambda1=1e-4, lambda2=1e-3,
        device="cpu", input_size=X.shape[1], n_classes=len(classes), init_hidden=[32,16],
        batch_size=64, verbose=False)
    best, _ = ev.run(tr, te); acc, _ = evaluate(best, te)
    return acc

# Formatos tabulares (ZIP testado separado)
TABULAR = sorted(e for e in ACCEPTED_EXTS if e != ".zip")

print(f"Testando {len(TABULAR)} formatos tabulares\n{'='*60}")
results = {}
for ext in TABULAR:
    try:
        path = make_file(ext)
    except Exception as e:
        results[ext] = ("GERADOR FALHOU", str(e))
        print(f"  {ext:8} -> nao consegui gerar amostra: {e}")
        continue
    try:
        X, y, classes = _parse_tabular(path)
        acc = quick_train(X, y, classes)
        results[ext] = ("OK", f"X={X.shape} {len(classes)}cls acc={acc*100:.1f}%")
        print(f"  {ext:8} -> OK   X={X.shape} {len(classes)}cls acc={acc*100:.1f}%")
    except Exception as e:
        results[ext] = ("ERRO", f"{type(e).__name__}: {e}")
        print(f"  {ext:8} -> ERRO {type(e).__name__}: {e}")

print(f"\n{'='*60}")
ok = sum(1 for v in results.values() if v[0] == "OK")
print(f"Resultado: {ok}/{len(TABULAR)} formatos OK")
fails = {k: v for k, v in results.items() if v[0] != "OK"}
if fails:
    print("\nFALHAS:")
    for k, (st, msg) in fails.items():
        print(f"  {k}: {msg}")
