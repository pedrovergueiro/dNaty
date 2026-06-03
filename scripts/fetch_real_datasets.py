"""Baixa datasets REAIS do UCI (sem auth) e salva como CSV limpo."""
import urllib.request, gzip, io, time
from pathlib import Path
import pandas as pd

OUT = Path("c:/tmp/real_datasets"); OUT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0"}

def get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()

# 1. UCI Adult / Census Income — 48K rows, prever income >50K (real-world)
try:
    cols = ["age","workclass","fnlwgt","education","education_num","marital_status",
            "occupation","relationship","race","sex","capital_gain","capital_loss",
            "hours_per_week","native_country","income"]
    raw = get("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data")
    df = pd.read_csv(io.BytesIO(raw), names=cols, skipinitialspace=True, na_values="?")
    df = df.dropna()
    df.to_csv(OUT/"uci_adult_income.csv", index=False)
    print(f"Adult Income: {df.shape} -> uci_adult_income.csv", flush=True)
except Exception as e:
    print("Adult FALHOU:", e, flush=True)

# 2. UCI Wine Quality (red) — real-world, prever qualidade
try:
    raw = get("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
    df = pd.read_csv(io.BytesIO(raw), sep=";")
    df.to_csv(OUT/"uci_wine_quality_red.csv", index=False)
    print(f"Wine Quality: {df.shape} -> uci_wine_quality_red.csv", flush=True)
except Exception as e:
    print("Wine FALHOU:", e, flush=True)

# 3. UCI Covertype — 581K rows, 54 feat, 7 classes (floresta). Subset 25K.
try:
    raw = get("https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz", timeout=120)
    data = gzip.decompress(raw)
    df = pd.read_csv(io.BytesIO(data), header=None)
    df = df.sample(n=25000, random_state=42).reset_index(drop=True)
    df.columns = [f"f{i}" for i in range(df.shape[1]-1)] + ["cover_type"]
    df.to_csv(OUT/"uci_covertype_25k.csv", index=False)
    print(f"Covertype (25k subset): {df.shape} -> uci_covertype_25k.csv", flush=True)
except Exception as e:
    print("Covertype FALHOU:", e, flush=True)

print("Datasets reais em:", OUT, flush=True)
