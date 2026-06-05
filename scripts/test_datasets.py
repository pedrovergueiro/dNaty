"""
Testa o pipeline completo de upload + treino com 3 formatos:
  1. NPZ  — CIFAR-10 flat (500 amostras, 10 classes, 3072 features)
  2. CSV  — WFP Food Prices (5k linhas, tabular misto)
  3. ZIP  — ImageFolder CIFAR-10 (240 imagens, 3 classes)
Roda localmente sem servidor HTTP — chama as funções internas diretamente.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dnaty_saas"))

import numpy as np
import torch
from routes.train import _parse_tabular
from models.dnaty_model import _load_dataset, _make_nas_subset_loader, _NAS_SUBSET_THRESHOLD, _NAS_SUBSET_SIZE, _FINETUNE_EPOCHS
from dnaty.evolution.evolver import DnatyEvolver
from dnaty.training.local_train import local_train, evaluate
from dnaty.core.arch import DynamicMLP
from dnaty.core.individual import Individual

DATASETS = {
    "NPZ  (cifar_flat)":     ("npz",  r"c:/tmp/test_datasets/cifar_flat.npz"),
    "CSV  (wfp_5k)":         ("csv",  r"c:/tmp/test_datasets/wfp_food_prices_5k.csv"),
    "ZIP  (cifar_images)":   ("zip",  r"c:/tmp/test_datasets/cifar_imagefolder.zip"),
}

NAS_GENS = 5
N_POP    = 8


def run_training(train_loader, test_loader, input_size, n_classes):
    n_train = len(train_loader.dataset)
    use_two_phase = n_train > _NAS_SUBSET_THRESHOLD
    nas_loader = _make_nas_subset_loader(train_loader, _NAS_SUBSET_SIZE) if use_two_phase else train_loader

    torch.manual_seed(42)
    evolver = DnatyEvolver(
        n_pop=N_POP, n_generations=NAS_GENS, t_local=1, lr=1e-3,
        lambda1=1e-4, lambda2=1e-3, device="cpu",
        input_size=input_size, n_classes=n_classes,
        init_hidden=[128, 64], batch_size=512, verbose=False,
    )
    t0 = time.time()
    best, history = evolver.run(nas_loader, test_loader)
    acc_nas, _ = evaluate(best, test_loader)

    if use_two_phase:
        fresh = Individual(DynamicMLP(list(best.model.layer_sizes), list(best.model.activations), n_classes))
        for _ in range(_FINETUNE_EPOCHS):
            local_train(fresh, train_loader, n_epochs=1, lr=1e-3,
                        lambda1=1e-4, lambda2=1e-3, device="cpu", batch_size=512)
        acc_final, _ = evaluate(fresh, test_loader)
    else:
        acc_final = acc_nas

    elapsed = time.time() - t0
    return acc_nas, acc_final, elapsed, use_two_phase, n_train


print(f"\n{'='*65}")
print(f"  dNATY — Teste de Datasets ({NAS_GENS} geracoes NAS, n_pop={N_POP})")
print(f"{'='*65}\n")

for name, (fmt, path) in DATASETS.items():
    print(f"[{name}]")
    t0 = time.time()

    try:
        if fmt in ("npz", "csv"):
            # Tabular: usa _parse_tabular + TensorDataset
            X, y, classes = _parse_tabular(path)
            print(f"  Parse:    {X.shape[0]} amostras, {X.shape[1]} features, {len(classes)} classes")
            # cria loaders manualmente
            Xt = torch.tensor(X, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.long)
            n = len(Xt)
            n_train = int(n * 0.8)
            idx = torch.randperm(n, generator=torch.Generator().manual_seed(42))
            from torch.utils.data import TensorDataset, DataLoader
            train_ds = TensorDataset(Xt[idx[:n_train]], yt[idx[:n_train]])
            test_ds  = TensorDataset(Xt[idx[n_train:]], yt[idx[n_train:]])
            train_loader = DataLoader(train_ds, batch_size=512, shuffle=True,  num_workers=0)
            test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0)
            input_size = X.shape[1]
            n_classes  = len(classes)

        else:  # zip — usa _load_dataset com custom_path
            import zipfile, shutil, tempfile
            tmpdir = Path(tempfile.mkdtemp()) / "extracted"
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tmpdir)
            train_loader, test_loader, input_size, n_classes = _load_dataset(
                "custom", "cpu", train_subset=None, custom_path=str(tmpdir)
            )
            print(f"  ImageFolder: {len(train_loader.dataset)+len(test_loader.dataset)} imgs, {n_classes} classes")

        parse_time = time.time() - t0
        print(f"  Tempo parse/load: {parse_time:.1f}s")

        acc_nas, acc_final, train_time, two_phase, n_train = run_training(
            train_loader, test_loader, input_size, n_classes
        )
        mode = f"2-phase (NAS {min(n_train,_NAS_SUBSET_SIZE)} + ft {_FINETUNE_EPOCHS}ep)" if two_phase else "single-phase"
        print(f"  Amostras treino: {n_train}  |  Modo: {mode}")
        print(f"  Acc NAS:   {acc_nas*100:.2f}%")
        print(f"  Acc final: {acc_final*100:.2f}%")
        print(f"  Tempo treino: {train_time:.1f}s")
        print(f"  Status: OK\n")

    except Exception as e:
        import traceback
        print(f"  ERRO: {e}")
        traceback.print_exc()
        print()
