import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, torch.nn as nn
from dnaty import compress
from dnaty.experiments.fast_dataset import FastDataset

for name in ["FashionMNIST"]:
    model = nn.Sequential(nn.Flatten(),
        nn.Linear(784,512), nn.ReLU(),
        nn.Linear(512,256), nn.ReLU(),
        nn.Linear(256,10))
    ds = FastDataset(name, device="cpu", train_subset=10_000)
    t0 = time.time()
    r = compress(model, ds, target_flops=0.5, n_generations=30, seed=42, verbose=False)
    dt = time.time() - t0
    print(f"{name}: {dt:.1f}s = {dt/60:.1f}min | {r.summary()}")
