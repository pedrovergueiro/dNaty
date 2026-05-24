#!/usr/bin/env python3
"""
dNATY cifar_proof.py -- CIFAR-10 com CnnEvolver.

Prova:
  1. dNATY CNN vs RandomNAS CNN (ablation) -- curvas de convergencia
  2. Custo de busca total em GFLOPs
  3. VRAM pico (se CUDA disponivel)
  4. Comparacao com ResNet-8 treinado do zero (mesmo budget)

Uso:
  python scripts/cifar_proof.py           # 20 gens, 20K (~40 min CPU)
  python scripts/cifar_proof.py --quick   # 10 gens, 10K (~12 min CPU)
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from dnaty.experiments.fast_dataset import FastDataset
from dnaty.evolution.evolver import CnnEvolver
from dnaty.operators.mutations_cnn import CNN_OPERATORS, apply_cnn_operator
from dnaty.core.arch_cnn import DynamicCNN
from dnaty.core.individual import Individual
from dnaty.core.memory import EpisodicMemory

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="10 gens, 10K (~12 min CPU)")
parser.add_argument("--gpu",   action="store_true", help="30 gens, 50K full, batch 512 (GPU)")
args = parser.parse_args()

if args.quick:
    N_GEN, N_POP, SUBSET, BATCH = 10, 8, 10_000, 256
elif args.gpu:
    N_GEN, N_POP, SUBSET, BATCH = 30, 12, 50_000, 512
else:
    N_GEN, N_POP, SUBSET, BATCH = 20, 10, 20_000, 256

T_LOCAL = 3
SEED    = 42
LAMBDA2 = 3e-7  # pressao de FLOPs: evita crescimento descontrolado da CNN
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
W = 64


# ── Visuais ───────────────────────────────────────────────────────────────────
def banner(t): print(f"\n{'='*W}\n  {t}\n{'='*W}")
def section(t): print(f"\n{'-'*W}\n  {t}\n{'-'*W}")
def ok(m): print(f"  [OK] {m}")
def fail(m): print(f"  [!!] {m}")
def info(m): print(f"  ... {m}")


# ── RandomNAS CNN (ablation) ──────────────────────────────────────────────────
class RandomSearchCNN(CnnEvolver):
    def _mutate_population(self, population):
        mutated = []
        for ind in population:
            op = np.random.choice(CNN_OPERATORS)
            new_ind, success = apply_cnn_operator(ind, op)
            if not success or not new_ind.model.is_valid():
                new_ind = ind.clone()
                new_ind.last_op = "no_op"
            mutated.append(new_ind)
        return mutated

    def _update_memory(self, *a, **kw):
        return 0.0


# ── ResNet-8 baseline ─────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.block(x) + x)


class ResNet8(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResBlock(128), nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, n_classes)
    def forward(self, x): return self.fc(self.net(x).view(x.size(0), -1))
    def count_params(self): return sum(p.numel() for p in self.parameters())
    def count_flops(self):
        # Approximate: 2 conv blocks (64ch) + 2 (128ch) on 32x32 and 16x16
        return 2 * (9*3*64*32*32 + 9*64*64*32*32 + 9*64*64*32*32 +
                    9*64*128*16*16 + 9*128*128*16*16 + 9*128*128*16*16 +
                    2*128*10)


def train_resnet(fast_ds, n_epochs, device, seed=42):
    torch.manual_seed(seed)
    model = ResNet8().to(device)
    opt  = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-4)
    n_batches = max(1, fast_ds.n_train // BATCH)

    for epoch in range(n_epochs):
        model.train()
        for _ in range(n_batches):
            xb, yb = fast_ds.get_train_batch(BATCH)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            crit(model(xb), yb).backward()
            opt.step()
        sched.step()

    model.eval()
    vx, vy = fast_ds.get_val()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(vx), 512):
            xb = vx[i:i+512].to(device)
            yb = vy[i:i+512].to(device)
            correct += (model(xb).argmax(1) == yb).sum().item()
            total += len(yb)
    return correct / max(total, 1), model.count_params()


# ── Custo de busca ────────────────────────────────────────────────────────────
def estimate_search_gflops(n_gens_run, n_pop, init_flops, subset, batch, t_local):
    """GFLOPs totais consumidos durante o search (aproximacao)."""
    batches_per_epoch = max(1, subset // batch)
    # Por individuo por gen: treino (3x para fwd+bwd) + avaliacao (1x)
    flops_train = init_flops * batches_per_epoch * t_local * 3
    flops_eval  = init_flops * (10_000 // batch)  # val set fixo 10K
    flops_per_ind = flops_train + flops_eval
    total = n_gens_run * n_pop * flops_per_ind
    return round(total / 1e9, 2)  # GFLOPs


# ── VRAM ──────────────────────────────────────────────────────────────────────
def reset_vram():
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

def peak_vram_mb():
    if DEVICE == "cuda":
        return round(torch.cuda.max_memory_allocated() / 1e6, 1)
    return None


# ── Fase 1: NAS CIFAR-10 ─────────────────────────────────────────────────────
def run_cifar_nas(ds) -> dict:
    section(f"CIFAR-10 NAS: dNATY vs RandomNAS | {N_GEN} gens | pop={N_POP} | {SUBSET//1000}K samples")

    init_model = DynamicCNN(
        conv_configs=[
            {"type": "conv", "in_ch": 3,  "out_ch": 32, "stride": 1},
            {"type": "conv", "in_ch": 32, "out_ch": 64, "stride": 2},
            {"type": "conv", "in_ch": 64, "out_ch": 128, "stride": 2},
        ],
        fc_sizes=[256, 128],
        n_classes=10,
    )
    init_flops  = init_model.count_flops()
    init_params = init_model.count_params()
    info(f"Arquitetura inicial: DynamicCNN 3->32->64->128 + FC[256,128]  FLOPs={init_flops:,}  params={init_params:,}")

    results = {}
    for label, cls in [("dNATY_CNN", CnnEvolver), ("RandomNAS_CNN", RandomSearchCNN)]:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        print(f"\n  [{label}]", flush=True)

        evolver = cls(
            n_pop=N_POP, n_generations=N_GEN, t_local=T_LOCAL,
            input_size=3*32*32, n_classes=10,
            batch_size=BATCH, device=DEVICE, verbose=False,
            lambda2=LAMBDA2,
        )

        acc_curve = []
        reset_vram()

        def _cb(log, lbl=label):
            sym = ">" if "dNATY" in lbl else " "
            print(f"  {sym} Gen {log.gen:3d}/{N_GEN}  acc={log.best_acc:.4f}  params={log.n_params:,}", flush=True)
            acc_curve.append(log.best_acc)

        t0 = time.perf_counter()
        best, history = evolver.run(ds, ds, progress_callback=_cb)
        elapsed = time.perf_counter() - t0
        vram = peak_vram_mb()

        pop = evolver.population
        threshold = best.acc - 0.005
        candidates = [ind for ind in pop if ind.acc >= threshold]
        best_eff = min(candidates or [best], key=lambda ind: ind.count_flops())

        n_gens_run = len(history)
        search_gflops = estimate_search_gflops(n_gens_run, N_POP, init_flops, SUBSET, BATCH, T_LOCAL)

        reduction = (1 - best_eff.count_flops() / init_flops) * 100

        print(f"\n  [{label}]  max-acc={best.acc:.4f}  params={best.count_params():,}")
        print(f"  [{label}]  Pareto-eff acc={best_eff.acc:.4f}  FLOPs={best_eff.count_flops():,}  -{reduction:.1f}%")
        print(f"  [{label}]  search={search_gflops:.1f} GFLOPs  tempo={elapsed:.0f}s" +
              (f"  VRAM={vram:.0f}MB" if vram else "  (CPU)"))

        results[label] = {
            "acc":           round(best.acc, 4),
            "acc_eff":       round(best_eff.acc, 4),
            "params":        best.count_params(),
            "flops":         best.count_flops(),
            "flops_eff":     best_eff.count_flops(),
            "reduction_pct": round(reduction, 1),
            "search_gflops": search_gflops,
            "vram_mb":       vram,
            "time_s":        round(elapsed, 1),
            "n_gens_run":    n_gens_run,
            "acc_curve":     [round(a, 4) for a in acc_curve],
        }

    d = results["dNATY_CNN"]
    r = results["RandomNAS_CNN"]
    delta = d["acc"] - r["acc"]

    # Speedup at best acc dNATY achieves
    target = round(d["acc"] - 0.001, 4)
    g_d = next((g+1 for g,a in enumerate(d["acc_curve"]) if a >= target), None)
    g_r = next((g+1 for g,a in enumerate(r["acc_curve"]) if a >= target), None)
    speedup = round(g_r/g_d, 1) if (g_d and g_r) else None

    print()
    if delta > 0:
        ok(f"dNATY CNN acc={d['acc']:.4f} > RandomNAS {r['acc']:.4f}  (+{delta:.4f} pp)")
    else:
        info(f"dNATY CNN acc={d['acc']:.4f} vs RandomNAS {r['acc']:.4f}  ({delta:+.4f} pp)")

    if speedup and speedup >= 1.5:
        ok(f"Speedup: {speedup}x  (dNATY gen {g_d} vs RandomNAS gen {g_r})")
    elif speedup:
        info(f"Speedup: {speedup}x  (dNATY gen {g_d} vs RandomNAS gen {g_r})")

    info(f"Search cost dNATY: {d['search_gflops']:.1f} GFLOPs  ({d['time_s']:.0f}s)")

    return {"init_flops": init_flops, "init_params": init_params,
            "target_for_speedup": target, "speedup": speedup,
            **results}


# ── Fase 2: ResNet-8 baseline ─────────────────────────────────────────────────
def run_resnet_baseline(ds) -> dict:
    section(f"ResNet-8 baseline | {N_GEN * T_LOCAL} epochs | {SUBSET//1000}K samples")
    # Match compute: ResNet trains for N_GEN×T_LOCAL epochs (same as total dNATY local epochs)
    n_epochs_resnet = N_GEN * T_LOCAL
    reset_vram()
    t0 = time.perf_counter()
    acc, params = train_resnet(ds, n_epochs=n_epochs_resnet, device=DEVICE, seed=SEED)
    elapsed = time.perf_counter() - t0
    vram = peak_vram_mb()
    info(f"ResNet-8  acc={acc:.4f}  params={params:,}  tempo={elapsed:.0f}s" +
         (f"  VRAM={vram:.0f}MB" if vram else "  (CPU)"))
    return {"acc": round(acc, 4), "params": params, "time_s": round(elapsed, 1), "vram_mb": vram,
            "n_epochs": n_epochs_resnet}


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(nas: dict, resnet: dict) -> None:
    banner("CIFAR-10 PROVA -- dNATY v5.1")

    d = nas["dNATY_CNN"]
    r = nas["RandomNAS_CNN"]
    delta = d["acc"] - r["acc"]
    speedup = nas.get("speedup")

    print(f"\n  [NAS CIFAR-10 -- {N_GEN} gens, pop={N_POP}, {SUBSET//1000}K samples]")
    m = "[OK]" if delta > 0 else "[--]"
    print(f"  {m} Acc:       {d['acc']:.4f} (dNATY)  vs  {r['acc']:.4f} (RandomNAS)  ({delta:+.4f} pp)")
    if speedup:
        m = "[OK]" if speedup >= 1.5 else "[--]"
        print(f"  {m} Speedup:   {speedup}x  (gen {nas.get('g_d','?')} vs gen {nas.get('g_r','?')})")

    m = "[OK]" if d["reduction_pct"] >= 10 else "[--]"
    print(f"  {m} FLOPs:    init {nas['init_flops']:,} -> Pareto-eff {d['flops_eff']:,}  (-{d['reduction_pct']:.1f}%)")
    print(f"  ... Search:   {d['search_gflops']:.1f} GFLOPs  ({d['time_s']:.0f}s)")
    if d["vram_mb"]:
        print(f"  ... VRAM:     {d['vram_mb']:.0f} MB pico")
    else:
        print(f"  ... VRAM:     N/A (CPU run -- re-rodar com GPU para medir)")

    print(f"\n  [Baseline ResNet-8 fixo -- {resnet['n_epochs']} epochs, {SUBSET//1000}K samples]")
    diff = d["acc"] - resnet["acc"]
    m = "[OK]" if diff >= -0.01 else "[--]"
    print(f"  {m} Acc:       {d['acc']:.4f} (dNATY)  vs  {resnet['acc']:.4f} (ResNet-8)  ({diff:+.4f} pp)")

    print(f"\n{'='*W}")
    all_pass = delta > 0 and d["reduction_pct"] >= 10
    if all_pass:
        print("  CIFAR-10 NAS VERIFICADO [OK]")
    else:
        issues = []
        if delta <= 0: issues.append("dNATY <= RandomNAS acc")
        if d["reduction_pct"] < 10: issues.append(f"FLOPs {d['reduction_pct']:.1f}% < 10%")
        print(f"  Pendente: {' | '.join(issues)}")
    print(f"{'='*W}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    banner(
        f"dNATY cifar_proof.py\n"
        f"  {N_GEN} gens | pop={N_POP} | CIFAR-10 {SUBSET//1000}K | {DEVICE.upper()}"
    )

    t_total = time.time()

    print("\n[Carregando CIFAR-10 ...]", flush=True)
    ds = FastDataset("CIFAR10", device=DEVICE, train_subset=SUBSET)

    nas_out    = run_cifar_nas(ds)
    resnet_out = run_resnet_baseline(ds)

    print_summary(nas_out, resnet_out)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    output = {
        "timestamp":  datetime.now().isoformat(),
        "config": {"device": DEVICE, "n_gen": N_GEN, "n_pop": N_POP, "subset": SUBSET},
        "nas":    nas_out,
        "resnet": resnet_out,
    }
    json_path = out_dir / "cifar_proof_results.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    info(f"Resultados: {json_path}  ({time.time()-t_total:.0f}s total)")


if __name__ == "__main__":
    main()
