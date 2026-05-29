#!/usr/bin/env python3
"""
dNATY v5.1 -- Treinamento completo em um comando.

Roda tudo em sequencia e mostra cada claim ao vivo:
  1. NAS: dNATY (memory-guided) vs RandomNAS (ablation)
     -> prova que memoria episodica guia melhor que random
     -> prova compressao de FLOPs via NSGA-II
  2. CL:  Split-MNIST com 5 tasks sequenciais
     -> prova BWT ~= 0 vs EWC que esquece muito

Modos:
  python scripts/train.py           # ~8 min CPU (20 gens, 10K samples)
  python scripts/train.py --quick   # ~3 min CPU (15 gens, 10K samples)
  python scripts/train.py --full    # ~45 min CPU (50 gens, 60K samples)
  python scripts/train.py --nas     # so a busca de arquitetura
  python scripts/train.py --cl      # so o experimento CL
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from dnaty.experiments.fast_dataset import FastDataset
from dnaty.evolution.evolver import DnatyEvolver
from dnaty.operators.mutations import OPERATORS, apply_operator

# --- Args ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="dNATY -- treinamento completo")
parser.add_argument("--quick",  action="store_true", help="15 gens, 10K (~3 min)")
parser.add_argument("--full",   action="store_true", help="50 gens, 60K (~45 min)")
parser.add_argument("--nas",    action="store_true", help="so NAS (sem CL)")
parser.add_argument("--cl",     action="store_true", help="so CL (sem NAS)")
args = parser.parse_args()

# --- Config -------------------------------------------------------------------
if args.quick:
    N_POP, N_GEN, SUBSET = 8, 15, 10_000
elif args.full:
    N_POP, N_GEN, SUBSET = 20, 50, 60_000
else:
    N_POP, N_GEN, SUBSET = 10, 20, 10_000

T_LOCAL = 3
BATCH   = 512
SEED    = 42
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"

RUN_NAS = not args.cl
RUN_CL  = not args.nas

W = 62  # largura do banner


# --- Utilitarios visuais ------------------------------------------------------
def banner(title: str) -> None:
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")


def section(title: str) -> None:
    print(f"\n{'-'*W}")
    print(f"  {title}")
    print(f"{'-'*W}")


def ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def info(msg: str) -> None:
    print(f"  ... {msg}")


# --- RandomNAS (ablation) -----------------------------------------------------
class RandomSearchNAS(DnatyEvolver):
    def _mutate_population(self, population):
        mutated = []
        for ind in population:
            op = np.random.choice(OPERATORS)
            new_ind, success = apply_operator(ind, op)
            if not success or not new_ind.model.is_valid():
                new_ind = ind.clone()
                new_ind.last_op = "no_op"
            mutated.append(new_ind)
        return mutated

    def _update_memory(self, *a, **kw):
        return 0.0


# Init grande o suficiente para NSGA-II ter espaco para comprimir
INIT_HIDDEN = [512, 256]


# --- Fase 1: NAS --------------------------------------------------------------
def run_nas(ds) -> dict:
    section("Fase 1/2 -- Architecture Search: dNATY vs RandomNAS")

    init_ind    = DnatyEvolver(input_size=784, n_classes=10,
                                init_hidden=INIT_HIDDEN, device=DEVICE)._make_individual()
    init_flops  = init_ind.count_flops()
    init_params = init_ind.count_params()
    info(f"Arquitetura inicial: MLP{[784] + INIT_HIDDEN} -> 10 classes  (grande p/ NSGA-II comprimir)")
    info(f"Params: {init_params:,}   FLOPs: {init_flops:,}")

    results = {}
    evolvers = {}
    for label, cls in [("dNATY (memory-guided)", DnatyEvolver),
                        ("RandomNAS (ablation)", RandomSearchNAS)]:
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        print(f"\n  [{label}]", flush=True)

        evolver = cls(
            n_pop=N_POP, n_generations=N_GEN, t_local=T_LOCAL,
            input_size=784, n_classes=10, init_hidden=INIT_HIDDEN,
            batch_size=BATCH, device=DEVICE, verbose=False,
        )

        def _cb(log, lbl=label):
            sym = ">" if "dNATY" in lbl else " "
            print(
                f"  {sym} Gen {log.gen:3d}/{N_GEN}"
                f"  acc={log.best_acc:.4f}"
                f"  params={log.n_params:,}",
                flush=True,
            )

        tracemalloc.start()
        t0 = time.perf_counter()
        best, history = evolver.run(ds, ds, progress_callback=_cb)
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        evolvers[label] = evolver

        # Pareto-optimal efficient solution:
        # melhor individuo com acc >= best_acc - 0.005 e minimos FLOPs
        # (NSGA-II mantem a frente de Pareto na populacao)
        pop = evolver.population
        threshold = best.acc - 0.005
        candidates = [ind for ind in pop if ind.acc >= threshold]
        best_eff = min(candidates or [best], key=lambda ind: ind.count_flops())

        flops_ratio_best = best.count_flops() / init_flops
        flops_ratio_eff  = best_eff.count_flops() / init_flops

        print(
            f"\n  Resultado [{label}]:"
            f"\n    max-acc:    acc={best.acc:.4f}  FLOPs={best.count_flops():,} ({(flops_ratio_best-1)*100:+.1f}%)"
            f"\n    Pareto-eff: acc={best_eff.acc:.4f}  FLOPs={best_eff.count_flops():,} ({(flops_ratio_eff-1)*100:+.1f}%)  tempo={elapsed:.0f}s"
        )

        acc_curve = [log.best_acc for log in history]
        results[label] = {
            "acc":              round(best.acc, 4),
            "acc_eff":          round(best_eff.acc, 4),
            "params":           best.count_params(),
            "flops":            best.count_flops(),
            "flops_eff":        best_eff.count_flops(),
            "flops_ratio":      round(flops_ratio_best, 4),
            "flops_ratio_eff":  round(flops_ratio_eff, 4),
            "time_s":           round(elapsed, 1),
            "peak_mb":          round(peak_bytes / 1e6, 1),
            "acc_curve":        [round(a, 4) for a in acc_curve],
            "arch":             getattr(best.model, "layer_sizes", []),
            "arch_eff":         getattr(best_eff.model, "layer_sizes", []),
        }

    dnaty  = results["dNATY (memory-guided)"]
    random = results["RandomNAS (ablation)"]
    delta_acc      = dnaty["acc"] - random["acc"]
    flops_eff_pct  = (dnaty["flops_ratio_eff"] - 1) * 100

    print()
    if delta_acc > 0:
        ok(f"dNATY supera RandomNAS em +{delta_acc:.4f} pp (max-acc)")
    else:
        info(f"dNATY vs RandomNAS: {delta_acc:+.4f} pp  (mais gens = mais sinal da memoria)")

    if flops_eff_pct < 0:
        ok(f"Pareto-eff: FLOPs {flops_eff_pct:.1f}% vs init -- NSGA-II comprimiu a arquitetura")
    else:
        info(f"Pareto-eff FLOPs: {flops_eff_pct:+.1f}%  (mais gens = mais compressao)")

    return {"init_flops": init_flops, "init_params": init_params, **results}


# --- Fase 2: CL ---------------------------------------------------------------
def run_cl() -> dict:
    section("Fase 2/2 -- Continual Learning: Split-MNIST")
    import importlib, sys as _sys
    # usa o experimento existente mas captura os resultados
    from dnaty.experiments.exp3_cl import (
        run_dnaty_cl_seed, run_ewc_cl_seed, run_mlp_cl_seed,
        N_TASKS,
    )

    seeds = [0, 1, 2] if not args.quick else [0]
    dnaty_res, ewc_res, mlp_res = [], [], []

    for seed in seeds:
        print(f"\n  Seed {seed}")

        t0 = time.time()
        info("dNATY CL (evolver + warm-start + replay) ?")
        dr = run_dnaty_cl_seed(seed, DEVICE)
        dnaty_res.append(dr)
        print(f"    BWT={dr['metrics']['BWT']:.4f}  {time.time()-t0:.0f}s")

        t0 = time.time()
        info("EWC baseline ?")
        er = run_ewc_cl_seed(seed, DEVICE)
        ewc_res.append(er)
        print(f"    BWT={er['metrics']['BWT']:.4f}  {time.time()-t0:.0f}s")

        t0 = time.time()
        info("MLP sem CL (baseline) ?")
        mr = run_mlp_cl_seed(seed, DEVICE)
        mlp_res.append(mr)
        print(f"    BWT={mr['metrics']['BWT']:.4f}  {time.time()-t0:.0f}s")

    def mean_bwt(res): return float(np.mean([r["metrics"]["BWT"] for r in res]))

    d_bwt = mean_bwt(dnaty_res)
    e_bwt = mean_bwt(ewc_res)
    m_bwt = mean_bwt(mlp_res)

    ratio = abs(e_bwt / d_bwt) if abs(d_bwt) > 1e-6 else float("inf")

    print()
    ok(f"dNATY CL  BWT = {d_bwt:.4f}")
    info(f"EWC       BWT = {e_bwt:.4f}")
    info(f"MLP noCL  BWT = {m_bwt:.4f}")

    if ratio >= 4:
        ok(f"dNATY esquece {ratio:.1f}x menos que EWC")
    else:
        info(f"dNATY vs EWC: {ratio:.1f}x menos esquecimento")

    return {
        "dnaty_bwt": round(d_bwt, 4),
        "ewc_bwt":   round(e_bwt, 4),
        "mlp_bwt":   round(m_bwt, 4),
        "ratio":     round(ratio, 1),
        "seeds":     seeds,
    }


# --- Sumario final ------------------------------------------------------------
def print_summary(nas_out: dict | None, cl_out: dict | None) -> None:
    banner("RESUMO FINAL -- dNATY v1.0")

    if nas_out:
        dnaty  = nas_out["dNATY (memory-guided)"]
        random = nas_out["RandomNAS (ablation)"]
        delta     = dnaty["acc"] - random["acc"]
        fd_eff    = (dnaty["flops_ratio_eff"] - 1) * 100

        print(f"\n  [NAS -- Architecture Search]")
        marker_acc  = "[OK]" if delta  > 0 else "[--]"
        marker_flop = "[OK]" if fd_eff < 0 else "[--]"
        print(f"  {marker_acc} Acuracia:  {dnaty['acc']:.4f} (dNATY)  vs  {random['acc']:.4f} (Random)  ->  {delta:+.4f} pp")
        print(f"  {marker_flop} FLOPs:     {nas_out['init_flops']:,} -> {dnaty['flops_eff']:,}  ({fd_eff:+.1f}%)  [Pareto-eficiente]")

    if cl_out:
        ratio = cl_out["ratio"]
        marker = "[OK]" if ratio >= 4 else "[--]"
        print(f"\n  [CL -- Continual Learning]")
        print(f"  {marker} Esquecimento (BWT): {cl_out['dnaty_bwt']:.4f} (dNATY)  vs  {cl_out['ewc_bwt']:.4f} (EWC)  ->  {ratio:.0f}x melhor")

    print(f"\n{'='*W}")

    if not (nas_out and cl_out):
        return

    all_pass = (
        (nas_out["dNATY (memory-guided)"]["acc"] > nas_out["RandomNAS (ablation)"]["acc"])
        and (nas_out["dNATY (memory-guided)"]["flops_ratio_eff"] < 1.0)
        and (cl_out["ratio"] >= 4)
    )
    if all_pass:
        print("  TODOS OS CLAIMS VERIFICADOS [OK]")
    else:
        print("  Resultado parcial -- aumente --full para convergencia completa")
    print(f"{'='*W}\n")


# --- Main ---------------------------------------------------------------------
def main() -> None:
    mode = "QUICK" if args.quick else ("FULL" if args.full else "DEFAULT")
    banner(
        f"dNATY v5.1 -- Treinamento Completo\n"
        f"  MNIST {SUBSET:,} samples | {N_GEN} gens | pop={N_POP} | {DEVICE.upper()} [{mode}]"
    )

    t_total = time.time()
    nas_out = cl_out = None

    if RUN_NAS:
        print("\n[Carregando dataset ...]", flush=True)
        ds = FastDataset("MNIST", device=DEVICE, train_subset=SUBSET)
        nas_out = run_nas(ds)

    if RUN_CL:
        cl_out = run_cl()

    print_summary(nas_out, cl_out)

    # Salva JSON
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "device": DEVICE, "n_pop": N_POP, "n_gen": N_GEN,
            "subset": SUBSET, "seed": SEED, "mode": mode,
        },
        "nas": nas_out,
        "cl":  cl_out,
    }
    out_path = Path("results/train_results.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    info(f"Resultados em: {out_path}  ({time.time()-t_total:.0f}s total)")


if __name__ == "__main__":
    main()
