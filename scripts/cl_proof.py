#!/usr/bin/env python3
"""
dNATY cl_proof.py -- prova completa de Continual Learning.

Roda em sequencia:
  1. Split-MNIST: dNATY vs EWC vs DER++ vs MLP
  2. Permuted-MNIST: dNATY vs EWC

Uso:
  python scripts/cl_proof.py           # 3 seeds, tudo
  python scripts/cl_proof.py --quick   # 1 seed
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from dnaty.experiments.exp3_cl import (
    run_dnaty_cl_seed, run_ewc_cl_seed, run_mlp_cl_seed,
    run_derpp_cl_seed,
    run_dnaty_permuted_seed, run_ewc_permuted_seed,
    N_TASKS,
)

parser = argparse.ArgumentParser()
parser.add_argument("--quick", action="store_true", help="1 seed")
args = parser.parse_args()

SEEDS  = [0] if args.quick else [0, 1, 2]
DEVICE = "cuda"
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

W = 64
def banner(t): print(f"\n{'='*W}\n  {t}\n{'='*W}")
def section(t): print(f"\n{'-'*W}\n  {t}\n{'-'*W}")
def ok(m): print(f"  [OK] {m}")
def info(m): print(f"  ... {m}")


# ── Split-MNIST ───────────────────────────────────────────────────────────────
def run_split_mnist() -> dict:
    section(f"Split-MNIST | 5 tasks | {len(SEEDS)} seed(s)")

    methods = {
        "dNATY":  (run_dnaty_cl_seed,  []),
        "DER++":  (run_derpp_cl_seed,  []),
        "EWC":    (run_ewc_cl_seed,    []),
        "MLP":    (run_mlp_cl_seed,    []),
    }

    for seed in SEEDS:
        print(f"\n  Seed {seed}")
        for name, (fn, results) in methods.items():
            t0 = time.time()
            info(f"{name} ...")
            r  = fn(seed, DEVICE)
            results.append(r)
            print(f"    {name:8s} BWT={r['metrics']['BWT']:.4f}  {time.time()-t0:.0f}s")

    def mbwt(res): return float(np.mean([r["metrics"]["BWT"] for r in res]))

    summary = {name: round(mbwt(data), 4) for name, (_, data) in methods.items()}

    print()
    best = summary["dNATY"]
    for name, bwt in summary.items():
        ratio = abs(bwt / best) if abs(best) > 1e-6 else float("inf")
        marker = "[OK]" if (name == "dNATY" or bwt < best) else "    "
        vs = f" ({ratio:.1f}x pior)" if name != "dNATY" else ""
        print(f"  {marker} {name:8s}  BWT = {bwt:.4f}{vs}")

    ratio_ewc = abs(summary["EWC"] / best) if abs(best) > 1e-6 else float("inf")
    ratio_der = abs(summary["DER++"] / best) if abs(best) > 1e-6 else float("inf")
    if ratio_ewc >= 5:
        ok(f"dNATY {ratio_ewc:.1f}x melhor que EWC")
    if ratio_der >= 1.5:
        ok(f"dNATY {ratio_der:.1f}x melhor que DER++")
    else:
        info(f"dNATY vs DER++: {ratio_der:.2f}x  (DER++ e um baseline forte)")

    return {"summary": summary, "ratio_ewc": round(ratio_ewc, 1), "ratio_derpp": round(ratio_der, 2)}


# ── Permuted-MNIST ────────────────────────────────────────────────────────────
def run_permuted() -> dict:
    section(f"Permuted-MNIST | 5 tasks | {len(SEEDS)} seed(s)")

    dnaty_res, ewc_res = [], []

    for seed in SEEDS:
        print(f"\n  Seed {seed}")
        t0 = time.time()
        info("dNATY (replay balanceado) ...")
        dr = run_dnaty_permuted_seed(seed, DEVICE)
        dnaty_res.append(dr)
        print(f"    dNATY   BWT={dr['BWT']:.4f}  {time.time()-t0:.0f}s")

        t0 = time.time()
        info("EWC ...")
        er = run_ewc_permuted_seed(seed, DEVICE)
        ewc_res.append(er)
        print(f"    EWC     BWT={er['BWT']:.4f}  {time.time()-t0:.0f}s")

    d_bwt = float(np.mean([r["BWT"] for r in dnaty_res]))
    e_bwt = float(np.mean([r["BWT"] for r in ewc_res]))
    ratio = abs(e_bwt / d_bwt) if abs(d_bwt) > 1e-6 else float("inf")

    print()
    ok(f"dNATY  BWT = {d_bwt:.4f}  |  EWC  BWT = {e_bwt:.4f}  |  {ratio:.1f}x melhor")

    return {"dnaty_bwt": round(d_bwt, 4), "ewc_bwt": round(e_bwt, 4), "ratio": round(ratio, 1)}


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(split: dict, permuted: dict) -> None:
    banner("CL PROVA FINAL -- dNATY v5.1")

    print(f"\n  [Split-MNIST -- 5 tasks, {len(SEEDS)} seed(s)]")
    for name, bwt in split["summary"].items():
        m = "[OK]" if name == "dNATY" else "    "
        print(f"  {m} {name:8s}  BWT = {bwt:.4f}")
    m1 = "[OK]" if split["ratio_ewc"] >= 5 else "[--]"
    m2 = "[OK]" if split["ratio_derpp"] >= 1.2 else "[--]"
    print(f"  {m1} vs EWC:   {split['ratio_ewc']:.1f}x menos esquecimento")
    print(f"  {m2} vs DER++: {split['ratio_derpp']:.2f}x menos esquecimento")

    print(f"\n  [Permuted-MNIST -- 5 tasks, {len(SEEDS)} seed(s)]")
    m = "[OK]" if permuted["ratio"] >= 3 else "[--]"
    print(f"  {m} dNATY={permuted['dnaty_bwt']:.4f}  EWC={permuted['ewc_bwt']:.4f}  {permuted['ratio']:.1f}x melhor")

    print(f"\n{'='*W}")
    all_pass = split["ratio_ewc"] >= 5 and permuted["ratio"] >= 3
    if all_pass:
        print("  CL VERIFICADO EM DOIS BENCHMARKS [OK]")
    else:
        print("  CL parcialmente verificado")
    print(f"{'='*W}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    banner(f"dNATY cl_proof.py | {len(SEEDS)} seed(s) | {DEVICE.upper()}")
    t_total = time.time()

    split    = run_split_mnist()
    permuted = run_permuted()

    print_summary(split, permuted)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "config":    {"device": DEVICE, "seeds": SEEDS},
        "split_mnist":    split,
        "permuted_mnist": permuted,
    }
    json_path = out_dir / "cl_proof_results.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    info(f"Resultados: {json_path}  ({time.time()-t_total:.0f}s total)")


if __name__ == "__main__":
    main()
