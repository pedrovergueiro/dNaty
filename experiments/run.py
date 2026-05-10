"""Versioned experiment runner for dNaty.

Examples:
    python -m experiments.run --profile smoke
    python -m experiments.run --profile prevalidation --experiment exp2_cifar
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from dnaty.tracking import build_manifest, copy_existing, create_run_dir, repo_root, write_json


PROFILE_CONFIGS = {
    "smoke": "experiments/configs/smoke.json",
    "prevalidation": "experiments/configs/prevalidation.json",
    "full_gpu": "experiments/configs/full_gpu.json",
}

EXPERIMENT_OUTPUTS = {
    "exp1_mnist": ["results/exp1_results.json"],
    "exp2_cifar": ["results/exp2_cifar10_results.json"],
    "exp3_cl": ["results/exp3_cl_results.json"],
}


def load_config(profile: str) -> dict[str, Any]:
    path = repo_root() / PROFILE_CONFIGS[profile]
    return json.loads(path.read_text(encoding="utf-8"))


def run_smoke_test(name: str) -> None:
    script = {
        "sanity": "tests/test_sanity.py",
        "exp23": "tests/test_exp23.py",
    }[name]
    subprocess.run([sys.executable, script], cwd=repo_root(), check=True)


def apply_cifar_config(config: dict[str, Any]) -> None:
    cifar_cfg = config.get("cifar")
    if not cifar_cfg:
        return
    module = importlib.import_module("dnaty.experiments.exp2_cifar")
    module.SEEDS = cifar_cfg["seeds"]
    module.N_GENERATIONS = cifar_cfg["n_generations"]
    module.N_POP = cifar_cfg["n_pop"]
    module.T_LOCAL = cifar_cfg["t_local"]
    module.BATCH_SIZE = cifar_cfg["batch_size"]
    module.CIFAR_TRAIN_SUBSET = cifar_cfg["train_subset"]


def run_experiment(name: str, config: dict[str, Any]) -> list[str]:
    if name in ("sanity", "exp23"):
        run_smoke_test(name)
        return []

    if name == "exp2_cifar":
        apply_cifar_config(config)

    module_name = {
        "exp1_mnist": "dnaty.experiments.exp1_mnist",
        "exp2_cifar": "dnaty.experiments.exp2_cifar",
        "exp3_cl": "dnaty.experiments.exp3_cl",
    }[name]
    module = importlib.import_module(module_name)
    module.main()
    return EXPERIMENT_OUTPUTS.get(name, [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dNaty experiments with tracking")
    parser.add_argument("--profile", choices=PROFILE_CONFIGS, default="smoke")
    parser.add_argument(
        "--experiment",
        choices=["sanity", "exp23", "exp1_mnist", "exp2_cifar", "exp3_cl"],
        action="append",
        help="Override experiments from the profile. Can be passed more than once.",
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    config = load_config(args.profile)
    experiments = args.experiment or config["experiments"]
    run_id = f"{args.profile}_{'_'.join(experiments)}"
    run_dir = create_run_dir(run_id)
    write_json(run_dir / "config.json", config)

    outputs: list[str] = []
    status = "completed"
    try:
        for experiment in experiments:
            outputs.extend(run_experiment(experiment, config))
    except Exception:
        status = "failed"
        manifest = build_manifest(
            experiment_id=run_id,
            config=config,
            command=sys.argv,
            outputs=[],
            status=status,
            notes=args.notes,
        )
        write_json(run_dir / "manifest.json", manifest)
        raise

    copied = copy_existing(outputs, run_dir / "outputs")
    manifest = build_manifest(
        experiment_id=run_id,
        config=config,
        command=sys.argv,
        outputs=copied,
        status=status,
        notes=args.notes,
    )
    write_json(run_dir / "manifest.json", manifest)
    print(f"\n[OK] Run saved at: {run_dir.relative_to(repo_root())}")


if __name__ == "__main__":
    main()
