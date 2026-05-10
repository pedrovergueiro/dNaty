"""Compatibility wrapper for the tracked dNaty experiment runner.

Preferred usage:
    python -m experiments.run --profile smoke
    python -m experiments.run --profile prevalidation --experiment exp2_cifar
"""
from experiments.run import main


if __name__ == "__main__":
    main()
