"""Download/cache datasets needed by dNaty experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import torchvision
import torchvision.transforms as T


def prepare(data_dir: str = "data", datasets: list[str] | None = None) -> None:
    requested = set(datasets or ["mnist", "fashionmnist", "cifar10"])
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    if "mnist" in requested:
        torchvision.datasets.MNIST(
            root, train=True, download=True, transform=T.ToTensor()
        )
        torchvision.datasets.MNIST(
            root, train=False, download=True, transform=T.ToTensor()
        )
        print("[OK] MNIST cached")

    if "fashionmnist" in requested:
        torchvision.datasets.FashionMNIST(
            root, train=True, download=True, transform=T.ToTensor()
        )
        torchvision.datasets.FashionMNIST(
            root, train=False, download=True, transform=T.ToTensor()
        )
        print("[OK] FashionMNIST cached")

    if "cifar10" in requested:
        torchvision.datasets.CIFAR10(
            root, train=True, download=True, transform=T.ToTensor()
        )
        torchvision.datasets.CIFAR10(
            root, train=False, download=True, transform=T.ToTensor()
        )
        print("[OK] CIFAR-10 cached")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dNaty datasets")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["mnist", "fashionmnist", "cifar10"],
        default=["mnist", "fashionmnist", "cifar10"],
    )
    args = parser.parse_args()
    prepare(args.data_dir, args.datasets)


if __name__ == "__main__":
    main()
