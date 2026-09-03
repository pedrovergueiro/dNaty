"""
Bounded replay buffer for lifelong on-device adaptation (v2.2.0).

Reservoir sampling over the incoming data stream: every sample ever seen has
the same probability of being in the buffer, regardless of when it arrived —
so replaying from it approximates replaying from the full history at O(capacity)
memory. This is what lets `CompressResult.adapt()` learn from drifted data
without forgetting the old distribution.
"""
from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-capacity reservoir over (x, y) samples.

    add_batch(x, y) streams samples in; sample(n) draws a uniform random
    subset without replacement. state_dict()/load_state_dict() persist the
    buffer (tensors only, torch.save-compatible).
    """

    def __init__(self, capacity: int = 2048, seed: int | None = None):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._x: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._size = 0
        self._seen = 0

    def __len__(self) -> int:
        return self._size

    @property
    def seen(self) -> int:
        """Total samples ever streamed through the buffer."""
        return self._seen

    def _ensure_storage(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if self._x is None:
            self._x = torch.empty((self.capacity, *x.shape[1:]), dtype=x.dtype)
            self._y = torch.empty((self.capacity, *y.shape[1:]), dtype=y.dtype)

    def add_batch(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Stream a batch of samples into the reservoir."""
        if len(x) != len(y):
            raise ValueError(f"x and y disagree on batch size: {len(x)} vs {len(y)}")
        if len(x) == 0:
            return
        x = x.detach().cpu()
        y = y.detach().cpu()
        self._ensure_storage(x, y)
        for i in range(len(x)):
            if self._size < self.capacity:
                slot = self._size
                self._size += 1
            else:
                # Classic reservoir: keep with probability capacity / (seen + 1).
                j = int(self._rng.integers(0, self._seen + 1))
                if j >= self.capacity:
                    self._seen += 1
                    continue
                slot = j
            self._x[slot] = x[i]
            self._y[slot] = y[i]
            self._seen += 1

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform random subset of up to n stored samples (no replacement)."""
        if self._size == 0:
            raise ValueError("cannot sample from an empty ReplayBuffer")
        k = min(int(n), self._size)
        idx = self._rng.choice(self._size, size=k, replace=False)
        idx = torch.as_tensor(idx, dtype=torch.long)
        return self._x[idx].clone(), self._y[idx].clone()

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": self._size,
            "seen": self._seen,
            "x": None if self._x is None else self._x[: self._size].clone(),
            "y": None if self._y is None else self._y[: self._size].clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state["capacity"])
        self._size = int(state["size"])
        self._seen = int(state["seen"])
        if state["x"] is None:
            self._x = self._y = None
            return
        x, y = state["x"], state["y"]
        self._x = torch.empty((self.capacity, *x.shape[1:]), dtype=x.dtype)
        self._y = torch.empty((self.capacity, *y.shape[1:]), dtype=y.dtype)
        self._x[: self._size] = x
        self._y[: self._size] = y

    def __repr__(self) -> str:
        return (f"ReplayBuffer(size={self._size}/{self.capacity}, "
                f"seen={self._seen})")
