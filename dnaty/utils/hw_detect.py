"""
Hardware detection and latency scaling tables.

Detects the current CPU architecture and provides scaling factors
to estimate latency on other target devices from x86 measurements.

Pre-calibrated scaling factors (x86_64 desktop baseline = 1.0):
  Device            Scale   Notes
  ─────────────────────────────────────────────────────
  x86_64 desktop    1.0x    i5/i7/Ryzen baseline
  x86_64 server     0.7x    typically faster per-core
  Raspberry Pi 4    9.0x    ARM Cortex-A72 @ 1.8GHz
  Raspberry Pi 5    4.5x    ARM Cortex-A76 @ 2.4GHz
  Jetson Nano       5.0x    ARM Cortex-A57 (CPU-only)
  Orange Pi 5       3.5x    ARM Cortex-A76 @ 2.4GHz
  Apple M1          0.6x    fast ARM, good single-thread

These are conservative median values measured from MLP inference
benchmarks. Actual values vary ±30% based on model size and memory.
"""
from __future__ import annotations

import os
import platform
import subprocess


_SCALE_TABLE: dict[str, float] = {
    "x86_64":   1.0,
    "amd64":    1.0,
    "x86":      1.4,
    "arm64":    1.8,    # generic ARM64 (unknown board)
    "aarch64":  1.8,
    "rpi4":     9.0,
    "rpi5":     4.5,
    "jetson_nano": 5.0,
    "orange_pi5": 3.5,
    "apple_m1": 0.6,
    "apple_m2": 0.5,
    "cpu":      1.0,    # alias → use measured x86
}

_DEVICE_ALIASES: dict[str, str] = {
    "rpi4":        "rpi4",
    "raspberry_pi_4": "rpi4",
    "raspberrypi4": "rpi4",
    "rpi5":        "rpi5",
    "raspberry_pi_5": "rpi5",
    "jetson":      "jetson_nano",
    "jetson_nano": "jetson_nano",
    "orange_pi":   "orange_pi5",
    "apple_m1":    "apple_m1",
    "m1":          "apple_m1",
    "apple_m2":    "apple_m2",
    "m2":          "apple_m2",
}


def detect_hw() -> dict:
    """
    Returns current hardware info dict:
      {"arch": str, "device_class": str, "cores": int, "scale_vs_x86": float}
    """
    arch = platform.machine().lower()
    cores = os.cpu_count() or 1

    # Detect RPi by /proc/cpuinfo model name (Linux only)
    device_class = _canonical_arch(arch)
    try:
        with open("/proc/cpuinfo") as f:
            info = f.read().lower()
        if "raspberry pi 5" in info:
            device_class = "rpi5"
        elif "raspberry pi 4" in info or "cortex-a72" in info:
            device_class = "rpi4"
        elif "jetson" in info or "cortex-a57" in info:
            device_class = "jetson_nano"
    except (FileNotFoundError, PermissionError):
        pass

    scale = _SCALE_TABLE.get(device_class, 1.0)
    return {"arch": arch, "device_class": device_class,
            "cores": cores, "scale_vs_x86": scale}


def latency_scale(device: str) -> float:
    """
    Return scaling factor from x86 baseline for a target device string.

    Examples:
        latency_scale("rpi4")   → 9.0
        latency_scale("cpu")    → 1.0
        latency_scale("x86_64") → 1.0
    """
    key = _DEVICE_ALIASES.get(device.lower(), device.lower())
    return _SCALE_TABLE.get(key, 1.0)


def estimate_latency(measured_ms: float, from_device: str = "cpu",
                     to_device: str = "rpi4") -> float:
    """
    Estimate latency on target device from a measurement on source device.

    estimate_latency(measured_ms=2.1, from_device="cpu", to_device="rpi4")
    → ~18.9ms (9x scale)
    """
    src_scale = latency_scale(from_device)
    dst_scale = latency_scale(to_device)
    return measured_ms * (dst_scale / src_scale)


def _canonical_arch(arch: str) -> str:
    if arch in ("amd64", "x86_64"):
        return "x86_64"
    if arch in ("arm64", "aarch64"):
        return "aarch64"
    return arch
