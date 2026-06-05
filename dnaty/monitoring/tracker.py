"""
Production pipeline tracker (Inner-Image-6313, r/deeplearning).

Wraps a compressed model with:
  - Prediction logging (class distribution, confidence)
  - Automatic drift detection every N samples
  - Edge-case flagging (low-confidence + high-uncertainty predictions)
  - Failure tracking (ground-truth comparison when labels available)

Usage:
    from dnaty.monitoring import ProductionTracker

    tracker = ProductionTracker(result.model, drift_detector=detector)
    tracker.fit_baseline(train_x)

    # In your inference loop:
    pred, meta = tracker.predict(x)
    if meta["alert"]:
        print(meta["alert"])   # "DRIFT DETECTED" or "LOW CONFIDENCE"

    # Periodic report:
    print(tracker.report())
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from dnaty.monitoring.drift import DriftDetector


class ProductionTracker:
    """
    Thin wrapper that adds observability to a compressed model.

    Args:
        model:            Compressed PyTorch model (from CompressResult.model).
        drift_detector:   Pre-fitted DriftDetector, or None to auto-create.
        drift_every:      Check drift every N predictions.
        confidence_threshold: Flag predictions below this softmax confidence.
        max_history:      Max samples kept in rolling window for drift.
        device:           Inference device.
    """

    def __init__(
        self,
        model: nn.Module,
        drift_detector: Optional[DriftDetector] = None,
        drift_every: int = 500,
        confidence_threshold: float = 0.6,
        max_history: int = 2000,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.drift_detector = drift_detector or DriftDetector()
        self.drift_every = drift_every
        self.confidence_threshold = confidence_threshold

        # Rolling window for drift detection
        self._input_buffer: deque[np.ndarray] = deque(maxlen=max_history)

        # Counters
        self._n_predictions = 0
        self._n_low_confidence = 0
        self._n_drifts_detected = 0
        self._class_counts: dict[int, int] = defaultdict(int)
        self._latencies_ms: deque[float] = deque(maxlen=500)

        # Failure tracking (when labels are provided to .record_outcome)
        self._n_correct = 0
        self._n_labelled = 0

        self._baseline_fitted = False
        self._last_drift_report: Optional[dict] = None

    # ------------------------------------------------------------------
    def fit_baseline(self, data: "np.ndarray | torch.Tensor") -> "ProductionTracker":
        """Fit the drift detector on training data to establish baseline."""
        arr = _flat(data)
        self.drift_detector.fit(arr)
        self._baseline_fitted = True
        return self

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict(
        self,
        x: "np.ndarray | torch.Tensor",
    ) -> tuple[np.ndarray, dict]:
        """Run inference and return (predictions, metadata).

        Args:
            x: Input batch. Shape (N, ...) or single sample (...).

        Returns:
            predictions: int array of class indices, shape (N,).
            meta: dict with keys:
              confidences  — max softmax probability per sample
              alert        — None or alert string ("DRIFT DETECTED", "LOW CONFIDENCE")
              drift_score  — latest drift PSI (None until first check)
              latency_ms   — inference latency for this batch
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.asarray(x, dtype=np.float32))
        if x.dim() == 1 or (x.dim() > 1 and x.shape[0] == 1 and x.numel() == x.shape[-1]):
            x = x.unsqueeze(0)
        x = x.to(self.device)

        t0 = time.perf_counter()
        self.model.eval()
        logits = self.model(x)
        latency_ms = (time.perf_counter() - t0) * 1000
        self._latencies_ms.append(latency_ms)

        probs = torch.softmax(logits, dim=-1)
        confidences = probs.max(dim=-1).values.cpu().numpy()
        predictions = probs.argmax(dim=-1).cpu().numpy()

        # Update counters
        self._n_predictions += len(predictions)
        for p in predictions:
            self._class_counts[int(p)] += 1
        low_conf = int((confidences < self.confidence_threshold).sum())
        self._n_low_confidence += low_conf

        # Buffer inputs for drift detection
        flat_x = _flat(x.cpu())
        for row in flat_x:
            self._input_buffer.append(row)

        # Drift check every N samples
        alert = None
        drift_score = None
        if (self._baseline_fitted
                and self._n_predictions % self.drift_every < len(predictions)):
            buf = np.stack(list(self._input_buffer))
            report = self.drift_detector.score(buf)
            drift_score = report["psi_mean"]
            self._last_drift_report = report
            if report["drifted"]:
                self._n_drifts_detected += 1
                alert = f"DRIFT DETECTED (PSI={drift_score:.3f})"

        if alert is None and low_conf > 0:
            alert = f"LOW CONFIDENCE ({low_conf}/{len(predictions)} samples below {self.confidence_threshold:.0%})"

        return predictions, {
            "confidences": confidences,
            "alert": alert,
            "drift_score": drift_score,
            "latency_ms": latency_ms,
        }

    # ------------------------------------------------------------------
    def record_outcome(
        self,
        predictions: np.ndarray,
        ground_truth: "np.ndarray | torch.Tensor",
    ) -> float:
        """Track prediction accuracy with ground-truth labels.

        Returns:
            Accuracy for this batch.
        """
        gt = np.asarray(ground_truth)
        correct = int((predictions == gt).sum())
        self._n_correct += correct
        self._n_labelled += len(gt)
        return correct / max(len(gt), 1)

    # ------------------------------------------------------------------
    def report(self) -> dict:
        """Return a production health summary."""
        drift_info = self._last_drift_report or {}
        return {
            "total_predictions": self._n_predictions,
            "low_confidence_rate": (
                self._n_low_confidence / max(self._n_predictions, 1)
            ),
            "drift_checks_triggered": self._n_drifts_detected,
            "psi_mean": drift_info.get("psi_mean"),
            "drifted": drift_info.get("drifted", False),
            "class_distribution": dict(self._class_counts),
            "accuracy": (
                self._n_correct / self._n_labelled
                if self._n_labelled > 0
                else None
            ),
            "latency_p50_ms": float(np.median(self._latencies_ms)) if self._latencies_ms else None,
            "latency_p95_ms": float(np.percentile(list(self._latencies_ms), 95)) if self._latencies_ms else None,
        }

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all counters (keep baseline)."""
        self._input_buffer.clear()
        self._n_predictions = 0
        self._n_low_confidence = 0
        self._n_drifts_detected = 0
        self._class_counts.clear()
        self._latencies_ms.clear()
        self._n_correct = 0
        self._n_labelled = 0
        self._last_drift_report = None


# ── helpers ─────────────────────────────────────────────────────────────────

def _flat(x: "np.ndarray | torch.Tensor") -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(1, -1)
