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

import json
import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Optional

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
        max_failures: int = 1000,
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
        self._consecutive_drift_count = 0
        self._class_counts: dict[int, int] = defaultdict(int)
        self._latencies_ms: deque[float] = deque(maxlen=500)

        # Failure tracking (when labels are provided to .record_outcome)
        self._n_correct = 0
        self._n_labelled = 0
        self._failure_buffer: list[dict] = []
        self._max_failures = max_failures

        # Honor a pre-fitted detector (the documented flow: fit() it, pass it in).
        self._baseline_fitted = bool(getattr(self.drift_detector, "_fitted", False))
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
              confidences  -- max softmax probability per sample
              alert        -- None or alert string ("DRIFT DETECTED", "LOW CONFIDENCE")
              drift_score  -- latest drift PSI (None until first check)
              latency_ms   -- inference latency for this batch
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
                self._consecutive_drift_count += 1
                alert = f"DRIFT DETECTED (PSI={drift_score:.3f})"
            else:
                self._consecutive_drift_count = 0

        if alert is None and low_conf > 0:
            alert = f"LOW CONFIDENCE ({low_conf}/{len(predictions)} samples below {self.confidence_threshold:.0%})"

        return predictions, {
            "confidences": confidences,
            "alert": alert,
            "drift_score": drift_score,
            "psi": drift_score,  # alias of drift_score -- matches the published docs
            "latency_ms": latency_ms,
            "n_samples": len(predictions),
        }

    # ------------------------------------------------------------------
    def record_outcome(
        self,
        predictions: np.ndarray,
        ground_truth: "np.ndarray | torch.Tensor",
        inputs: "np.ndarray | torch.Tensor | None" = None,
    ) -> float:
        """Track prediction accuracy with ground-truth labels.

        Args:
            predictions:  Output of predict()[0].
            ground_truth: True class labels for this batch.
            inputs:       Raw input batch (same order as predictions). When provided,
                          wrong predictions are stored in the failure buffer for
                          export_failure_report().

        Returns:
            Accuracy for this batch.
        """
        gt = np.asarray(ground_truth)
        correct = int((predictions == gt).sum())
        self._n_correct += correct
        self._n_labelled += len(gt)

        if inputs is not None and len(self._failure_buffer) < self._max_failures:
            flat = _flat(inputs)
            wrong = np.where(predictions != gt)[0]
            for i in wrong:
                if len(self._failure_buffer) >= self._max_failures:
                    break
                self._failure_buffer.append({
                    "predicted": int(predictions[i]),
                    "expected": int(gt[i]),
                    "input": flat[i].tolist(),
                })

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
    def auto_retrigger(
        self,
        compress_fn: Callable,
        train_data: "np.ndarray | torch.Tensor",
        consecutive_drifts: int = 3,
        on_trigger: Optional[Callable[["ProductionTracker"], None]] = None,
    ) -> bool:
        """Recompress the model automatically when drift persists.

        Call this after each predict() call (or periodically). When the drift
        detector has fired positively for ``consecutive_drifts`` consecutive
        checks, ``compress_fn(train_data)`` is called, the tracker baseline is
        re-fitted on ``train_data``, and the drift counter is reset.

        Args:
            compress_fn:        Callable that accepts ``train_data`` and returns a
                                new compressed ``nn.Module`` (e.g. a lambda wrapping
                                ``dnaty.compress``).
            train_data:         Training data to re-fit the drift baseline after
                                recompression.
            consecutive_drifts: Number of consecutive drift checks that must fire
                                before triggering recompression (default 3).
            on_trigger:         Optional callback called with ``self`` after
                                recompression (useful for webhooks / notifications).

        Returns:
            True if recompression was triggered this call, False otherwise.
        """
        if self._consecutive_drift_count < consecutive_drifts:
            return False

        new_model = compress_fn(train_data)
        if new_model is not None:
            self.model = new_model.to(self.device)

        self.fit_baseline(train_data)
        self._consecutive_drift_count = 0

        if on_trigger is not None:
            try:
                on_trigger(self)
            except Exception:
                pass

        return True

    # ------------------------------------------------------------------
    def export_failure_report(
        self,
        path: str,
        n_components: int = 2,
        db_uri: Optional[str] = None,
    ) -> dict:
        """Export a JSON report of failure cases (wrong predictions).

        Applies PCA to the stored failure inputs (when >= 3 samples) to
        provide 2-D projections for visualisation and clustering.
        Optionally persists to a SQLite database via ``db_uri``.

        Args:
            path:          File path for the JSON report (.json).
            n_components:  Number of PCA components in the report (default 2).
            db_uri:        Optional SQLite URI, e.g. ``"sqlite:///failures.db"``
                           or a raw path like ``"failures.db"``. When provided,
                           rows are appended to a ``failures`` table.

        Returns:
            The report dict (same content written to ``path``).
        """
        failures = list(self._failure_buffer)
        n = len(failures)

        pca_coords: list[list[float]] = []
        if n >= 3:
            X = np.array([f["input"] for f in failures], dtype=np.float32)
            X_c = X - X.mean(axis=0)
            cov = np.cov(X_c.T) if X_c.shape[1] > 1 else np.array([[float(np.var(X_c))]])
            vals, vecs = np.linalg.eigh(cov)
            k = min(n_components, vecs.shape[1])
            top_k = vecs[:, np.argsort(vals)[::-1][:k]]
            proj = (X_c @ top_k).tolist()
            pca_coords = proj

        report = {
            "n_failures": n,
            "failure_rate": n / max(self._n_labelled, 1),
            "class_breakdown": _count_class_errors(failures),
            "pca_components": n_components,
            "pca_coords": pca_coords,
            "samples": failures,
        }

        Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")

        if db_uri is not None:
            _write_failures_sqlite(db_uri, failures, pca_coords)

        return report

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear all counters (keep baseline)."""
        self._input_buffer.clear()
        self._n_predictions = 0
        self._n_low_confidence = 0
        self._n_drifts_detected = 0
        self._consecutive_drift_count = 0
        self._class_counts.clear()
        self._latencies_ms.clear()
        self._n_correct = 0
        self._n_labelled = 0
        self._last_drift_report = None
        self._failure_buffer.clear()


# -- helpers -----------------------------------------------------------------

def _flat(x: "np.ndarray | torch.Tensor") -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(x.shape[0], -1) if x.ndim > 1 else x.reshape(1, -1)


def _count_class_errors(failures: list[dict]) -> dict:
    counts: dict[str, int] = defaultdict(int)
    for f in failures:
        key = f"{f['expected']}->{f['predicted']}"
        counts[key] += 1
    return dict(counts)


def _write_failures_sqlite(db_uri: str, failures: list[dict], pca_coords: list) -> None:
    path = db_uri.replace("sqlite:///", "").replace("sqlite://", "")
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS failures ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "predicted INTEGER,"
            "expected INTEGER,"
            "input_json TEXT,"
            "pca_x REAL,"
            "pca_y REAL"
            ")"
        )
        rows = []
        for i, f in enumerate(failures):
            pca_x = pca_coords[i][0] if i < len(pca_coords) else None
            pca_y = pca_coords[i][1] if i < len(pca_coords) and len(pca_coords[i]) > 1 else None
            rows.append((f["predicted"], f["expected"], json.dumps(f["input"]), pca_x, pca_y))
        conn.executemany(
            "INSERT INTO failures (predicted, expected, input_json, pca_x, pca_y) VALUES (?,?,?,?,?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()
