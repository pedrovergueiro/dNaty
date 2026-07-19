"""
Data drift detection for production monitoring (Inner-Image-6313, r/deeplearning).

Uses Population Stability Index (PSI) and KL divergence to detect when the
input distribution has shifted from the training baseline.

PSI interpretation:
  < 0.1  -> no significant change
  0.1-0.2 -> minor shift, monitor
  > 0.2  -> significant drift -- consider recompression/retraining

Usage:
    from dnaty.monitoring import DriftDetector

    detector = DriftDetector()
    detector.fit(train_x)          # establish baseline from training data

    # Later in production:
    report = detector.score(batch_x)
    if report["drifted"]:
        print(f"Drift detected! PSI={report['psi_mean']:.3f}")
        # trigger recompression or alert
"""
from __future__ import annotations

import numpy as np
import torch


class DriftDetector:
    """
    Monitors input feature distributions using PSI + KL divergence.

    Works with any 2-D array (N, features) or flat 1-D array.
    For images, pass flattened or pooled representations.
    """

    def __init__(
        self,
        n_bins: int = 10,
        psi_threshold: float = 0.2,
        smoothing: float = 1e-6,
        threshold: "float | None" = None,
    ):
        """
        Args:
            n_bins:         Histogram bins per feature.
            psi_threshold:  PSI above this -> drifted=True (0.2 = industry standard).
            smoothing:      Additive smoothing to avoid log(0).
            threshold:      Alias for psi_threshold (matches the published docs).
        """
        self.n_bins = n_bins
        self.psi_threshold = psi_threshold if threshold is None else threshold
        self.smoothing = smoothing
        self._baselines: list[np.ndarray] = []
        self._edges: list[np.ndarray] = []
        self._n_features: int = 0
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, data: "np.ndarray | torch.Tensor") -> "DriftDetector":
        """Compute baseline histograms from training data.

        Args:
            data: Shape (N,) or (N, F). For images use flattened or pooled.
        """
        arr = _to_numpy(data)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self._n_features = arr.shape[1]
        self._baselines, self._edges = [], []
        for i in range(self._n_features):
            col = arr[:, i]
            hist, edges = np.histogram(col, bins=self.n_bins)
            hist = _smooth(hist, self.smoothing)
            self._baselines.append(hist)
            self._edges.append(edges)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def score(self, data: "np.ndarray | torch.Tensor") -> dict:
        """Compute drift scores against the baseline.

        Returns:
            dict with keys:
              psi_mean     -- mean PSI across features
              psi_max      -- max PSI (worst feature)
              kl_mean      -- mean KL divergence
              psi_per_feature -- list[float]
              kl_per_feature  -- list[float]
              drifted      -- True if psi_mean > psi_threshold
              n_samples    -- number of samples scored
        """
        self._check_fitted()
        arr = _to_numpy(data)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {arr.shape[1]}"
            )

        psi_list, kl_list = [], []
        for i, (baseline, edges) in enumerate(zip(self._baselines, self._edges)):
            # Clip to the baseline range so out-of-range samples land in the
            # first/last bin instead of being silently dropped by np.histogram —
            # otherwise a large mean shift excludes most production samples from
            # the comparison and distorts PSI.
            col = np.clip(arr[:, i], edges[0], edges[-1])
            hist, _ = np.histogram(col, bins=edges)
            actual = _smooth(hist, self.smoothing)

            # PSI = Sum (actual - expected) x ln(actual / expected)
            psi = float(np.sum((actual - baseline) * np.log(actual / baseline)))
            psi_list.append(psi)

            # KL(baseline || actual) = Sum baseline x ln(baseline / actual)
            kl = float(np.sum(baseline * np.log(baseline / actual)))
            kl_list.append(kl)

        psi_mean = float(np.mean(psi_list))
        return {
            "psi": psi_mean,  # alias of psi_mean -- matches the published docs
            "psi_mean": psi_mean,
            "psi_max": float(np.max(psi_list)),
            "kl_mean": float(np.mean(kl_list)),
            "kl_divergence": float(np.mean(kl_list)),  # alias -- matches the published docs
            "psi_per_feature": psi_list,
            "kl_per_feature": kl_list,
            "drifted": psi_mean > self.psi_threshold,
            "n_samples": len(arr),
        }

    # ------------------------------------------------------------------
    def is_drifted(self, data: "np.ndarray | torch.Tensor") -> bool:
        """Convenience wrapper: True if drift detected."""
        return self.score(data)["drifted"]

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call .fit(train_data) before scoring.")


# -- helpers -----------------------------------------------------------------

def _to_numpy(data: "np.ndarray | torch.Tensor") -> np.ndarray:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().float().numpy()
    return np.asarray(data, dtype=np.float32)


def _smooth(hist: np.ndarray, eps: float) -> np.ndarray:
    h = hist.astype(np.float64) + eps
    return h / h.sum()
