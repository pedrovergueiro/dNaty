"""
Continual Learning metrics -- Lopez-Paz et al. (2017).
BWT, FWT, FM implemented as formalized in Section 1.5.
"""
from __future__ import annotations
import numpy as np


def compute_cl_metrics(R: np.ndarray, baselines: np.ndarray | None = None) -> dict[str, float]:
    """
    R[i, j] = accuracy on task j after training sequentially up to task i.
    R is indexed from 0 (after task 0) to T-1 (after task T-1).
    baselines[j] = single-task accuracy on task j (for FWT).
    """
    T = R.shape[1]

    # BWT: Backward Transfer -- forgetting
    # BWT = (1/(T-1)) * sum_{i=1}^{T-1} (R[T-1,i] - R[i,i])
    bwt_terms = [R[T - 1, i] - R[i, i] for i in range(T - 1)]
    BWT = float(np.mean(bwt_terms)) if bwt_terms else 0.0

    # FWT: Forward Transfer
    if baselines is not None:
        fwt_terms = [R[i - 1, i] - baselines[i] for i in range(1, T)]
        FWT = float(np.mean(fwt_terms)) if fwt_terms else 0.0
    else:
        FWT = 0.0

    # FM: Forgetting Measure -- drop from peak
    fm_terms = []
    for i in range(T - 1):
        peak = max(R[j, i] for j in range(T))
        fm_terms.append(peak - R[T - 1, i])
    FM = float(np.mean(fm_terms)) if fm_terms else 0.0

    return {
        "BWT": round(BWT, 4),
        "FWT": round(FWT, 4),
        "FM": round(FM, 4),
    }
