"""
Statistical analysis: paired t-test, Cohen's d, ANOVA + Tukey HSD.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def paired_ttest(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Return (t_stat, p_value, cohen_d)."""
    a_arr, b_arr = np.array(a), np.array(b)
    if len(a_arr) != len(b_arr):
        raise ValueError("paired_ttest requires lists of equal length")
    if len(a_arr) < 2:
        raise ValueError("paired_ttest requires at least 2 pairs")
    t_stat, p_val = stats.ttest_rel(a_arr, b_arr)
    diff = a_arr - b_arr
    d = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
    return float(t_stat), float(p_val), d


def _independent_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Welch's t-test (unequal variance) + pooled-SD Cohen's d for two
    INDEPENDENT samples (e.g. two methods measured on separate runs)."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    t_stat, p_val = stats.ttest_ind(a_arr, b_arr, equal_var=False)
    na, nb = len(a_arr), len(b_arr)
    sa2, sb2 = a_arr.var(ddof=1), b_arr.var(ddof=1)
    pooled = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / max(na + nb - 2, 1))
    d = float((a_arr.mean() - b_arr.mean()) / (pooled + 1e-12))
    return float(t_stat), float(p_val), d


def anova_tukey(groups: dict[str, list[float]]) -> dict[str, object]:
    """One-way ANOVA + Bonferroni-corrected pairwise post-hoc comparisons.

    Groups are treated as INDEPENDENT samples (different methods/seeds), so the
    post-hoc uses Welch's t-test with a pooled-SD Cohen's d and Bonferroni
    correction over the number of pairwise comparisons. Groups may have
    different sizes.

    (An earlier version mislabelled this as "Tukey HSD" and used a *paired*
    t-test, which is invalid for independent groups and raised on unequal group
    sizes; `p` here is the corrected value, `p_uncorrected` the raw one.)
    """
    names = list(groups.keys())
    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    f_stat, p_anova = stats.f_oneway(*arrays)
    result = {
        "f_stat": float(f_stat),
        "p_anova": float(p_anova),
        "significant": bool(p_anova < 0.05),
        "groups": names,
    }
    n_pairs = max(len(names) * (len(names) - 1) // 2, 1)
    pairs = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            t, p, d = _independent_ttest(arrays[i], arrays[j])
            p_corr = min(1.0, p * n_pairs)  # Bonferroni
            pairs[f"{names[i]} vs {names[j]}"] = {
                "p": round(p_corr, 4),
                "p_uncorrected": round(p, 4),
                "d": round(d, 3),
                "sig": bool(p_corr < 0.05),
            }
    result["pairs"] = pairs
    return result


def summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std()), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }
