"""Distribution-drift statistics: Population Stability Index and KS tests.

PSI = sum_i (a_i - e_i) * ln(a_i / e_i), where e_i / a_i are the shares of the
reference (expected) and current (actual) populations in bin i. Bins are the
reference deciles, so every reference bin holds ~10% of the population. Empty bins are
floored at a small epsilon to keep the logarithm finite.

Conventional reading: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from credit_risk.config import MonitoringConfig

EPSILON = 1e-4


def _shares(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    return np.clip(counts / total if total else counts, EPSILON, None)


def psi_numeric(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = _shares(np.histogram(ref, edges)[0].astype(float))
    a = _shares(np.histogram(cur, edges)[0].astype(float))
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    cats = sorted(set(reference.dropna().astype(str)) | set(current.dropna().astype(str)))
    e = _shares(reference.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy(float))
    a = _shares(current.astype(str).value_counts().reindex(cats, fill_value=0).to_numpy(float))
    return float(np.sum((a - e) * np.log(a / e)))


def psi_status(value: float, cfg: MonitoringConfig) -> str:
    if np.isnan(value):
        return "n/a"
    if value >= cfg.psi_significant:
        return "significant"
    if value >= cfg.psi_moderate:
        return "moderate"
    return "stable"


def ks_test(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan"), float("nan")
    result = stats.ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)
