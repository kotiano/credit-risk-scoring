"""PD -> score scaling and analytical risk bands.

Score scaling uses the industry-standard points-to-double-odds (PDO) convention:

    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)
    score  = offset + factor * ln(odds_good_to_bad),  odds = (1 - PD) / PD

With base_score=600 at 50:1 odds and PDO=20, every 20 points doubles the good:bad odds.
Bands are *analytical groupings of model output* for monitoring, not credit policy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_risk.config import ModelingConfig

PD_FLOOR, PD_CAP = 1e-4, 1 - 1e-4


def pd_to_score(pd_values: np.ndarray, cfg: ModelingConfig) -> np.ndarray:
    p = np.clip(np.asarray(pd_values, dtype=float), PD_FLOOR, PD_CAP)
    factor = cfg.score_pdo / np.log(2)
    offset = cfg.score_base - factor * np.log(cfg.score_base_odds)
    return np.round(offset + factor * np.log((1 - p) / p)).astype(int)


def score_to_pd(scores: np.ndarray, cfg: ModelingConfig) -> np.ndarray:
    factor = cfg.score_pdo / np.log(2)
    offset = cfg.score_base - factor * np.log(cfg.score_base_odds)
    odds = np.exp((np.asarray(scores, dtype=float) - offset) / factor)
    return 1 / (1 + odds)


def assign_risk_band(pd_values: np.ndarray, cfg: ModelingConfig) -> np.ndarray:
    uppers = [u for u, _ in cfg.risk_bands]
    labels = np.array([label for _, label in cfg.risk_bands])
    idx = np.searchsorted(np.array(uppers), np.asarray(pd_values, dtype=float), side="right")
    return labels[np.clip(idx, 0, len(labels) - 1)]


def band_order(cfg: ModelingConfig) -> list[str]:
    return [label for _, label in cfg.risk_bands]


def band_performance(
    frame: pd.DataFrame, cfg: ModelingConfig, pd_col: str = "pd", target_col: str = "target"
) -> pd.DataFrame:
    """Population, predicted PD and actual bad rate per analytical risk band."""
    out = (
        frame.groupby("risk_band", observed=True)
        .agg(
            population=(target_col, "size"),
            avg_predicted_pd=(pd_col, "mean"),
            actual_bad_rate=(target_col, "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .reindex(band_order(cfg))
        .dropna(subset=["population"])
    )
    out["population_share"] = out["population"] / out["population"].sum()
    return out.reset_index()
