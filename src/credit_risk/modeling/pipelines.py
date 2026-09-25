"""Model pipelines and the small, explicit hyper-parameter grids searched on validation.

Design choices (docs/modeling.md):
* No class re-weighting: re-weighting distorts predicted PDs, and PDs must stay usable
  for calibration and expected-loss analysis. Imbalance is handled in evaluation
  (PR-AUC, threshold analysis) instead.
* Linear model inputs are winsorised at train-set 1st/99th percentiles, log1p-transformed
  when skewed, imputed with a median plus a missing indicator ("no history" is signal),
  and standardised so coefficients are comparable.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from credit_risk.features.registry import (
    categorical_features,
    numeric_features,
    skewed_features,
)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned on the training data."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99) -> None:
        self.lower = lower
        self.upper = upper

    def fit(self, X: np.ndarray, y: Any = None) -> Winsorizer:  # noqa: N803
        arr = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(arr, self.lower, axis=0)
        self.upper_ = np.nanquantile(arr, self.upper, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(input_features, dtype=object)


def _log1p_nonneg(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0, None))


def _one_hot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def linear_preprocessor() -> ColumnTransformer:
    skewed = skewed_features()
    plain = [f for f in numeric_features() if f not in skewed]
    return ColumnTransformer(
        [
            (
                "skewed",
                Pipeline(
                    [
                        ("winsor", Winsorizer()),
                        ("log", FunctionTransformer(_log1p_nonneg, feature_names_out="one-to-one")),
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                skewed,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("winsor", Winsorizer()),
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                plain,
            ),
            ("categorical", _one_hot(), categorical_features()),
        ],
        verbose_feature_names_out=True,
    )


def tree_preprocessor(impute: bool) -> ColumnTransformer:
    numeric: Any = SimpleImputer(strategy="median", add_indicator=True) if impute else "passthrough"
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_features()),
            ("categorical", _one_hot(), categorical_features()),
        ],
        verbose_feature_names_out=True,
    )


@dataclass(frozen=True)
class Candidate:
    family: str
    display_name: str
    grid: list[dict[str, Any]]

    def build(self, params: dict[str, Any], random_state: int) -> Pipeline:
        if self.family == "logistic_regression":
            model: Any = LogisticRegression(max_iter=5000, random_state=random_state, **params)
            return Pipeline([("prep", linear_preprocessor()), ("model", model)])
        if self.family == "random_forest":
            model = RandomForestClassifier(
                n_estimators=400, n_jobs=-1, random_state=random_state, **params
            )
            return Pipeline([("prep", tree_preprocessor(impute=True)), ("model", model)])
        if self.family == "hist_gradient_boosting":
            model = HistGradientBoostingClassifier(
                random_state=random_state,
                min_samples_leaf=50,
                l2_regularization=1.0,
                early_stopping=False,
                **params,
            )
            return Pipeline([("prep", tree_preprocessor(impute=False)), ("model", model)])
        raise ValueError(f"Unknown model family {self.family}")

    def configurations(self) -> Iterator[dict[str, Any]]:
        yield from self.grid


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        "logistic_regression", "Logistic Regression", list(ParameterGrid({"C": [0.01, 0.1, 1.0]}))
    ),
    Candidate(
        "random_forest",
        "Random Forest",
        list(ParameterGrid({"min_samples_leaf": [20, 50], "max_features": ["sqrt", 0.3]})),
    ),
    Candidate(
        "hist_gradient_boosting",
        "Hist Gradient Boosting",
        [
            {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 7},
            {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 15},
            {"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31},
            {"learning_rate": 0.1, "max_iter": 150, "max_leaf_nodes": 15},
            {"learning_rate": 0.1, "max_iter": 150, "max_leaf_nodes": 31},
        ],
    ),
)


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select registry features only, with categorical columns as plain strings."""
    X = df[numeric_features() + categorical_features()].copy()  # noqa: N806
    for col in numeric_features():
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for col in categorical_features():
        X[col] = X[col].astype(str)
    return X
