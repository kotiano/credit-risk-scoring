"""Train candidate models out-of-time, select a champion, and score every loan.

Protocol (docs/modeling.md):
1. Fit every hyper-parameter configuration on the *train* window only.
2. Pick each family's best configuration by validation ROC-AUC.
3. Champion selection with a parsimony rule: keep Logistic Regression unless a
   challenger improves validation ROC-AUC by more than ``parsimony_auc_margin``.
4. Choose the reference decision threshold on validation (max F1) - never on test.
5. The test window is scored once, after all choices are frozen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from credit_risk.config import Settings
from credit_risk.features.registry import categorical_features, numeric_features
from credit_risk.logging_utils import get_logger, log_event
from credit_risk.modeling.metrics import best_f1_threshold
from credit_risk.modeling.pipelines import CANDIDATES, feature_frame
from credit_risk.modeling.scorecard import assign_risk_band, pd_to_score

logger = get_logger(__name__)

LABELLED_SPLITS = ("train", "validation", "test", "recent")


@dataclass
class TrainingResult:
    models: dict[str, Pipeline]
    champion: str
    threshold: float
    tuning: pd.DataFrame
    predictions: pd.DataFrame
    metadata: dict[str, Any]


def load_model_dataset(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("SELECT * FROM features.model_dataset ORDER BY decision_date, loan_id").df()
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    return df


def select_champion(val_auc: dict[str, float], margin: float) -> tuple[str, str]:
    baseline = "logistic_regression"
    challenger = max((k for k in val_auc if k != baseline), key=lambda k: val_auc[k])
    uplift = val_auc[challenger] - val_auc[baseline]
    if uplift > margin:
        return challenger, (
            f"{challenger} improves validation ROC-AUC by {uplift:.4f} (> {margin:.2f} "
            "parsimony margin) over the logistic baseline."
        )
    comparison = (
        f"trails the logistic baseline by {-uplift:.4f}"
        if uplift <= 0
        else f"beats the logistic baseline by only {uplift:.4f}"
    )
    return baseline, (
        f"Best challenger ({challenger}) {comparison} validation ROC-AUC "
        f"(parsimony margin {margin:.2f}); the interpretable logistic baseline is kept."
    )


def train_models(con: duckdb.DuckDBPyConnection, settings: Settings) -> TrainingResult:
    cfg = settings.modeling
    df = load_model_dataset(con)
    train = df[df["sample_split"] == "train"]
    valid = df[df["sample_split"] == "validation"]
    if train["target"].nunique() < 2 or valid["target"].nunique() < 2:
        raise ValueError("Train and validation windows must both contain goods and bads")
    X_train, y_train = feature_frame(train), train["target"].astype(int)  # noqa: N806
    X_valid, y_valid = feature_frame(valid), valid["target"].astype(int)  # noqa: N806

    tuning_rows: list[dict[str, Any]] = []
    best_models: dict[str, Pipeline] = {}
    best_auc: dict[str, float] = {}
    for candidate in CANDIDATES:
        for params in candidate.configurations():
            pipe = candidate.build(params, cfg.random_state)
            pipe.fit(X_train, y_train)
            train_auc = roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1])
            val_auc = roc_auc_score(y_valid, pipe.predict_proba(X_valid)[:, 1])
            tuning_rows.append(
                {
                    "model": candidate.family,
                    "params": json.dumps(params),
                    "train_roc_auc": train_auc,
                    "validation_roc_auc": val_auc,
                }
            )
            if val_auc > best_auc.get(candidate.family, -np.inf):
                best_auc[candidate.family] = val_auc
                best_models[candidate.family] = pipe
            log_event(
                logger,
                "fitted candidate",
                model=candidate.family,
                params=params,
                train_auc=round(train_auc, 4),
                val_auc=round(val_auc, 4),
            )

    champion, rationale = select_champion(best_auc, cfg.parsimony_auc_margin)
    threshold = best_f1_threshold(
        y_valid.to_numpy(), best_models[champion].predict_proba(X_valid)[:, 1]
    )

    X_all = feature_frame(df)  # noqa: N806
    predictions = df[
        ["loan_id", "borrower_id", "decision_date", "sample_split", "target", "product_type"]
    ].copy()
    for family, model in best_models.items():
        predictions[f"pd_{family}"] = model.predict_proba(X_all)[:, 1]
    predictions["pd"] = predictions[f"pd_{champion}"]
    predictions["score"] = pd_to_score(predictions["pd"].to_numpy(), cfg)
    predictions["risk_band"] = assign_risk_band(predictions["pd"].to_numpy(), cfg)

    tuning = pd.DataFrame(tuning_rows)
    metadata = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "champion": champion,
        "selection_rationale": rationale,
        "decision_threshold": threshold,
        "threshold_rule": "max F1 on the validation window",
        "validation_roc_auc": best_auc,
        "best_params": {
            f: json.loads(
                tuning.loc[
                    (tuning["model"] == f) & (tuning["validation_roc_auc"] == best_auc[f]), "params"
                ].iloc[0]
            )
            for f in best_models
        },
        "features": {"numeric": numeric_features(), "categorical": categorical_features()},
        "windows": {
            "train": f"decision_date <= {cfg.train_end}",
            "validation": f"{cfg.train_end} < decision_date <= {cfg.validation_end}",
            "test": f"{cfg.validation_end} < decision_date <= {cfg.test_end}",
        },
        "sample_sizes": df.groupby("sample_split").size().to_dict(),
        "target": f"bad_30dpd: loan reached {settings.risk.target_dpd} DPD",
    }
    return TrainingResult(best_models, champion, threshold, tuning, predictions, metadata)


def persist_training(
    result: TrainingResult, con: duckdb.DuckDBPyConnection, settings: Settings
) -> None:
    paths = settings.paths
    paths.models.mkdir(parents=True, exist_ok=True)
    paths.evaluation.mkdir(parents=True, exist_ok=True)
    for family, model in result.models.items():
        joblib.dump(model, paths.models / f"{family}.joblib")
    (paths.models / "metadata.json").write_text(
        json.dumps(result.metadata, indent=2, default=str), encoding="utf-8"
    )
    result.tuning.to_csv(paths.evaluation / "tuning_results.csv", index=False)
    result.predictions.to_parquet(paths.evaluation / "predictions.parquet", index=False)

    scores = result.predictions[["loan_id", "sample_split", "pd", "score", "risk_band"]].copy()
    scores["model_name"] = result.champion
    con.register("scores_df", scores)
    con.execute("CREATE SCHEMA IF NOT EXISTS model")
    con.execute("CREATE OR REPLACE TABLE model.loan_scores AS SELECT * FROM scores_df")
    log_event(
        logger,
        "persisted models and scores",
        champion=result.champion,
        threshold=round(result.threshold, 4),
        scored_loans=len(scores),
    )
