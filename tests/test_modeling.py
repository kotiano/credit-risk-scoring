"""Model training, selection protocol, evaluation artifacts and monitoring."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from conftest import PipelineRun

from credit_risk.features.registry import categorical_features, numeric_features
from credit_risk.modeling.metrics import best_f1_threshold
from credit_risk.modeling.train import select_champion


@pytest.fixture(scope="module")
def predictions(pipeline: PipelineRun) -> pd.DataFrame:
    return pd.read_parquet(pipeline.settings.paths.evaluation / "predictions.parquet")


@pytest.fixture(scope="module")
def metadata(pipeline: PipelineRun) -> dict:
    return json.loads((pipeline.settings.paths.models / "metadata.json").read_text())


def test_probabilities_are_valid(predictions: pd.DataFrame) -> None:
    for col in [c for c in predictions if c.startswith("pd")]:
        assert predictions[col].between(0, 1).all()
        assert predictions[col].std() > 0


def test_models_use_exactly_the_registered_features(pipeline: PipelineRun) -> None:
    model = joblib.load(pipeline.settings.paths.models / "logistic_regression.joblib")
    assert list(model.feature_names_in_) == numeric_features() + categorical_features()


def test_models_rank_better_than_random_out_of_time(predictions: pd.DataFrame) -> None:
    from sklearn.metrics import roc_auc_score

    test = predictions[predictions["sample_split"] == "test"]
    assert roc_auc_score(test["target"].astype(int), test["pd"]) > 0.6


def test_threshold_was_chosen_on_validation(predictions: pd.DataFrame, metadata: dict) -> None:
    val = predictions[predictions["sample_split"] == "validation"]
    expected = best_f1_threshold(
        val["target"].astype(int).to_numpy(), val[f"pd_{metadata['champion']}"].to_numpy()
    )
    assert metadata["decision_threshold"] == pytest.approx(expected)


def test_parsimony_rule() -> None:
    assert (
        select_champion(
            {"logistic_regression": 0.78, "random_forest": 0.785, "hist_gradient_boosting": 0.781},
            0.01,
        )[0]
        == "logistic_regression"
    )
    assert (
        select_champion(
            {"logistic_regression": 0.70, "random_forest": 0.72, "hist_gradient_boosting": 0.75},
            0.01,
        )[0]
        == "hist_gradient_boosting"
    )


def test_every_loan_is_scored_and_banded(pipeline: PipelineRun) -> None:
    n_scores, n_loans, n_bands = pipeline.con.execute(
        "SELECT count(*), (SELECT count(*) FROM clean.loans), count(risk_band) "
        "FROM model.loan_scores"
    ).fetchone()
    assert n_scores == n_loans == n_bands


def test_evaluation_contains_required_views(pipeline: PipelineRun) -> None:
    ev = json.loads((pipeline.settings.paths.evaluation / "evaluation.json").read_text())
    perf = pd.DataFrame(ev["performance"])
    assert {
        "roc_auc",
        "pr_auc",
        "brier",
        "ks",
        "gini",
        "precision",
        "recall",
        "specificity",
        "f1",
        "calibration_ratio",
    } <= set(perf.columns)
    assert set(perf["split"]) == {"train", "validation", "test", "recent"}
    deciles = pd.DataFrame(ev["deciles_test"])
    assert len(deciles) == 10 and deciles["cum_bads_captured_pct"].iloc[-1] == pytest.approx(1)
    assert ev["local_explanations"] and ev["fairness_test"] and ev["permutation_importance"]
    report = (pipeline.settings.paths.reports / "model_validation_report.md").read_text()
    assert "Predicted risk vs actual outcomes" in report


def test_monitoring_outputs(pipeline: PipelineRun) -> None:
    mon = json.loads((pipeline.settings.paths.evaluation / "monitoring.json").read_text())
    drift = pd.DataFrame(mon["feature_drift"])
    assert set(drift["window"]) == {"validation", "test", "current"}
    assert (drift["psi"].dropna() >= 0).all()
    perf = pd.DataFrame(mon["performance"]).set_index("window")
    assert perf.loc["train", "roc_auc_change_vs_train"] == 0
    monthly = pd.DataFrame(mon["monthly"])
    # Performance is only reported for months with matured outcomes.
    assert monthly.loc[monthly["matured"] == 0, "roc_auc"].isna().all()


def test_segmentation_outputs(pipeline: PipelineRun) -> None:
    clusters = pipeline.con.execute("SELECT * FROM marts.borrower_clusters").df()
    bm = pipeline.con.execute("SELECT count(*) FROM marts.borrower_mart").fetchone()[0]
    assert len(clusters) == bm and clusters["cluster"].nunique() >= 3
    assert np.isfinite(clusters["cluster"].str.extract(r"(\d+)").astype(int)).all().all()
