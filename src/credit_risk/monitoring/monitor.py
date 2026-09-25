"""Model monitoring: data drift, prediction drift and performance drift.

Populations (by decision date):
* reference - the training window (what the model learned from);
* comparison windows - validation, test, and "current" (decisions after the test
  window, including loans whose outcome is not yet observable).

Data and prediction drift can be measured immediately for every scored application.
Performance drift (AUC, PR-AUC, calibration) needs *matured* outcomes: a loan's
bad_30dpd is only known once due_date + 30 days has passed, so performance is reported
only for matured loans and the most recent months are necessarily incomplete.
"""

from __future__ import annotations

import json
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from credit_risk.config import Settings
from credit_risk.features.registry import categorical_features, numeric_features
from credit_risk.logging_utils import get_logger, log_event
from credit_risk.modeling.train import load_model_dataset
from credit_risk.monitoring.drift import ks_test, psi_categorical, psi_numeric, psi_status
from credit_risk.reporting.markdown import GENERATED_BANNER, df_to_markdown

logger = get_logger(__name__)

WINDOWS = ("validation", "test", "current")
MIN_MATURED_FOR_AUC = 100


def _window(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "current":
        return df[df["sample_split"].isin(["recent", "unobserved"])]
    return df[df["sample_split"] == name]


def feature_drift(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    cfg = settings.monitoring
    reference = frame[frame["sample_split"] == "train"]
    rows = []
    for window in WINDOWS:
        current = _window(frame, window)
        for col in numeric_features():
            psi = psi_numeric(
                reference[col].to_numpy(float), current[col].to_numpy(float), cfg.psi_bins
            )
            ks, p_value = ks_test(reference[col].to_numpy(float), current[col].to_numpy(float))
            rows.append(
                {
                    "window": window,
                    "feature": col,
                    "psi": psi,
                    "status": psi_status(psi, cfg),
                    "ks_statistic": ks,
                    "ks_p_value": p_value,
                    "reference_mean": reference[col].mean(),
                    "current_mean": current[col].mean(),
                }
            )
        for col in categorical_features():
            psi = psi_categorical(reference[col], current[col])
            rows.append(
                {
                    "window": window,
                    "feature": col,
                    "psi": psi,
                    "status": psi_status(psi, cfg),
                    "ks_statistic": np.nan,
                    "ks_p_value": np.nan,
                    "reference_mean": np.nan,
                    "current_mean": np.nan,
                }
            )
    return pd.DataFrame(rows)


def prediction_drift(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    cfg = settings.monitoring
    reference = frame[frame["sample_split"] == "train"]
    rows = []
    for window in WINDOWS:
        current = _window(frame, window)
        psi = psi_numeric(reference["pd"].to_numpy(), current["pd"].to_numpy(), cfg.psi_bins)
        ks, p_value = ks_test(reference["pd"].to_numpy(), current["pd"].to_numpy())
        rows.append(
            {
                "window": window,
                "n": len(current),
                "mean_pd": current["pd"].mean(),
                "reference_mean_pd": reference["pd"].mean(),
                "score_psi": psi,
                "status": psi_status(psi, cfg),
                "ks_statistic": ks,
                "ks_p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def monthly_monitoring(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Per decision month: volume, score PSI vs train, and matured-outcome performance."""
    cfg = settings.monitoring
    reference_pd = frame.loc[frame["sample_split"] == "train", "pd"].to_numpy()
    frame = frame.assign(month=frame["decision_date"].dt.to_period("M").dt.to_timestamp())
    rows = []
    for month, g in frame.groupby("month"):
        matured = g[g["target"].notna()]
        y = matured["target"].astype(int).to_numpy()
        p = matured["pd"].to_numpy()
        enough = len(matured) >= MIN_MATURED_FOR_AUC and 0 < y.sum() < len(y)
        rows.append(
            {
                "month": pd.Timestamp(month).date().isoformat(),  # type: ignore[arg-type]
                "split": g["sample_split"].mode().iloc[0],
                "applications": len(g),
                "matured": len(matured),
                "maturity_share": len(matured) / len(g),
                "mean_pd": g["pd"].mean(),
                "actual_bad_rate": y.mean() if len(y) else np.nan,
                "mean_pd_matured": p.mean() if len(p) else np.nan,
                "calibration_ratio": p.mean() / y.mean() if len(y) and y.mean() else np.nan,
                "roc_auc": roc_auc_score(y, p) if enough else np.nan,
                "pr_auc": average_precision_score(y, p) if enough else np.nan,
                "brier": brier_score_loss(y, p) if len(y) else np.nan,
                "score_psi_vs_train": psi_numeric(reference_pd, g["pd"].to_numpy(), cfg.psi_bins),
            }
        )
    return pd.DataFrame(rows)


def performance_by_window(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for window in ("train", "validation", "test", "recent"):
        g = frame[(frame["sample_split"] == window) & frame["target"].notna()]
        y, p = g["target"].astype(int).to_numpy(), g["pd"].to_numpy()
        rows.append(
            {
                "window": window,
                "matured_loans": len(g),
                "bad_rate": y.mean(),
                "mean_pd": p.mean(),
                "calibration_ratio": p.mean() / y.mean(),
                "roc_auc": roc_auc_score(y, p),
                "pr_auc": average_precision_score(y, p),
                "brier": brier_score_loss(y, p),
            }
        )
    out = pd.DataFrame(rows)
    base = out.iloc[0]
    out["roc_auc_change_vs_train"] = out["roc_auc"] - base["roc_auc"]
    out["pr_auc_change_vs_train"] = out["pr_auc"] - base["pr_auc"]
    return out


def run_monitoring(con: duckdb.DuckDBPyConnection, settings: Settings) -> dict[str, Any]:
    paths = settings.paths
    dataset = load_model_dataset(con)
    predictions = pd.read_parquet(paths.evaluation / "predictions.parquet")[["loan_id", "pd"]]
    frame = dataset.merge(predictions, on="loan_id")
    frame["target"] = frame["target"].astype("Float64")

    features = feature_drift(frame, settings)
    predictions_drift = prediction_drift(frame, settings)
    monthly = monthly_monitoring(frame, settings)
    performance = performance_by_window(frame)

    for name, df in (
        ("feature_drift", features),
        ("prediction_drift", predictions_drift),
        ("monthly", monthly),
        ("performance", performance),
    ):
        con.register(f"{name}_df", df)
        con.execute(f"CREATE OR REPLACE TABLE monitoring.{name} AS SELECT * FROM {name}_df")

    payload = {
        "feature_drift": json.loads(features.to_json(orient="records")),
        "prediction_drift": json.loads(predictions_drift.to_json(orient="records")),
        "monthly": json.loads(monthly.to_json(orient="records")),
        "performance": json.loads(performance.to_json(orient="records")),
    }
    (paths.evaluation / "monitoring.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    render_monitoring_report(features, predictions_drift, monthly, performance, settings)
    current = features[features["window"] == "current"]
    log_event(
        logger,
        "monitoring complete",
        significant_features=int((current["status"] == "significant").sum()),
        moderate_features=int((current["status"] == "moderate").sum()),
    )
    return payload


def render_monitoring_report(
    features: pd.DataFrame,
    prediction: pd.DataFrame,
    monthly: pd.DataFrame,
    performance: pd.DataFrame,
    settings: Settings,
) -> None:
    cfg = settings.monitoring
    current = features[features["window"] == "current"].sort_values(
        "psi", ascending=False, kind="stable"
    )
    flagged = current[current["status"] != "stable"]
    status_counts = current["status"].value_counts().to_dict()
    recent_perf = performance.set_index("window").loc["recent"]
    lines = [
        "# Model Monitoring Report",
        "",
        GENERATED_BANNER,
        "",
        "Reference population: **train window**. `current` = all applications decided",
        "after the test window (matured or not). PSI thresholds: "
        f"< {cfg.psi_moderate} stable, {cfg.psi_moderate}-{cfg.psi_significant} moderate, "
        f"> {cfg.psi_significant} significant.",
        "",
        "## 1. Prediction drift (champion PD)",
        "",
        df_to_markdown(prediction, formats={"n": ",d", "ks_p_value": ".2e"}),
        "",
        "## 2. Performance drift (matured outcomes only)",
        "",
        df_to_markdown(performance, formats={"matured_loans": ",d"}),
        "",
        f"Most recent matured window: ROC-AUC {recent_perf['roc_auc']:.3f} "
        f"({recent_perf['roc_auc_change_vs_train']:+.3f} vs train), calibration ratio "
        f"{recent_perf['calibration_ratio']:.2f}. Performance metrics require matured",
        "outcome windows (due date + 30 days), so the latest months cannot be assessed yet.",
        "",
        "## 3. Feature drift - current window vs train",
        "",
        f"Status counts: {status_counts}.",
        "",
        df_to_markdown(
            flagged[["feature", "psi", "status", "ks_statistic", "reference_mean", "current_mean"]]
        )
        if not flagged.empty
        else "_All features stable._",
        "",
        "Top 10 features by PSI in each window:",
        "",
        df_to_markdown(
            features.sort_values("psi", ascending=False, kind="stable")
            .groupby("window")
            .head(10)[["window", "feature", "psi", "status"]]
        ),
        "",
        "## 4. Monthly monitoring",
        "",
        df_to_markdown(monthly, formats={"applications": ",d", "matured": ",d"}),
        "",
        "## 5. Interpretation guide",
        "",
        "- Score PSI and feature PSI are early-warning signals available on day one.",
        "- A falling calibration ratio with stable ROC-AUC means the *level* of risk moved",
        "  (e.g. macro stress) while rank-ordering still holds -> recalibrate, do not rebuild.",
        "- Falling ROC-AUC across several matured months would indicate the relationships",
        "  themselves changed -> investigate features and consider redevelopment.",
        "",
    ]
    path = settings.paths.reports / "monitoring_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
