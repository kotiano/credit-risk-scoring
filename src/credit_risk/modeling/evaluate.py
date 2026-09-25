"""Model validation: performance vs actual outcomes, calibration, explainability, fairness.

Reads the persisted models and predictions (``make train``) so evaluation is a
separate, repeatable step. Writes ``artifacts/evaluation/evaluation.json`` (consumed by
the dashboard) and ``reports/model_validation_report.md``.
"""

from __future__ import annotations

import json
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd

from credit_risk.config import Settings
from credit_risk.logging_utils import get_logger, log_event
from credit_risk.modeling.explain import (
    impurity_importances,
    local_explanation,
    logistic_coefficients,
    permutation_importances,
    reference_values,
)
from credit_risk.modeling.fairness import attach_attributes, fairness_review
from credit_risk.modeling.metrics import (
    calibration_table,
    classification_report,
    curve_points,
    decile_table,
    threshold_metrics,
    threshold_table,
)
from credit_risk.modeling.pipelines import feature_frame
from credit_risk.modeling.scorecard import band_performance
from credit_risk.modeling.train import LABELLED_SPLITS, load_model_dataset
from credit_risk.reporting.markdown import GENERATED_BANNER, df_to_markdown

logger = get_logger(__name__)

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "Hist Gradient Boosting",
}
OVERFIT_GAP = 0.05  # train - validation ROC-AUC gap considered material


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def evaluate_models(con: duckdb.DuckDBPyConnection, settings: Settings) -> dict[str, Any]:
    paths, cfg = settings.paths, settings.modeling
    metadata = json.loads((paths.models / "metadata.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(paths.evaluation / "predictions.parquet")
    champion, threshold = metadata["champion"], metadata["decision_threshold"]
    models = {f: joblib.load(paths.models / f"{f}.joblib") for f in MODEL_NAMES}
    labelled = predictions[predictions["sample_split"].isin(LABELLED_SPLITS)].copy()
    labelled["target"] = labelled["target"].astype(int)

    # 1. Performance by model and split ------------------------------------------------
    perf_rows = []
    for family in MODEL_NAMES:
        for split in LABELLED_SPLITS:
            d = labelled[labelled["sample_split"] == split]
            perf_rows.append(
                {
                    "model": family,
                    "split": split,
                    **classification_report(
                        d["target"].to_numpy(), d[f"pd_{family}"].to_numpy(), threshold
                    ),
                }
            )
    performance = pd.DataFrame(perf_rows)
    wide = performance.pivot_table(index="model", columns="split", values="roc_auc")
    diagnostics = pd.DataFrame(
        {
            "model": wide.index,
            "train_auc": wide["train"].to_numpy(),
            "validation_auc": wide["validation"].to_numpy(),
            "test_auc": wide["test"].to_numpy(),
            "recent_auc": wide["recent"].to_numpy(),
        }
    )
    diagnostics["overfit_gap"] = diagnostics["train_auc"] - diagnostics["validation_auc"]
    diagnostics["temporal_change"] = diagnostics["recent_auc"] - diagnostics["validation_auc"]
    diagnostics["diagnosis"] = np.where(
        diagnostics["overfit_gap"] > OVERFIT_GAP,
        "overfits (train >> validation)",
        "generalises (small train-validation gap)",
    )

    # 2. Champion vs actual outcomes ---------------------------------------------------
    test = labelled[labelled["sample_split"] == "test"]
    y_test, p_test = test["target"].to_numpy(), test["pd"].to_numpy()
    deciles = decile_table(y_test, p_test)
    calibration = {
        split: calibration_table(
            labelled.loc[labelled["sample_split"] == split, "target"].to_numpy(),
            labelled.loc[labelled["sample_split"] == split, "pd"].to_numpy(),
        )
        for split in LABELLED_SPLITS
    }
    bands = pd.concat(
        [
            band_performance(labelled[labelled["sample_split"] == s], cfg).assign(split=s)
            for s in LABELLED_SPLITS
        ],
        ignore_index=True,
    )
    thresholds = threshold_table(y_test, p_test)
    confusion = threshold_metrics(y_test, p_test, threshold)
    curves = {
        family: curve_points(y_test, test[f"pd_{family}"].to_numpy()) for family in MODEL_NAMES
    }
    pd_distribution = {
        "bins": np.linspace(0, 1, 41).round(3).tolist(),
        "good": np.histogram(p_test[y_test == 0], bins=np.linspace(0, 1, 41))[0].tolist(),
        "bad": np.histogram(p_test[y_test == 1], bins=np.linspace(0, 1, 41))[0].tolist(),
    }

    # 3. Explainability ------------------------------------------------------------------
    dataset = load_model_dataset(con)
    train_rows = dataset[dataset["sample_split"] == "train"]
    valid_rows = dataset[dataset["sample_split"] == "validation"]
    X_train = feature_frame(train_rows)  # noqa: N806
    X_valid = feature_frame(valid_rows)  # noqa: N806
    y_valid = valid_rows["target"].astype(int)
    perm = pd.concat(
        [
            permutation_importances(m, X_valid, y_valid, cfg.random_state).assign(model=f)
            for f, m in models.items()
        ],
        ignore_index=True,
    )
    coefficients = logistic_coefficients(models["logistic_regression"])
    impurity = impurity_importances(models["random_forest"])

    reference = reference_values(X_train)
    test_rows = dataset[dataset["loan_id"].isin(test["loan_id"])].reset_index(drop=True)
    test_rows = test_rows.merge(test[["loan_id", "pd"]], on="loan_id")
    examples = {}
    for label, q in (("high_risk", 0.98), ("median_risk", 0.50), ("low_risk", 0.02)):
        pds = test_rows["pd"].to_numpy(dtype=float)
        idx = int(np.abs(pds - np.quantile(pds, q)).argmin())
        record = test_rows.iloc[idx]
        row = feature_frame(test_rows.iloc[[idx]])
        expl = local_explanation(models[champion], row, reference)
        examples[label] = {
            "loan_id": str(record["loan_id"]),
            "pd": float(pds[idx]),
            "actual_outcome": int(record["target"]),
            "contributions": _records(expl.astype({"value": str, "reference": str})),
        }

    # 4. Fairness ------------------------------------------------------------------------
    fairness = fairness_review(attach_attributes(con, test), threshold)

    payload: dict[str, Any] = {
        "metadata": metadata,
        "performance": _records(performance),
        "diagnostics": _records(diagnostics),
        "deciles_test": _records(deciles),
        "calibration": {k: _records(v) for k, v in calibration.items()},
        "risk_bands": _records(bands),
        "thresholds_test": _records(thresholds),
        "confusion_test": confusion.__dict__,
        "curves_test": curves,
        "pd_distribution_test": pd_distribution,
        "permutation_importance": _records(perm),
        "logistic_coefficients": _records(coefficients),
        "rf_impurity_importance": _records(impurity.head(25)),
        "local_explanations": examples,
        "fairness_test": _records(fairness),
    }
    (paths.evaluation / "evaluation.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    render_validation_report(payload, settings)
    log_event(
        logger,
        "evaluation complete",
        champion=champion,
        test_auc=round(
            float(performance.query("model == @champion and split == 'test'")["roc_auc"].iloc[0]), 4
        ),
    )
    return payload


def _conclusions(diag: pd.DataFrame, champ_perf: pd.DataFrame, champion: str) -> list[str]:
    """Data-driven conclusions so the narrative can never contradict the metrics."""
    out = []
    aucs = champ_perf.loc[list(LABELLED_SPLITS), "roc_auc"]
    out.append(
        f"- {MODEL_NAMES[champion]} ROC-AUC ranges {aucs.min():.3f}-{aucs.max():.3f} across "
        f"train/validation/test/recent (spread {aucs.max() - aucs.min():.3f})."
    )
    overfit = diag[diag["overfit_gap"] > OVERFIT_GAP]
    if not overfit.empty:
        out.append(
            "- Over-fitting: "
            + ", ".join(
                f"{r.model} (train-validation gap {r.overfit_gap:.3f})"
                for r in overfit.itertuples()
            )
            + ". Higher training fit did not translate into better out-of-time ranking."
        )
    ratio = float(
        champ_perf["calibration_ratio"].to_numpy(dtype=float)[
            list(champ_perf.index).index("recent")
        ]
    )
    if ratio < 0.95:
        out.append(
            f"- The champion under-predicts default in the most recent matured window "
            f"(mean PD / bad rate = {ratio:.2f}). Rank-ordering is retained, so the "
            "proportionate response is recalibration on recent matured outcomes rather "
            "than redevelopment (see reports/monitoring_report.md)."
        )
    elif ratio > 1.05:
        out.append(f"- The champion over-predicts default recently (ratio {ratio:.2f}).")
    else:
        out.append(f"- Calibration-in-the-large holds recently (ratio {ratio:.2f}).")
    out.append(
        "- All figures come from synthetic data and illustrate the validation process; "
        "they are not evidence of real-world predictive power."
    )
    return out


def render_validation_report(payload: dict[str, Any], settings: Settings) -> None:
    meta = payload["metadata"]
    champion = meta["champion"]
    perf = pd.DataFrame(payload["performance"])
    perf["model"] = perf["model"].map(MODEL_NAMES)
    cols = [
        "model",
        "split",
        "n",
        "bad_rate",
        "mean_pd",
        "roc_auc",
        "gini",
        "ks",
        "pr_auc",
        "brier",
        "calibration_ratio",
        "ece",
    ]
    diag = pd.DataFrame(payload["diagnostics"])
    diag["model"] = diag["model"].map(MODEL_NAMES)
    deciles = pd.DataFrame(payload["deciles_test"])
    bands = pd.DataFrame(payload["risk_bands"])
    thresholds = pd.DataFrame(payload["thresholds_test"])
    perm = pd.DataFrame(payload["permutation_importance"])
    perm_champ = perm[perm["model"] == champion].head(15)
    coefs = pd.DataFrame(payload["logistic_coefficients"]).head(15)
    fair = pd.DataFrame(payload["fairness_test"])
    conf = payload["confusion_test"]
    champ_perf = perf[perf["model"] == MODEL_NAMES[champion]].set_index("split")
    cal = {k: pd.DataFrame(v) for k, v in payload["calibration"].items()}

    local_sections = []
    for label, ex in payload["local_explanations"].items():
        contrib = pd.DataFrame(ex["contributions"])
        local_sections += [
            f"**{label.replace('_', ' ').title()}** — loan `{ex['loan_id']}`, "
            f"predicted PD {ex['pd']:.1%}, actual outcome: "
            f"{'bad (30+ DPD)' if ex['actual_outcome'] else 'good'}",
            "",
            df_to_markdown(contrib, formats={"pd_contribution": "+.4f"}),
            "",
        ]

    lines = [
        "# Model Validation Report",
        "",
        GENERATED_BANNER,
        "",
        "## 1. Scope",
        "",
        f"- **Target:** {meta['target']} within the performance window "
        "(disbursement to due date + 30 days).",
        f"- **Windows (out-of-time):** train `{meta['windows']['train']}`, validation "
        f"`{meta['windows']['validation']}`, test `{meta['windows']['test']}`; `recent` = "
        "matured loans decided after the test window (used for monitoring).",
        f"- **Sample sizes:** {meta['sample_sizes']}",
        f"- **Champion:** {MODEL_NAMES[champion]}. {meta['selection_rationale']}",
        f"- **Reference threshold:** PD >= {meta['decision_threshold']:.3f} "
        f"({meta['threshold_rule']}). Analytical only - not a lending policy.",
        "",
        "## 2. Discrimination and calibration by model and window",
        "",
        df_to_markdown(perf[cols], formats={"n": ",d"}),
        "",
        "## 3. Over-fitting and temporal stability (ROC-AUC)",
        "",
        df_to_markdown(diag),
        "",
        "Reading: a large train-validation gap indicates over-fitting; a negative",
        "`temporal_change` (recent - validation) indicates degradation over time.",
        "",
        "## 4. Predicted risk vs actual outcomes (champion, test window)",
        "",
        "### Risk deciles (1 = highest predicted PD)",
        "",
        df_to_markdown(deciles, formats={"population": ",d", "bads": ",d"}),
        "",
        "### Analytical risk bands",
        "",
        df_to_markdown(
            bands, formats={"population": ",d", "min_score": ",.0f", "max_score": ",.0f"}
        ),
        "",
        "### Calibration (equal-frequency PD bins)",
        "",
        "Test window:",
        "",
        df_to_markdown(cal["test"], formats={"n": ",d", "bads": ",d"}),
        "",
        f"Calibration-in-the-large (mean PD / observed bad rate): train "
        f"{champ_perf.loc['train', 'calibration_ratio']:.3f}, validation "
        f"{champ_perf.loc['validation', 'calibration_ratio']:.3f}, test "
        f"{champ_perf.loc['test', 'calibration_ratio']:.3f}, recent "
        f"{champ_perf.loc['recent', 'calibration_ratio']:.3f}. Values below 1 mean the "
        "model under-predicts default in that period.",
        "",
        "### Confusion matrix at the reference threshold (test)",
        "",
        "| | Predicted bad | Predicted good |",
        "|---|---|---|",
        f"| **Actual bad** | {conf['tp']:,} (TP) | {conf['fn']:,} (FN) |",
        f"| **Actual good** | {conf['fp']:,} (FP) | {conf['tn']:,} (TN) |",
        "",
        f"Precision {conf['precision']:.3f}, recall/sensitivity {conf['recall']:.3f}, "
        f"specificity {conf['specificity']:.3f}, F1 {conf['f1']:.3f}.",
        "",
        "### Threshold analysis (test)",
        "",
        df_to_markdown(thresholds.drop(columns=["tp", "fp", "tn", "fn"])),
        "",
        "## 5. Explainability",
        "",
        f"### Permutation importance - {MODEL_NAMES[champion]} (validation ROC-AUC drop)",
        "",
        df_to_markdown(perm_champ[["feature", "group", "auc_drop_mean", "auc_drop_std"]]),
        "",
        "### Logistic regression - largest standardised coefficients",
        "",
        df_to_markdown(coefs[["term", "coefficient", "odds_ratio_per_sd"]]),
        "",
        "### Individual explanations (baseline substitution, champion)",
        "",
        "`pd_contribution` = PD(actual) - PD(with the feature set to the training",
        "median/mode). Positive values increase predicted risk.",
        "",
        *local_sections,
        "## 6. Responsible-modeling review (test window)",
        "",
        "Gender, age and region are **not model inputs**; they are joined only to check",
        "for proxy effects. `flagged_rate` is the share at or above the reference threshold.",
        "",
        df_to_markdown(fair, formats={"n": ",d"}),
        "",
        "## 7. Conclusions (derived from the numbers above)",
        "",
        *_conclusions(diag, champ_perf, champion),
        "",
    ]
    path = settings.paths.reports / "model_validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
