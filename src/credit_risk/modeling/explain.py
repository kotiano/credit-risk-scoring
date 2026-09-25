"""Model explainability: global importance and individual (local) explanations.

* Logistic regression: standardised coefficients (log-odds change per 1 s.d.).
* All models: permutation importance on the validation window (drop in ROC-AUC when a
  raw input feature is shuffled) - model-agnostic and computed out-of-sample.
* Random forest: impurity importance (reported, but biased toward high-cardinality
  features, so permutation importance is the primary measure).
* Local explanations: baseline substitution. For one application, each feature is
  replaced by its training-population reference value (median / mode) and the change
  in predicted PD is recorded. Positive values push PD up. This is model-agnostic,
  simple to audit, and needs no additional dependency (SHAP was considered; see
  docs/modeling.md). Only model features are shown - no demographics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from credit_risk.features.registry import FEATURES, categorical_features, numeric_features

FEATURE_GROUP = {f.name: f.group.value for f in FEATURES}


def logistic_coefficients(model: Pipeline) -> pd.DataFrame:
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["model"].coef_.ravel()
    df = pd.DataFrame({"term": names, "coefficient": coefs})
    df["odds_ratio_per_sd"] = np.exp(df["coefficient"])
    df["abs_coefficient"] = df["coefficient"].abs()
    return df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def permutation_importances(
    model: Pipeline,
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,
    random_state: int,
    n_repeats: int = 5,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    df = pd.DataFrame(
        {
            "feature": X.columns,
            "auc_drop_mean": result.importances_mean,
            "auc_drop_std": result.importances_std,
        }
    )
    df["group"] = df["feature"].map(FEATURE_GROUP)
    return df.sort_values("auc_drop_mean", ascending=False).reset_index(drop=True)


def impurity_importances(model: Pipeline) -> pd.DataFrame:
    names = model.named_steps["prep"].get_feature_names_out()
    return (
        pd.DataFrame({"term": names, "importance": model.named_steps["model"].feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def reference_values(X_train: pd.DataFrame) -> pd.Series:  # noqa: N803
    ref = {c: X_train[c].median() for c in numeric_features()}
    ref.update({c: X_train[c].mode().iloc[0] for c in categorical_features()})
    return pd.Series(ref)


def local_explanation(
    model: Pipeline, row: pd.DataFrame, reference: pd.Series, top_k: int = 6
) -> pd.DataFrame:
    """Baseline-substitution contributions for a single application (one-row frame)."""
    base_pd = float(model.predict_proba(row)[:, 1][0])
    variants = pd.concat([row] * len(reference), ignore_index=True)
    for i, feature in enumerate(reference.index):
        variants.loc[i, feature] = reference[feature]
    substituted = model.predict_proba(variants)[:, 1]
    out = pd.DataFrame(
        {
            "feature": reference.index,
            "value": [row.iloc[0][f] for f in reference.index],
            "reference": reference.to_numpy(),
            "pd_contribution": base_pd - substituted,
        }
    )
    out["abs"] = out["pd_contribution"].abs()
    return out.sort_values("abs", ascending=False).head(top_k).drop(columns="abs")
