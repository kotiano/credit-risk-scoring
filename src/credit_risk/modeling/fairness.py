"""Responsible-modeling review: performance across borrower groups.

Protected / sensitive attributes (gender, age, region) are NOT model inputs. They are
joined here only to test whether the model behaves differently across groups through
proxies. Results are descriptive: on synthetic data they demonstrate the review
process, not a fairness certification (docs/limitations.md).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MIN_GROUP_SIZE = 50


def attach_attributes(con: duckdb.DuckDBPyConnection, predictions: pd.DataFrame) -> pd.DataFrame:
    attrs = con.execute(
        """
        SELECT
            l.loan_id,
            b.gender,
            b.region,
            CASE
                WHEN b.age + date_diff('year', b.onboarding_date, l.disbursement_date) < 25
                    THEN '18-24'
                WHEN b.age + date_diff('year', b.onboarding_date, l.disbursement_date) < 35
                    THEN '25-34'
                WHEN b.age + date_diff('year', b.onboarding_date, l.disbursement_date) < 45
                    THEN '35-44'
                WHEN b.age + date_diff('year', b.onboarding_date, l.disbursement_date) < 55
                    THEN '45-54'
                ELSE '55+'
            END AS age_band
        FROM clean.loans AS l
        JOIN clean.borrowers AS b USING (borrower_id)
        """
    ).df()
    return predictions.merge(attrs, on="loan_id", how="left")


def group_metrics(
    frame: pd.DataFrame, attribute: str, threshold: float, pd_col: str = "pd"
) -> pd.DataFrame:
    rows = []
    for group, g in frame.groupby(attribute):
        y = g["target"].astype(int).to_numpy()
        p = g[pd_col].to_numpy()
        flagged = p >= threshold
        auc = roc_auc_score(y, p) if len(g) >= MIN_GROUP_SIZE and 0 < y.sum() < len(y) else np.nan
        rows.append(
            {
                "attribute": attribute,
                "group": group,
                "n": len(g),
                "bad_rate": y.mean(),
                "mean_pd": p.mean(),
                "calibration_ratio": p.mean() / y.mean() if y.mean() else np.nan,
                "roc_auc": auc,
                "flagged_rate": flagged.mean(),
                "tpr": flagged[y == 1].mean() if (y == 1).any() else np.nan,
                "fpr": flagged[y == 0].mean() if (y == 0).any() else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    overall_flag = (frame[pd_col] >= threshold).mean()
    out["flagged_rate_ratio"] = out["flagged_rate"] / overall_flag if overall_flag else np.nan
    out["small_sample"] = out["n"] < MIN_GROUP_SIZE
    return out


def fairness_review(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return pd.concat(
        [group_metrics(frame, attr, threshold) for attr in ("gender", "age_band", "region")],
        ignore_index=True,
    )
