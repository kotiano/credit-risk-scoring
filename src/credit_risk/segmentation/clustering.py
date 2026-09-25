"""Unsupervised behavioural segmentation compared with the rule-based segments.

Rule-based segments (``marts.borrower_mart.risk_segment``, defined in SQL) are
transparent business rules. K-Means is run on scaled behavioural features to check
whether the data contain natural groupings the rules miss or split differently.

Choices (docs/metrics.md#segmentation):
* Features describe *behaviour* only (repayment, delinquency, activity), never
  demographics, so clusters cannot become proxies for protected groups.
* Heavy-tailed counts and DPD are log1p-transformed, then standardised, because
  K-Means uses Euclidean distance and would otherwise be dominated by scale.
* Silhouette scores for k = 3..8 are reported. On this data they rise slowly and
  monotonically with no clear optimum (a continuous behaviour space rather than
  well-separated groups), so k is fixed at 6 to match the number of rule segments and
  allow a like-for-like comparison. A clear silhouette peak would take precedence.
* Agreement with the rules is measured with the Adjusted Rand Index (ARI) and
  normalised mutual information (NMI); the cross-tab explains where they differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from credit_risk.config import Settings
from credit_risk.logging_utils import get_logger, log_event
from credit_risk.reporting.markdown import GENERATED_BANNER, df_to_markdown

logger = get_logger(__name__)

CLUSTER_FEATURES = {
    # feature: log-transform?
    "total_loans": True,
    "repayment_rate": False,
    "late_payment_ratio": False,
    "max_dpd_ever": True,
    "current_dpd": True,
    "avg_repayment_timing_days": False,
    "loans_reached_30dpd": True,
    "repayment_frequency_30d": True,
    "days_since_last_loan": True,
}
COMPARISON_K = 6
CLEAR_PEAK_MARGIN = 0.05  # a silhouette optimum must beat k=6 by this much to override

SEGMENT_ORDER = [
    "Low Risk / Strong Repayer",
    "Low Risk / New Borrower",
    "Moderate Risk",
    "Emerging Delinquency",
    "High Risk / Persistent Delinquency",
    "Default / Recovery",
]


@dataclass
class SegmentationResult:
    assignments: pd.DataFrame
    k: int
    silhouette_by_k: dict[int, float]
    ari: float
    nmi: float
    profiles: pd.DataFrame
    crosstab: pd.DataFrame
    segment_summary: pd.DataFrame


def prepare_matrix(borrowers: pd.DataFrame) -> np.ndarray:
    X = pd.DataFrame(index=borrowers.index)  # noqa: N806
    for col, log in CLUSTER_FEATURES.items():
        values = borrowers[col].astype(float)
        # Borrowers with no matured/settled loans have no timing or ratio history;
        # neutral imputation (0 = on time / no late loans / fully repaid so far).
        fill = 1.0 if col == "repayment_rate" else 0.0
        values = values.fillna(fill)
        X[col] = np.log1p(values.clip(lower=0)) if log else values
    X["avg_repayment_timing_days"] = X["avg_repayment_timing_days"].clip(-30, 120)
    return StandardScaler().fit_transform(X.to_numpy())


def run_segmentation(con: duckdb.DuckDBPyConnection, settings: Settings) -> SegmentationResult:
    seed = settings.modeling.random_state
    borrowers = con.execute("SELECT * FROM marts.borrower_mart ORDER BY borrower_id").df()
    X = prepare_matrix(borrowers)  # noqa: N806

    rng = np.random.default_rng(seed)
    sample = rng.choice(len(X), size=min(3000, len(X)), replace=False)
    silhouettes = {}
    for k in range(3, 9):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        silhouettes[k] = float(silhouette_score(X[sample], labels[sample]))
    peak_k = max(silhouettes, key=lambda k: silhouettes[k])
    interior_peak = peak_k not in (min(silhouettes), max(silhouettes))
    clear_peak = silhouettes[peak_k] - silhouettes[COMPARISON_K] > CLEAR_PEAK_MARGIN
    best_k = peak_k if interior_peak and clear_peak else COMPARISON_K
    model = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit(X)
    borrowers["cluster"] = model.labels_

    # Order clusters by observed default share so "Cluster 1" is the lowest-risk group.
    risk = borrowers.groupby("cluster")["n_defaults"].apply(lambda s: (s > 0).mean())
    remap = {old: new for new, old in enumerate(risk.sort_values().index, start=1)}
    borrowers["cluster"] = borrowers["cluster"].map(remap).map(lambda c: f"Cluster {c}")

    profiles = (
        borrowers.groupby("cluster")
        .agg(
            borrowers=("borrower_id", "size"),
            total_loans=("total_loans", "median"),
            repayment_rate=("repayment_rate", "mean"),
            late_payment_ratio=("late_payment_ratio", "mean"),
            max_dpd_ever=("max_dpd_ever", "median"),
            current_dpd=("current_dpd", "mean"),
            share_ever_defaulted=("n_defaults", lambda s: (s > 0).mean()),
            dominant_rule_segment=("risk_segment", lambda s: s.mode().iloc[0]),
        )
        .reset_index()
    )
    crosstab = pd.crosstab(borrowers["risk_segment"], borrowers["cluster"]).reindex(
        [s for s in SEGMENT_ORDER if s in set(borrowers["risk_segment"])]
    )
    summary = (
        borrowers.groupby("risk_segment")
        .agg(
            borrowers=("borrower_id", "size"),
            total_loans=("total_loans", "sum"),
            outstanding_principal=("outstanding_principal", "sum"),
            repayment_rate=("repayment_rate", "mean"),
            late_payment_ratio=("late_payment_ratio", "mean"),
            share_ever_30dpd=("loans_reached_30dpd", lambda s: (s > 0).mean()),
        )
        .reindex([s for s in SEGMENT_ORDER if s in set(borrowers["risk_segment"])])
        .reset_index()
    )
    summary["borrower_share"] = summary["borrowers"] / summary["borrowers"].sum()

    ari = float(adjusted_rand_score(borrowers["risk_segment"], borrowers["cluster"]))
    nmi = float(normalized_mutual_info_score(borrowers["risk_segment"], borrowers["cluster"]))
    assignments = borrowers[["borrower_id", "risk_segment", "cluster"]]
    con.register("assign_df", assignments)
    con.execute("CREATE OR REPLACE TABLE marts.borrower_clusters AS SELECT * FROM assign_df")
    log_event(logger, "segmentation complete", k=best_k, ari=round(ari, 3), nmi=round(nmi, 3))
    result = SegmentationResult(
        assignments, best_k, silhouettes, ari, nmi, profiles, crosstab, summary
    )
    render_segmentation_report(result, settings)
    return result


def render_segmentation_report(result: SegmentationResult, settings: Settings) -> None:
    sil = pd.DataFrame(
        {"k": list(result.silhouette_by_k), "silhouette": list(result.silhouette_by_k.values())}
    )
    lines = [
        "# Borrower Segmentation Report",
        "",
        GENERATED_BANNER,
        "",
        "Borrower behaviour is measured **as of the snapshot date** from",
        "`marts.borrower_mart`. Segments are descriptive monitoring groups, not credit",
        "decisions, and the labels are analytical conventions rather than established truths.",
        "",
        "## 1. Rule-based segments",
        "",
        "Rules are evaluated top-down (first match wins) in `sql/marts/07_borrower_mart.sql`:",
        "",
        "1. **Default / Recovery** - any loan reached 90+ DPD.",
        "2. **High Risk / Persistent Delinquency** - currently 30+ DPD, or 2+ loans reached 30+ DPD.",
        "3. **Emerging Delinquency** - currently 1-29 DPD, or the latest settled loan reached 7+ DPD.",
        "4. **Moderate Risk** - ever 7+ DPD, or more than 1/3 of settled loans repaid late.",
        "5. **Low Risk / New Borrower** - at most one settled loan and none of the above.",
        "6. **Low Risk / Strong Repayer** - everyone else (2+ settled loans, clean history).",
        "",
        df_to_markdown(
            result.segment_summary,
            formats={
                "borrowers": ",d",
                "total_loans": ",d",
                "outstanding_principal": ",.0f",
                "borrower_share": ".1%",
            },
        ),
        "",
        "## 2. K-Means on behavioural features",
        "",
        f"Features (log1p for counts/DPD, then standardised): {', '.join(CLUSTER_FEATURES)}.",
        "",
        df_to_markdown(sil, formats={"k": "d"}),
        "",
        f"Selected **k = {result.k}**. Silhouette scores are low (< 0.4) and change slowly",
        "with k, i.e. borrower behaviour forms a continuum rather than well-separated",
        "groups. Unless an interior silhouette peak clearly dominates, k is fixed at 6 to",
        "compare like-for-like with the six rule segments. Clusters are numbered by",
        "ascending share of borrowers who ever defaulted.",
        "",
        df_to_markdown(result.profiles, formats={"borrowers": ",d"}),
        "",
        "## 3. Rules vs clusters",
        "",
        f"Adjusted Rand Index = **{result.ari:.3f}**, normalised mutual information = "
        f"**{result.nmi:.3f}**.",
        "",
        df_to_markdown(
            result.crosstab.reset_index(), formats={c: ",d" for c in result.crosstab.columns}
        ),
        "",
        "### How to read the differences",
        "",
        "- Rules encode *severity and recency* with hard thresholds (e.g. 7 or 30 DPD);",
        "  K-Means groups by overall *similarity*, so borrowers just either side of a",
        "  threshold can share a cluster while the rules separate them.",
        "- Tenure and activity (number of loans, borrowing frequency) are strong axes of",
        "  variation for K-Means but only a tie-breaker in the rules, so clusters tend to",
        "  split low-risk borrowers by engagement rather than by delinquency.",
        "- Rules are preferred for monitoring because they are stable and explainable;",
        "  clustering is used as a diagnostic that the rule boundaries are not arbitrary.",
        "",
    ]
    path = settings.paths.reports / "segmentation_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
