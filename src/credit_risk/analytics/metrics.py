"""Python implementation of the lending metrics defined in docs/metrics.md.

The SQL marts are the system of record. These functions exist so that (a) the dashboard
can recompute metrics for any filter combination from loan-level rows, and (b) tests
can assert SQL/Python parity. Both read their thresholds from
:class:`credit_risk.config.RiskDefinitions`, so the definitions cannot drift apart.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from credit_risk.config import RiskDefinitions

ON_BOOK_STATES = ("active", "overdue")


def days_past_due(
    due_date: pd.Series, settled_date: pd.Series, as_of: date | pd.Timestamp
) -> pd.Series:
    """Contractual loan-level DPD at the end of ``as_of``.

    DPD = as_of - due_date while the loan is unsettled at ``as_of`` and past due, else 0.
    ``settled_date`` is the payoff date (NaT when not settled).
    """
    as_of_ts = pd.Timestamp(as_of)
    due = pd.to_datetime(due_date)
    settled = pd.to_datetime(settled_date)
    unsettled = settled.isna() | (settled > as_of_ts)
    dpd = (as_of_ts - due).dt.days.clip(lower=0)
    return dpd.where(unsettled, 0).astype(int)


def dpd_bucket(dpd: pd.Series | np.ndarray, risk: RiskDefinitions) -> np.ndarray:
    lowers = np.array([lb for lb, _ in risk.dpd_buckets])
    labels = np.array(risk.bucket_labels)
    idx = np.searchsorted(lowers, np.asarray(dpd), side="right") - 1
    return labels[np.clip(idx, 0, len(labels) - 1)]


def portfolio_at_risk(loans: pd.DataFrame, threshold: int) -> float:
    """PAR_n = principal of on-book loans with DPD >= n / principal of on-book loans.

    Expects columns ``outstanding_principal``, ``dpd`` and ``loan_state``. Settled and
    written-off loans are excluded from numerator and denominator.
    """
    on_book = loans[loans["loan_state"].isin(ON_BOOK_STATES)]
    denominator = on_book["outstanding_principal"].sum()
    if denominator == 0:
        return float("nan")
    numerator = on_book.loc[on_book["dpd"] >= threshold, "outstanding_principal"].sum()
    return float(numerator / denominator)


def par_profile(loans: pd.DataFrame, risk: RiskDefinitions) -> dict[str, float]:
    return {f"par{t}": portfolio_at_risk(loans, t) for t in risk.par_thresholds}


def repayment_rate(loans: pd.DataFrame, as_of: date | pd.Timestamp) -> float:
    """Amount repaid / amount due over loans whose due date has passed."""
    matured = loans[pd.to_datetime(loans["due_date"]) <= pd.Timestamp(as_of)]
    due = matured["total_due"].sum()
    return float(matured["total_paid"].sum() / due) if due else float("nan")


def on_time_collection_rate(loans: pd.DataFrame, as_of: date | pd.Timestamp) -> float:
    """Amount repaid by the due date / amount due over matured loans."""
    matured = loans[pd.to_datetime(loans["due_date"]) <= pd.Timestamp(as_of)]
    due = matured["total_due"].sum()
    return float(matured["paid_by_due_date"].sum() / due) if due else float("nan")


def completion_rate(loans: pd.DataFrame, as_of: date | pd.Timestamp) -> float:
    matured = loans[pd.to_datetime(loans["due_date"]) <= pd.Timestamp(as_of)]
    return float(matured["is_settled"].mean()) if len(matured) else float("nan")


def timing_rates(loans: pd.DataFrame) -> dict[str, float]:
    """Early / on-time / late shares among settled loans."""
    settled = loans[loans["is_settled"]]
    if settled.empty:
        return {"early": float("nan"), "on_time": float("nan"), "late": float("nan")}
    shares = settled["repayment_timing"].value_counts(normalize=True)
    return {k: float(shares.get(k, 0.0)) for k in ("early", "on_time", "late")}


def default_rate(loans: pd.DataFrame) -> float:
    """Share of loans reaching default DPD, among loans old enough to have done so."""
    observable = loans[loans["is_default_observable"]]
    return float(observable["is_default"].mean()) if len(observable) else float("nan")


def bad_rate(loans: pd.DataFrame) -> float:
    """Share of matured loans that reached the target DPD (the model target)."""
    observable = loans[loans["is_target_observable"]]
    return float(observable["bad_30dpd"].mean()) if len(observable) else float("nan")


def delinquency_stats(loans: pd.DataFrame) -> dict[str, float]:
    on_book = loans[loans["loan_state"].isin(ON_BOOK_STATES)]
    delinquent = on_book[on_book["dpd"] > 0]
    return {
        "delinquency_rate": len(delinquent) / len(on_book) if len(on_book) else float("nan"),
        "avg_dpd": float(delinquent["dpd"].mean()) if len(delinquent) else 0.0,
        "median_dpd": float(delinquent["dpd"].median()) if len(delinquent) else 0.0,
        "max_dpd": float(delinquent["dpd"].max()) if len(delinquent) else 0.0,
    }


def roll_rate_matrix(transitions: pd.DataFrame, weight: str = "n_loans") -> pd.DataFrame:
    """Row-normalised migration matrix from ``from_state``/``to_state`` rows."""
    matrix = transitions.pivot_table(
        index="from_state", columns="to_state", values=weight, aggfunc="sum", fill_value=0
    )
    return matrix.div(matrix.sum(axis=1), axis=0)


def portfolio_kpis(
    loans: pd.DataFrame, risk: RiskDefinitions, as_of: date | pd.Timestamp
) -> dict[str, float]:
    """Headline KPIs for any subset of ``marts.loan_repayment_summary`` rows."""
    on_book = loans[loans["loan_state"].isin(ON_BOOK_STATES)]
    return {
        "total_borrowers": float(loans["borrower_id"].nunique()),
        "active_borrowers": float(on_book["borrower_id"].nunique()),
        "total_loans": float(len(loans)),
        "active_loans": float(len(on_book)),
        "total_disbursed": float(loans["loan_amount"].sum()),
        "outstanding_principal": float(on_book["outstanding_principal"].sum()),
        "total_collections": float(loans["total_paid"].sum()),
        "repayment_rate": repayment_rate(loans, as_of),
        "on_time_collection_rate": on_time_collection_rate(loans, as_of),
        "completion_rate": completion_rate(loans, as_of),
        "default_rate": default_rate(loans),
        "bad_30dpd_rate": bad_rate(loans),
        **par_profile(loans, risk),
        **delinquency_stats(loans),
    }
