"""Deterministic, hand-computed cases for the Python metric layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.analytics import metrics as lm
from credit_risk.config import ModelingConfig, RiskDefinitions
from credit_risk.modeling.metrics import decile_table, ks_statistic, threshold_metrics
from credit_risk.modeling.scorecard import assign_risk_band, pd_to_score, score_to_pd
from credit_risk.monitoring.drift import psi_categorical, psi_numeric

RISK = RiskDefinitions()


def book(rows: list[tuple[float, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["outstanding_principal", "dpd", "loan_state"])


def test_par30_worked_example_from_spec() -> None:
    # Loan A: outstanding 1000 at 35 DPD; Loan B: 2000 current -> PAR30 = 1000/3000.
    loans = book([(1000, 35, "overdue"), (2000, 0, "active")])
    assert lm.portfolio_at_risk(loans, 30) == pytest.approx(1 / 3)
    assert lm.portfolio_at_risk(loans, 60) == 0.0


@pytest.mark.parametrize(("dpd", "in_par30"), [(29, False), (30, True), (31, True)])
def test_par_threshold_is_inclusive(dpd: int, in_par30: bool) -> None:
    loans = book([(500, dpd, "overdue"), (500, 0, "active")])
    assert lm.portfolio_at_risk(loans, 30) == pytest.approx(0.5 if in_par30 else 0.0)


def test_par_excludes_written_off_and_repaid_loans() -> None:
    loans = book(
        [(1000, 35, "overdue"), (2000, 0, "active"), (5000, 200, "written_off"), (0, 0, "repaid")]
    )
    assert lm.portfolio_at_risk(loans, 30) == pytest.approx(1 / 3)
    assert lm.portfolio_at_risk(loans, 90) == 0.0


def test_par_is_nan_for_empty_book() -> None:
    assert np.isnan(lm.portfolio_at_risk(book([(0, 0, "repaid")]), 30))


def test_par_equals_sum_of_buckets_at_or_above_threshold() -> None:
    rng = np.random.default_rng(0)
    dpd = rng.integers(0, 179, size=500)
    loans = book(
        [
            (float(p), int(d), "overdue" if d else "active")
            for p, d in zip(rng.uniform(100, 5000, 500), dpd, strict=True)
        ]
    )
    buckets = lm.dpd_bucket(loans["dpd"], RISK)
    share = (
        loans.groupby(buckets)["outstanding_principal"].sum() / loans["outstanding_principal"].sum()
    )
    assert lm.portfolio_at_risk(loans, 30) == pytest.approx(share[["30-59", "60-89", "90+"]].sum())


@pytest.mark.parametrize(
    ("dpd", "bucket"),
    [
        (0, "current"),
        (1, "1-6"),
        (6, "1-6"),
        (7, "7-29"),
        (29, "7-29"),
        (30, "30-59"),
        (59, "30-59"),
        (60, "60-89"),
        (89, "60-89"),
        (90, "90+"),
        (400, "90+"),
    ],
)
def test_dpd_bucket_boundaries(dpd: int, bucket: str) -> None:
    assert lm.dpd_bucket(np.array([dpd]), RISK)[0] == bucket


def test_days_past_due_edge_cases() -> None:
    as_of = pd.Timestamp("2025-12-31")
    due = pd.Series(
        pd.to_datetime(["2025-11-26", "2026-01-15", "2025-12-31", "2025-11-01", "2025-11-01"])
    )
    settled = pd.Series(pd.to_datetime([None, None, None, "2025-11-10", "2026-01-05"]))
    # overdue 35 | not yet due | due today (0) | settled before as_of | settled after as_of
    assert lm.days_past_due(due, settled, as_of).tolist() == [35, 0, 0, 0, 60]


def test_repayment_and_collection_rates_use_matured_loans_only() -> None:
    loans = pd.DataFrame(
        {
            "due_date": pd.to_datetime(["2025-01-10", "2025-02-10", "2026-02-01"]),
            "total_due": [1000.0, 1000.0, 5000.0],
            "total_paid": [1000.0, 500.0, 0.0],
            "paid_by_due_date": [1000.0, 0.0, 0.0],
            "is_settled": [True, False, False],
        }
    )
    as_of = pd.Timestamp("2025-12-31")
    assert lm.repayment_rate(loans, as_of) == pytest.approx(0.75)
    assert lm.on_time_collection_rate(loans, as_of) == pytest.approx(0.5)
    assert lm.completion_rate(loans, as_of) == pytest.approx(0.5)


def test_roll_rate_matrix_rows_sum_to_one() -> None:
    t = pd.DataFrame(
        {
            "from_state": ["current", "current", "1-6", "1-6"],
            "to_state": ["current", "7-29", "repaid", "7-29"],
            "n_loans": [3, 1, 1, 1],
        }
    )
    m = lm.roll_rate_matrix(t)
    assert m.loc["current", "current"] == pytest.approx(0.75)
    assert np.allclose(m.sum(axis=1), 1.0)


def test_ks_and_threshold_metrics_known_values() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert ks_statistic(y, p) == pytest.approx(1.0)
    m = threshold_metrics(y, np.array([0.1, 0.6, 0.3, 0.7, 0.4, 0.9]), 0.5)
    assert (m.tp, m.fp, m.tn, m.fn) == (2, 1, 2, 1)
    assert m.precision == pytest.approx(2 / 3)
    assert m.specificity == pytest.approx(2 / 3)


def test_decile_table_gains_reach_one_and_lift_is_consistent() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < p).astype(int)
    d = decile_table(y, p)
    assert d["cum_bads_captured_pct"].iloc[-1] == pytest.approx(1.0)
    assert d["actual_default_rate"].iloc[0] > d["actual_default_rate"].iloc[-1]
    assert d["population"].sum() == 2000


def test_score_scaling_points_to_double_odds() -> None:
    cfg = ModelingConfig()
    # 50:1 good:bad odds -> PD = 1/51 -> base score 600; doubling odds adds PDO points.
    assert pd_to_score(np.array([1 / 51]), cfg)[0] == 600
    assert pd_to_score(np.array([1 / 101]), cfg)[0] == 620
    assert score_to_pd(np.array([600]), cfg)[0] == pytest.approx(1 / 51)
    pds = np.linspace(0.01, 0.9, 50)
    assert np.all(np.diff(pd_to_score(pds, cfg)) <= 0)  # higher PD -> lower score


def test_risk_bands_are_ordered_and_cover_all_pds() -> None:
    cfg = ModelingConfig()
    bands = assign_risk_band(np.array([0.0, 0.039, 0.04, 0.10, 0.2, 0.99, 1.0]), cfg)
    assert bands.tolist() == [
        "A - Very Low",
        "A - Very Low",
        "B - Low",
        "C - Moderate",
        "D - Elevated",
        "E - High",
        "E - High",
    ]


def test_psi_zero_for_identical_and_large_for_shifted() -> None:
    rng = np.random.default_rng(3)
    ref = rng.normal(size=5000)
    assert psi_numeric(ref, ref) == pytest.approx(0.0, abs=1e-9)
    assert psi_numeric(ref, rng.normal(size=5000)) < 0.02
    assert psi_numeric(ref, rng.normal(loc=1.0, size=5000)) > 0.25
    a = pd.Series(["x"] * 50 + ["y"] * 50)
    assert psi_categorical(a, a) == pytest.approx(0.0, abs=1e-9)
    assert psi_categorical(a, pd.Series(["x"] * 90 + ["y"] * 10)) > 0.25
