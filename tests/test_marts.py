"""SQL marts on the generated portfolio: aggregation correctness and SQL/Python parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import PipelineRun

from credit_risk.analytics import metrics as lm


@pytest.fixture(scope="module")
def loans(pipeline: PipelineRun) -> pd.DataFrame:
    return pipeline.con.execute("SELECT * FROM marts.loan_repayment_summary").df()


def test_one_row_per_loan_and_balances_reconcile(
    pipeline: PipelineRun, loans: pd.DataFrame
) -> None:
    n_clean = pipeline.con.execute("SELECT count(*) FROM clean.loans").fetchone()[0]
    assert len(loans) == n_clean == loans["loan_id"].nunique()
    unsettled = loans[~loans["is_settled"]]
    assert np.allclose(
        unsettled["outstanding_total"], unsettled["total_due"] - unsettled["total_paid"]
    )
    assert (loans["outstanding_principal"] <= loans["loan_amount"] + 1e-9).all()


def test_dpd_matches_python_definition(pipeline: PipelineRun, loans: pd.DataFrame) -> None:
    as_of = pipeline.settings.generator.snapshot_date
    expected = lm.days_past_due(loans["due_date"], loans["payoff_date"], as_of)
    assert (expected.to_numpy() == loans["dpd"].to_numpy()).all()
    assert (lm.dpd_bucket(loans["dpd"], pipeline.settings.risk) == loans["dpd_bucket"]).all()


def test_monthly_par_sql_equals_python(pipeline: PipelineRun) -> None:
    con, risk = pipeline.con, pipeline.settings.risk
    snap = con.execute("SELECT * FROM marts.loan_monthly_snapshot").df()
    pm = con.execute("SELECT * FROM marts.portfolio_monthly").df().set_index("month_end")
    for month, g in snap.groupby("month_end"):
        for t in risk.par_thresholds:
            py = lm.portfolio_at_risk(g, t)
            assert py == pytest.approx(pm.loc[month, f"par{t}"], nan_ok=True)


def test_snapshot_par_at_last_month_equals_loan_summary(
    pipeline: PipelineRun, loans: pd.DataFrame
) -> None:
    last = pipeline.con.execute(
        "SELECT par30, par90 FROM marts.portfolio_monthly ORDER BY month_end DESC LIMIT 1"
    ).fetchone()
    assert lm.portfolio_at_risk(loans, 30) == pytest.approx(last[0])
    assert lm.portfolio_at_risk(loans, 90) == pytest.approx(last[1])


def test_borrower_mart_aggregations(pipeline: PipelineRun, loans: pd.DataFrame) -> None:
    bm = pipeline.con.execute("SELECT * FROM marts.borrower_mart").df().set_index("borrower_id")
    g = loans.groupby("borrower_id")
    pd.testing.assert_series_equal(
        g.size().sort_index(), bm["total_loans"].sort_index(), check_names=False, check_dtype=False
    )
    assert np.allclose(g["total_paid"].sum().sort_index(), bm["total_repaid"].sort_index())
    assert np.allclose(g["loan_amount"].sum().sort_index(), bm["total_disbursed"].sort_index())
    assert (g["is_default"].sum().sort_index() == bm["n_defaults"].sort_index()).all()


def test_segment_rules_first_match_wins(pipeline: PipelineRun) -> None:
    bm = pipeline.con.execute("SELECT * FROM marts.borrower_mart").df()
    assert (bm.loc[bm["n_defaults"] > 0, "risk_segment"] == "Default / Recovery").all()
    no_default = bm[bm["n_defaults"] == 0]
    assert (
        no_default.loc[no_default["current_dpd"] >= 30, "risk_segment"]
        == "High Risk / Persistent Delinquency"
    ).all()
    strong = bm[bm["risk_segment"] == "Low Risk / Strong Repayer"]
    assert (strong["completed_loans"] >= 2).all() and (strong["max_dpd_ever"] < 7).all()


def test_roll_rates_conserve_loans(pipeline: PipelineRun) -> None:
    rr = pipeline.con.execute("SELECT * FROM marts.roll_rates").df()
    snap = (
        pipeline.con.execute(
            "SELECT month_end, count(*) AS n FROM marts.loan_monthly_snapshot "
            "WHERE loan_state IN ('active','overdue') GROUP BY 1"
        )
        .df()
        .set_index("month_end")["n"]
    )
    by_month = rr.groupby("from_month")["n_loans"].sum()
    for month, n in by_month.items():
        assert n == snap.loc[month]


def test_kpis_are_bounded(pipeline: PipelineRun, loans: pd.DataFrame) -> None:
    k = lm.portfolio_kpis(
        loans, pipeline.settings.risk, pd.Timestamp(pipeline.settings.generator.snapshot_date)
    )
    for key in ("repayment_rate", "default_rate", "par1", "par30", "par90"):
        assert 0 <= k[key] <= 1
    assert k["par1"] >= k["par7"] >= k["par30"] >= k["par60"] >= k["par90"]


def test_all_analytics_queries_return_rows(pipeline: PipelineRun) -> None:
    for path in sorted((pipeline.settings.paths.sql / "analytics").glob("*.sql")):
        df = pipeline.con.execute(path.read_text()).df()
        assert len(df) > 0, path.name
