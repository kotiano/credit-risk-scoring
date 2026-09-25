"""SQL marts on hand-built loans with known answers (DPD, PAR, waterfall, timing)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import duckdb
import pandas as pd
import pytest

from credit_risk.analytics import metrics as lm
from credit_risk.config import get_settings
from credit_risk.db import prepare_session, run_sql_file

SETTINGS = get_settings()
MARTS = SETTINGS.paths.sql / "marts"

# loan_id, principal, interest, fee, disbursed, due
LOANS = [
    ("A", 1000.0, 0.0, 0.0, "2025-11-01", "2025-11-26"),  # unpaid, 35 DPD
    ("B", 2000.0, 0.0, 0.0, "2025-12-16", "2026-01-15"),  # not yet due
    ("C", 800.0, 50.0, 10.0, "2025-10-01", "2025-10-31"),  # paid exactly on due date
    ("D", 500.0, 0.0, 0.0, "2025-05-01", "2025-06-14"),  # unpaid 200 DPD -> written off
    ("E", 1000.0, 100.0, 20.0, "2025-11-10", "2025-12-10"),  # partial 500, 21 DPD
    ("F", 300.0, 0.0, 0.0, "2025-09-01", "2025-09-30"),  # paid 1 day late
    ("G", 400.0, 0.0, 0.0, "2025-09-01", "2025-09-30"),  # 0.50 residue within tolerance
    ("H", 600.0, 0.0, 0.0, "2025-08-01", "2025-08-31"),  # early: 10 days before due
]
REPAYMENTS = [
    ("r1", "C", "2025-10-31", 860.0),
    ("r2", "E", "2025-12-01", 500.0),
    ("r3", "F", "2025-10-01", 300.0),
    ("r4", "G", "2025-09-29", 399.5),
    ("r5", "H", "2025-08-21", 600.0),
]


@pytest.fixture(scope="module")
def con() -> Iterator[duckdb.DuckDBPyConnection]:
    c = duckdb.connect()
    prepare_session(c, SETTINGS)
    loans = pd.DataFrame(
        LOANS,
        columns=[
            "loan_id",
            "loan_amount",
            "interest_amount",
            "fee_amount",
            "disbursement_date",
            "due_date",
        ],
    )
    loans["total_due"] = loans["loan_amount"] + loans["interest_amount"] + loans["fee_amount"]
    loans["borrower_id"] = "B1"
    c.register("loans_df", loans)
    c.execute("""
        CREATE TABLE clean.loans AS
        SELECT loan_id, borrower_id, 'micro_loan' AS product_type, loan_amount,
               0.1 AS interest_rate, interest_amount, fee_amount, total_due,
               date_diff('day', disbursement_date::DATE, due_date::DATE) AS term_days,
               disbursement_date::DATE AS disbursement_date, due_date::DATE AS due_date,
               'n/a' AS source_loan_status,
               row_number() OVER (ORDER BY disbursement_date) AS borrower_loan_sequence
        FROM loans_df""")
    reps = pd.DataFrame(REPAYMENTS, columns=["repayment_id", "loan_id", "repayment_date", "amount"])
    c.register("reps_df", reps)
    c.execute("""
        CREATE TABLE clean.repayments AS
        SELECT repayment_id, loan_id, 'B1' AS borrower_id, repayment_date::DATE AS repayment_date,
               amount, 'ussd' AS payment_channel, repayment_id AS transaction_reference
        FROM reps_df""")
    for name in (
        "01_loan_payment_events",
        "02_loan_repayment_summary",
        "03_loan_default_recovery",
        "04_loan_monthly_snapshot",
        "05_portfolio_monthly",
    ):
        run_sql_file(c, MARTS / f"{name}.sql")
    yield c
    c.close()


def summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("SELECT * FROM marts.loan_repayment_summary").df().set_index("loan_id")


def test_dpd_and_state_per_loan(con: duckdb.DuckDBPyConnection) -> None:
    s = summary(con)
    assert s.loc["A", "dpd"] == 35 and s.loc["A", "loan_state"] == "overdue"
    assert s.loc["B", "dpd"] == 0 and s.loc["B", "loan_state"] == "active"
    assert s.loc["C", "dpd"] == 0 and s.loc["C", "loan_state"] == "repaid"
    assert s.loc["D", "dpd"] == 200 and s.loc["D", "loan_state"] == "written_off"
    assert s.loc["E", "dpd"] == 21 and s.loc["E", "dpd_bucket"] == "7-29"


def test_repayment_timing_and_max_dpd(con: duckdb.DuckDBPyConnection) -> None:
    s = summary(con)
    assert s.loc["C", "repayment_timing"] == "on_time" and s.loc["C", "max_dpd_to_date"] == 0
    assert s.loc["F", "repayment_timing"] == "late" and s.loc["F", "max_dpd_to_date"] == 1
    assert s.loc["H", "repayment_timing"] == "early"
    assert s.loc["G", "is_settled"]  # 0.50 residue <= 1.00 settlement tolerance


def test_payment_waterfall_fees_then_interest_then_principal(
    con: duckdb.DuckDBPyConnection,
) -> None:
    s = summary(con)
    # E: paid 500 -> fee 20, interest 100, principal 380 -> 620 principal outstanding.
    assert s.loc["E", "principal_paid"] == pytest.approx(380)
    assert s.loc["E", "outstanding_principal"] == pytest.approx(620)
    assert s.loc["E", "outstanding_total"] == pytest.approx(620)


def test_target_observability(con: duckdb.DuckDBPyConnection) -> None:
    s = summary(con)
    assert s.loc["A", "bad_30dpd"] == 1  # due + 30 <= snapshot, unpaid
    assert pd.isna(s.loc["E", "bad_30dpd"])  # performance window still open
    assert s.loc["F", "bad_30dpd"] == 0


def test_portfolio_par_matches_hand_calculation(con: duckdb.DuckDBPyConnection) -> None:
    pm = (
        con.execute("SELECT * FROM marts.portfolio_monthly WHERE month_end = DATE '2025-12-31'")
        .df()
        .iloc[0]
    )
    on_book = 1000 + 2000 + 620  # A + B + E (D written off, others repaid)
    assert pm["outstanding_principal"] == pytest.approx(on_book)
    assert pm["par1"] == pytest.approx((1000 + 620) / on_book)
    assert pm["par7"] == pytest.approx((1000 + 620) / on_book)
    assert pm["par30"] == pytest.approx(1000 / on_book)
    assert pm["par60"] == 0 and pm["par90"] == 0


def test_sql_and_python_par_agree(con: duckdb.DuckDBPyConnection) -> None:
    s = summary(con)
    pm = (
        con.execute("SELECT * FROM marts.portfolio_monthly WHERE month_end = DATE '2025-12-31'")
        .df()
        .iloc[0]
    )
    for t in SETTINGS.risk.par_thresholds:
        assert lm.portfolio_at_risk(s, t) == pytest.approx(pm[f"par{t}"])


def test_monthly_snapshot_is_point_in_time(con: duckdb.DuckDBPyConnection) -> None:
    snap = con.execute(
        "SELECT * FROM marts.loan_monthly_snapshot WHERE loan_id = 'E' ORDER BY month_end"
    ).df()
    nov, dec = snap.iloc[0], snap.iloc[1]
    assert nov["month_end"].date() == date(2025, 11, 30)
    assert nov["dpd"] == 0 and nov["outstanding_principal"] == pytest.approx(1000)
    assert dec["dpd"] == 21 and dec["collected_in_month"] == pytest.approx(500)


def test_default_recovery_exposure(con: duckdb.DuckDBPyConnection) -> None:
    rec = con.execute("SELECT * FROM marts.loan_default_recovery").df().set_index("loan_id")
    assert list(rec.index) == ["D"]
    assert rec.loc["D", "exposure_at_default"] == pytest.approx(500)
    assert rec.loc["D", "recovered_amount"] == pytest.approx(0)
