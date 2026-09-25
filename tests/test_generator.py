"""Synthetic data generator: determinism, schema and behavioural relationships."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_risk.config import GeneratorConfig
from credit_risk.ingestion.synthetic import generate_portfolio


@pytest.fixture(scope="module")
def clean_tables() -> dict[str, pd.DataFrame]:
    tables, _ = generate_portfolio(
        GeneratorConfig(n_borrowers=1500, seed=5, inject_quality_issues=False)
    )
    return tables


def test_generation_is_deterministic() -> None:
    cfg = GeneratorConfig(n_borrowers=200, seed=9)
    a, ma = generate_portfolio(cfg)
    b, mb = generate_portfolio(cfg)
    assert ma == mb
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_schemas(clean_tables: dict[str, pd.DataFrame]) -> None:
    assert set(clean_tables["borrowers"].columns) >= {
        "borrower_id",
        "age",
        "gender",
        "employment_status",
        "monthly_income",
        "region",
        "account_open_date",
        "previous_loans",
        "previous_defaults",
        "transaction_activity",
    }
    assert set(clean_tables["loans"].columns) >= {
        "loan_id",
        "borrower_id",
        "loan_amount",
        "disbursement_date",
        "due_date",
        "term_days",
        "interest_rate",
        "fees",
        "total_due",
        "loan_status",
        "product_type",
    }
    assert set(clean_tables["repayments"].columns) >= {
        "repayment_id",
        "loan_id",
        "borrower_id",
        "repayment_date",
        "amount",
        "payment_channel",
        "transaction_reference",
    }
    assert set(clean_tables["transactions"].columns) >= {
        "transaction_id",
        "borrower_id",
        "transaction_date",
        "transaction_type",
        "amount",
        "balance",
        "channel",
    }
    assert not any(c.startswith("sim_") for c in clean_tables["borrowers"].columns)


def test_no_events_after_snapshot(clean_tables: dict[str, pd.DataFrame]) -> None:
    snap = GeneratorConfig().snapshot_date
    assert (clean_tables["repayments"]["repayment_date"] <= snap).all()
    assert (clean_tables["transactions"]["transaction_date"] <= snap).all()
    assert (clean_tables["loans"]["disbursement_date"] < snap).all()


def test_no_overpayment_and_non_negative_balances(clean_tables: dict[str, pd.DataFrame]) -> None:
    paid = clean_tables["repayments"].groupby("loan_id")["amount"].sum()
    due = clean_tables["loans"].set_index("loan_id")["total_due"]
    assert (paid <= due.reindex(paid.index) + 0.01).all()
    assert (clean_tables["transactions"]["balance"] >= 0).all()


def _loan_outcomes(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    loans = tables["loans"].merge(tables["borrowers"], on="borrower_id")
    return loans.assign(bad=loans["loan_status"].isin(["overdue", "written_off"]))


def test_bureau_defaults_raise_risk(clean_tables: dict[str, pd.DataFrame]) -> None:
    loans = _loan_outcomes(clean_tables)
    rate = loans.groupby(loans["previous_defaults"] > 0)["bad"].mean()
    assert rate[True] > rate[False]


def test_employment_type_orders_risk(clean_tables: dict[str, pd.DataFrame]) -> None:
    loans = _loan_outcomes(clean_tables)
    rate = loans.groupby("employment_status")["bad"].mean()
    assert rate["salaried"] < rate["unemployed"]


def test_repeat_borrowing_and_partial_payments_exist(
    clean_tables: dict[str, pd.DataFrame],
) -> None:
    per_borrower = clean_tables["loans"].groupby("borrower_id").size()
    assert (per_borrower > 1).mean() > 0.3
    payments = clean_tables["repayments"].groupby("loan_id").size()
    assert (payments > 1).mean() > 0.1
    assert set(clean_tables["loans"]["loan_status"]) == {
        "active",
        "overdue",
        "repaid",
        "written_off",
    }
