"""Data access for the dashboard: read-only warehouse queries + pipeline artifacts.

Every number displayed is loaded from the pipeline outputs (DuckDB marts, model
scores, evaluation/monitoring/data-quality JSON). Nothing is hard-coded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import duckdb
import pandas as pd
import streamlit as st

from credit_risk.config import Settings, get_settings

SETTINGS: Settings = get_settings()

LOANS_SQL = """
SELECT
    s.*,
    bm.region,
    bm.employment_status,
    bm.risk_segment,
    sc.pd,
    sc.score,
    sc.risk_band,
    sc.sample_split
FROM marts.loan_repayment_summary AS s
JOIN marts.borrower_mart AS bm USING (borrower_id)
LEFT JOIN model.loan_scores AS sc USING (loan_id)
"""


def warehouse_ready() -> tuple[bool, str]:
    path = SETTINGS.paths.warehouse
    if not path.exists():
        return False, f"Warehouse not found at `{path}`. Run `make all` first."
    needed = [SETTINGS.paths.evaluation / f for f in ("evaluation.json", "monitoring.json")]
    needed.append(SETTINGS.paths.artifacts / "data_quality.json")
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        return False, f"Missing pipeline artifacts: {', '.join(missing)}. Run `make all`."
    return True, ""


def _query(sql: str) -> pd.DataFrame:
    with duckdb.connect(str(SETTINGS.paths.warehouse), read_only=True) as con:
        return con.execute(sql).df()


@st.cache_data(show_spinner=False)
def load_loans() -> pd.DataFrame:
    df = _query(LOANS_SQL)
    for col in ("disbursement_date", "due_date", "payoff_date", "disbursement_month"):
        df[col] = pd.to_datetime(df[col])
    return df


@st.cache_data(show_spinner=False)
def load_monthly_snapshot() -> pd.DataFrame:
    df = _query(
        "SELECT loan_id, month_end, outstanding_principal, dpd, dpd_bucket, loan_state, "
        "collected_in_month FROM marts.loan_monthly_snapshot"
    )
    df["month_end"] = pd.to_datetime(df["month_end"])
    return df


@st.cache_data(show_spinner=False)
def load_repayments() -> pd.DataFrame:
    df = _query(
        "SELECT loan_id, repayment_month, amount, payment_channel, dpd_bucket_at_payment, "
        "is_post_default_recovery FROM marts.repayment_facts"
    )
    df["repayment_month"] = pd.to_datetime(df["repayment_month"])
    return df


@st.cache_data(show_spinner=False)
def load_clusters() -> pd.DataFrame:
    return _query("SELECT * FROM marts.borrower_clusters")


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict[str, Any]:
    folder = SETTINGS.paths.artifacts if name == "data_quality" else SETTINGS.paths.evaluation
    return json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Filters:
    start: date
    end: date
    products: tuple[str, ...]
    regions: tuple[str, ...]
    segments: tuple[str, ...]
    bands: tuple[str, ...]
    states: tuple[str, ...]


def apply_filters(loans: pd.DataFrame, f: Filters) -> pd.DataFrame:
    mask = (
        loans["disbursement_date"].dt.date.between(f.start, f.end)
        & loans["product_type"].isin(f.products)
        & loans["region"].isin(f.regions)
        & loans["risk_segment"].isin(f.segments)
        & loans["risk_band"].isin(f.bands)
        & loans["loan_state"].isin(f.states)
    )
    return loans[mask]
