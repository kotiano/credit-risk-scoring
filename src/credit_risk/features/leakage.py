"""Automated leakage audit.

Three independent checks:

1. **Static SQL scan** - the feature SQL must not reference outcome columns or
   snapshot-dated marts (comments are stripped before scanning).
2. **Truncation test** - for several decision dates ``T``, rebuild the inputs using only
   data that existed at the end of ``T - 1`` (loans disbursed <= T, repayments and
   transactions < T), recompute features, and require them to equal the production
   features for loans decided on ``T``. Any difference proves future information leaked.
3. **Too-good-to-be-true screen** - a single feature with univariate ROC-AUC above a
   threshold on the training sample is flagged for manual review.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_risk.config import Settings
from credit_risk.db import prepare_session, run_sql_file, set_variables
from credit_risk.features.registry import (
    FORBIDDEN_FEATURE_SQL_TOKENS,
    IDENTIFIERS,
    numeric_features,
)
from credit_risk.logging_utils import get_logger, log_event

logger = get_logger(__name__)

SUSPICIOUS_UNIVARIATE_AUC = 0.85


@dataclass(frozen=True)
class TruncationResult:
    cutoff: date
    loans_compared: int
    columns_compared: int
    mismatched_cells: int
    mismatched_columns: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.mismatched_cells == 0 and self.loans_compared > 0


def strip_sql_comments(sql: str) -> str:
    return re.sub(r"--[^\n]*", "", sql)


def static_sql_scan(feature_sql_path: Path) -> list[str]:
    """Return forbidden tokens referenced by the (comment-stripped) feature SQL."""
    sql = strip_sql_comments(feature_sql_path.read_text(encoding="utf-8")).lower()
    return [t for t in FORBIDDEN_FEATURE_SQL_TOKENS if re.search(rf"\b{t}\b", sql)]


def _build_truncated_features(
    source: duckdb.DuckDBPyConnection, settings: Settings, cutoff: date
) -> pd.DataFrame:
    """Recompute features in an isolated in-memory DB holding only pre-cutoff data.

    Tables are copied through Parquet so DuckDB types (DATE, DOUBLE) are preserved.
    """
    cut = f"DATE '{cutoff.isoformat()}'"
    queries = {
        "borrowers": "SELECT * FROM clean.borrowers",
        "loans": f"SELECT * FROM clean.loans WHERE disbursement_date <= {cut}",
        "repayments": f"SELECT * FROM clean.repayments WHERE repayment_date < {cut}",
        "transactions": f"SELECT * FROM clean.transactions WHERE transaction_date < {cut}",
    }
    mem = duckdb.connect()
    try:
        prepare_session(mem, settings)
        set_variables(mem, {"snapshot_date": cutoff - timedelta(days=1)})
        with tempfile.TemporaryDirectory() as tmp:
            for name, query in queries.items():
                path = Path(tmp) / f"{name}.parquet"
                source.execute(f"COPY ({query}) TO '{path}' (FORMAT parquet)")
                mem.execute(f"CREATE TABLE clean.{name} AS SELECT * FROM read_parquet('{path}')")
        # The source status is a snapshot field; blank it so any use would surface.
        mem.execute("UPDATE clean.loans SET source_loan_status = NULL")
        marts = settings.paths.sql / "marts"
        run_sql_file(mem, marts / "01_loan_payment_events.sql")
        run_sql_file(mem, marts / "02_loan_repayment_summary.sql")
        run_sql_file(mem, settings.paths.sql / "features" / "01_loan_features.sql")
        return mem.execute(f"SELECT * FROM features.loan_features WHERE decision_date = {cut}").df()
    finally:
        mem.close()


def _frames_differ(a: pd.Series, b: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        x, y = a.astype(float).to_numpy(), b.astype(float).to_numpy()
        both_nan = np.isnan(x) & np.isnan(y)
        return ~(both_nan | np.isclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=False))
    return (a.astype(str) != b.astype(str)).to_numpy()


def truncation_test(
    con: duckdb.DuckDBPyConnection, settings: Settings, cutoffs: list[date]
) -> list[TruncationResult]:
    results = []
    for cutoff in cutoffs:
        production = con.execute(
            "SELECT * FROM features.loan_features WHERE decision_date = ? ORDER BY loan_id",
            [cutoff],
        ).df()
        truncated = (
            _build_truncated_features(con, settings, cutoff)
            .sort_values("loan_id")
            .reset_index(drop=True)
        )
        columns = [c for c in production.columns if c not in IDENTIFIERS]
        if (
            len(production) != len(truncated)
            or not (production["loan_id"].to_numpy() == truncated["loan_id"].to_numpy()).all()
        ):
            results.append(TruncationResult(cutoff, len(production), len(columns), -1, ("rows",)))
            continue
        mismatched: dict[str, int] = {}
        for col in columns:
            diff = int(_frames_differ(production[col], truncated[col]).sum())
            if diff:
                mismatched[col] = diff
        results.append(
            TruncationResult(
                cutoff, len(production), len(columns), sum(mismatched.values()), tuple(mismatched)
            )
        )
        log_event(
            logger,
            "truncation test",
            cutoff=cutoff,
            loans=len(production),
            mismatches=sum(mismatched.values()),
        )
    return results


def default_cutoffs(con: duckdb.DuckDBPyConnection, n: int = 4) -> list[date]:
    """Pick decision dates spread over the sample with the most repeat-borrower loans."""
    df = con.execute(
        "SELECT decision_date, count(*) FILTER (WHERE number_of_loans > 0) AS repeat_loans "
        "FROM features.loan_features GROUP BY 1 ORDER BY 1"
    ).df()
    df["bucket"] = pd.qcut(df["decision_date"].rank(method="first"), n, labels=False)
    picks = df.sort_values("repeat_loans", ascending=False).groupby("bucket").head(1)
    return sorted(pd.to_datetime(picks["decision_date"]).dt.date.tolist())


def univariate_auc_screen(train: pd.DataFrame, target: str = "target") -> pd.DataFrame:
    rows = []
    y = train[target].astype(int)
    for col in numeric_features():
        x = train[col].astype(float)
        mask = x.notna()
        if mask.sum() < 50 or y[mask].nunique() < 2 or x[mask].nunique() < 2:
            continue
        auc = roc_auc_score(y[mask], x[mask])
        rows.append(
            {
                "feature": col,
                "coverage": float(mask.mean()),
                "univariate_auc": float(max(auc, 1 - auc)),
            }
        )
    out = pd.DataFrame(rows).sort_values("univariate_auc", ascending=False)
    out["flag"] = np.where(out["univariate_auc"] > SUSPICIOUS_UNIVARIATE_AUC, "REVIEW", "ok")
    return out.reset_index(drop=True)
