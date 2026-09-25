"""DuckDB connection management and SQL-file execution.

SQL lives in ``sql/`` as plain, reviewable files. Business parameters are injected as
DuckDB session variables (``getvariable('snapshot_date')``) and shared macros
(``dpd_bucket(dpd)``) generated from :mod:`credit_risk.config`, so the SQL layer and the
Python layer read one definition of every metric.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from credit_risk.config import Settings
from credit_risk.logging_utils import get_logger, log_event

logger = get_logger(__name__)

SCHEMAS = ("raw", "staging", "clean", "marts", "features", "model", "monitoring", "quality")

SqlValue = date | int | float | str


def connect(db_path: Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    if not read_only:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path), read_only=read_only)


def sql_variables(settings: Settings) -> dict[str, SqlValue]:
    """Parameters every SQL file may reference through ``getvariable``."""
    risk, gen, mod = settings.risk, settings.generator, settings.modeling
    return {
        "snapshot_date": gen.snapshot_date,
        "portfolio_start": gen.portfolio_start,
        "settlement_tolerance": risk.settlement_tolerance,
        "default_dpd": risk.default_dpd,
        "write_off_dpd": risk.write_off_dpd,
        "target_dpd": risk.target_dpd,
        "early_repayment_min_days": risk.early_repayment_min_days,
        "train_end": mod.train_end,
        "validation_end": mod.validation_end,
        "test_end": mod.test_end,
    }


def _sql_literal(value: SqlValue) -> str:
    if isinstance(value, date):
        return f"DATE '{value.isoformat()}'"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float):
        return repr(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def dpd_bucket_macro_sql(settings: Settings) -> str:
    """Build the ``dpd_bucket`` macro from the configured bucket table."""
    buckets = settings.risk.dpd_buckets
    whens = []
    # Evaluate from the highest bucket downwards.
    for lower, label in reversed(buckets[1:]):
        whens.append(f"WHEN dpd >= {lower} THEN '{label}'")
    whens_sql = " ".join(whens)
    return (
        f"CREATE OR REPLACE MACRO dpd_bucket(dpd) AS CASE {whens_sql} ELSE '{buckets[0][1]}' END;"
    )


def prepare_session(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    """Create schemas, register variables and shared macros on a connection."""
    con.execute(f"SET threads = {int(settings.duckdb_threads)}")
    for schema in SCHEMAS:
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
    set_variables(con, sql_variables(settings))
    con.execute(dpd_bucket_macro_sql(settings))
    labels = ", ".join(f"'{label}'" for label in settings.risk.bucket_labels)
    # Sort key so buckets order correctly in GROUP BY outputs.
    con.execute(
        f"CREATE OR REPLACE MACRO dpd_bucket_order(label) AS list_position([{labels}], label);"
    )


def set_variables(con: duckdb.DuckDBPyConnection, variables: Mapping[str, SqlValue]) -> None:
    for name, value in variables.items():
        con.execute(f"SET VARIABLE {name} = {_sql_literal(value)}")


def run_sql_file(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    sql = path.read_text(encoding="utf-8")
    try:
        con.execute(sql)
    except duckdb.Error as exc:
        raise RuntimeError(f"SQL file failed: {path}: {exc}") from exc
    log_event(logger, "executed sql", file=f"{path.parent.name}/{path.name}")


def run_sql_layer(con: duckdb.DuckDBPyConnection, layer_dir: Path) -> list[Path]:
    """Execute every ``NN_*.sql`` file in a layer directory in lexical order."""
    files = sorted(layer_dir.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"No SQL files in {layer_dir}")
    for path in files:
        run_sql_file(con, path)
    return files


def query_df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    return con.execute(sql).df()
