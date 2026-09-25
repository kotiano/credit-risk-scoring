"""Run the analytical SQL library (sql/analytics) and publish results.

Each query is executed as-is against the warehouse; results are written to
``artifacts/analytics/<query>.csv`` and summarised in ``reports/sql_analytics.md`` next
to the query's purpose (its leading comment), so reviewers see SQL and output together.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from credit_risk.config import Settings
from credit_risk.logging_utils import get_logger, log_event
from credit_risk.reporting.markdown import GENERATED_BANNER, df_to_markdown

logger = get_logger(__name__)

MAX_REPORT_ROWS = 30


def query_purpose(sql: str) -> str:
    lines = [ln[2:].strip() for ln in sql.splitlines() if ln.startswith("--")]
    return " ".join(lines)


def run_analytics(con: duckdb.DuckDBPyConnection, settings: Settings) -> dict[str, pd.DataFrame]:
    out_dir = settings.paths.artifacts / "analytics"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    sections = []
    for path in sorted((settings.paths.sql / "analytics").glob("*.sql")):
        sql = path.read_text(encoding="utf-8")
        try:
            df = con.execute(sql).df()
        except duckdb.Error as exc:
            raise RuntimeError(f"Analytics query failed: {path.name}: {exc}") from exc
        results[path.stem] = df
        df.to_csv(out_dir / f"{path.stem}.csv", index=False)
        shown = df.head(MAX_REPORT_ROWS)
        note = (
            f"_Showing {MAX_REPORT_ROWS} of {len(df)} rows; full output in "
            f"`artifacts/analytics/{path.stem}.csv`._"
            if len(df) > MAX_REPORT_ROWS
            else ""
        )
        sections += [
            f"## `{path.name}`",
            "",
            query_purpose(sql),
            "",
            df_to_markdown(shown, float_fmt=",.4f"),
            "",
            note,
            "",
        ]
    report = [
        "# SQL Analytics",
        "",
        GENERATED_BANNER,
        "",
        "Results of every query in [`sql/analytics/`](../sql/analytics), run against the",
        "DuckDB warehouse at the snapshot date. Metric definitions: `docs/metrics.md`.",
        "",
        *sections,
    ]
    Path(settings.paths.reports / "sql_analytics.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    log_event(logger, "analytics queries complete", queries=len(results))
    return results
