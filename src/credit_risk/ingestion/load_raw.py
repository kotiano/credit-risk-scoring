"""Land raw CSV extracts in the ``raw`` schema exactly as received.

Every column is loaded as VARCHAR: the raw layer is an immutable copy of the source
extract, and all typing/standardisation happens in ``sql/staging`` where casting
failures become visible data-quality findings instead of load errors.
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from credit_risk.logging_utils import get_logger, log_event

logger = get_logger(__name__)

RAW_TABLES = ("borrowers", "loans", "repayments", "transactions")


def load_raw_extracts(con: duckdb.DuckDBPyConnection, raw_dir: Path) -> dict[str, int]:
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    counts: dict[str, int] = {}
    for table in RAW_TABLES:
        path = raw_dir / f"{table}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw extract {path}; run `make generate-data`")
        con.execute(
            f"CREATE OR REPLACE TABLE raw.{table} AS "
            "SELECT *, ? AS _source_file, now() AS _loaded_at "
            "FROM read_csv(?, header = true, all_varchar = true)",
            [path.name, str(path)],
        )
        row = con.execute(f"SELECT count(*) FROM raw.{table}").fetchone()
        counts[table] = int(row[0]) if row else 0
    log_event(logger, "loaded raw extracts", **counts)
    return counts
