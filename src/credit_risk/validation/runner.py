"""Execute data-quality checks, profile datasets and persist results."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pandas as pd

from credit_risk.logging_utils import get_logger, log_event
from credit_risk.validation.checks import CLEAN_CHECKS, SOURCE_CHECKS, Check, Layer, Status

logger = get_logger(__name__)

DATASETS = {
    # dataset: (raw table, staged table, clean table, primary key)
    "borrowers": ("raw.borrowers", "staging.stg_borrowers", "clean.borrowers", "borrower_id"),
    "loans": ("raw.loans", "staging.stg_loans", "clean.loans", "loan_id"),
    "repayments": ("raw.repayments", "staging.stg_repayments", "clean.repayments", "repayment_id"),
    "transactions": (
        "raw.transactions",
        "staging.stg_transactions",
        "clean.transactions",
        "transaction_id",
    ),
}


class DataQualityError(RuntimeError):
    """Raised when a blocking data-quality check fails."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    dataset: str
    layer: str
    category: str
    description: str
    failing_records: int
    total_records: int
    failure_rate: float
    status: str
    treatment: str


def run_checks(con: duckdb.DuckDBPyConnection, checks: Iterable[Check]) -> list[CheckResult]:
    results = []
    for check in checks:
        row = con.execute(check.sql).fetchone()
        if row is None:
            raise DataQualityError(f"Check {check.name} returned no rows")
        failing, total = int(row[0] or 0), int(row[1] or 0)
        status = check.evaluate(failing, total)
        results.append(
            CheckResult(
                check.name,
                check.dataset,
                check.layer.value,
                check.category.value,
                check.description,
                failing,
                total,
                failing / total if total else 0.0,
                status.value,
                check.treatment,
            )
        )
    return results


def missingness_profile(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Column-level missing % in the raw extracts (empty CSV fields load as NULL)."""
    frames = []
    for dataset, (raw_table, *_rest) in DATASETS.items():
        cols = [
            c
            for (c,) in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'raw' AND table_name = ? AND column_name NOT LIKE '\\_%' "
                "ESCAPE '\\' ORDER BY ordinal_position",
                [raw_table.split(".")[1]],
            ).fetchall()
        ]
        exprs = ", ".join(f'count(*) - count("{c}") AS "{c}"' for c in cols)
        counts = con.execute(f"SELECT count(*) AS n, {exprs} FROM {raw_table}").df()
        n = int(counts.pop("n").iloc[0])
        frame = counts.T.reset_index()
        frame.columns = ["column", "missing"]
        frame.insert(0, "dataset", dataset)
        frame["missing_pct"] = frame["missing"] / n if n else 0.0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def dataset_profile(
    con: duckdb.DuckDBPyConnection, missing: pd.DataFrame, results: list[CheckResult]
) -> pd.DataFrame:
    rows = []
    source = pd.DataFrame([asdict(r) for r in results if r.layer == Layer.SOURCE.value])
    for dataset, (raw_table, _staged, clean_table, _pk) in DATASETS.items():
        raw_rows = _scalar(con, f"SELECT count(*) FROM {raw_table}")
        clean_rows = _scalar(con, f"SELECT count(*) FROM {clean_table}")
        n_cols = len(missing[missing["dataset"] == dataset])
        cells_missing = missing.loc[missing["dataset"] == dataset, "missing"].sum()
        ds = source[source["dataset"] == dataset]
        by_cat = ds.groupby("category")["failing_records"].sum()
        rows.append(
            {
                "dataset": dataset,
                "raw_rows": raw_rows,
                "columns": n_cols,
                "missing_cell_pct": cells_missing / (raw_rows * n_cols) if raw_rows else 0.0,
                "duplicate_pct": by_cat.get("uniqueness", 0) / raw_rows if raw_rows else 0.0,
                "invalid_records": int(
                    by_cat.get("validity", 0)
                    + by_cat.get("consistency", 0)
                    + by_cat.get("completeness", 0)
                ),
                "outliers_flagged": int(by_cat.get("outlier", 0)),
                "ri_failures": int(by_cat.get("referential_integrity", 0)),
                "clean_rows": clean_rows,
                "status": _worst(ds["status"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    return int(row[0]) if row else 0


def _worst(statuses: list[str]) -> str:
    for status in (Status.FAIL, Status.WARN):
        if status.value in statuses:
            return status.value
    return Status.PASS.value


def quarantine_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    parts = []
    for dataset in ("loans", "repayments", "transactions"):
        df = con.execute(
            f"SELECT '{dataset}' AS dataset, rejection_reason, count(*) AS records "
            f"FROM clean.rejected_{dataset} GROUP BY ALL ORDER BY records DESC"
        ).df()
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def run_data_quality(con: duckdb.DuckDBPyConnection, out_dir: Path) -> dict[str, object]:
    """Run all checks, persist results to ``quality.*`` and JSON; raise on FAIL."""
    results = run_checks(con, SOURCE_CHECKS) + run_checks(con, CLEAN_CHECKS)
    results_df = pd.DataFrame([asdict(r) for r in results])
    missing = missingness_profile(con)
    profile = dataset_profile(con, missing, results)
    quarantine = quarantine_summary(con)

    con.register("results_df", results_df)
    con.execute("CREATE OR REPLACE TABLE quality.check_results AS SELECT * FROM results_df")
    con.register("profile_df", profile)
    con.execute("CREATE OR REPLACE TABLE quality.dataset_profile AS SELECT * FROM profile_df")
    con.register("missing_df", missing)
    con.execute("CREATE OR REPLACE TABLE quality.missingness AS SELECT * FROM missing_df")

    source_records = profile["raw_rows"].sum()
    clean_records = profile["clean_rows"].sum()
    summary: dict[str, object] = {
        "overall_status": _worst(results_df["status"].tolist()),
        "checks_total": len(results),
        "checks_by_status": results_df["status"].value_counts().to_dict(),
        "clean_layer_status": _worst(
            results_df.loc[results_df["layer"] == "clean", "status"].tolist()
        ),
        # Share of check evaluations that passed outright (PASS), across both layers.
        "check_pass_rate": float((results_df["status"] == "PASS").mean()),
        "record_retention_rate": float(clean_records / source_records) if source_records else 0,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "checks": results_df.to_dict(orient="records"),
        "profile": profile.to_dict(orient="records"),
        "missingness": missing.to_dict(orient="records"),
        "quarantine": quarantine.to_dict(orient="records"),
    }
    (out_dir / "data_quality.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    log_event(
        logger,
        "data quality evaluated",
        **{k: v for k, v in summary.items() if not isinstance(v, dict)},
    )
    failed = results_df[results_df["status"] == Status.FAIL.value]
    if not failed.empty:
        raise DataQualityError(
            "Blocking data-quality failures: " + ", ".join(failed["name"].tolist())
        )
    return payload
