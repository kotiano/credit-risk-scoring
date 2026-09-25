"""Render the data-quality report (reports/data_quality_report.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from credit_risk.reporting.markdown import GENERATED_BANNER, df_to_markdown

STATUS_ICON = {"PASS": "PASS", "WARN": "WARN (handled)", "FAIL": "FAIL"}


def render_data_quality_report(payload: dict[str, Any], path: Path) -> None:
    summary = payload["summary"]
    checks = pd.DataFrame(payload["checks"])
    profile = pd.DataFrame(payload["profile"])
    missing = pd.DataFrame(payload["missingness"])
    quarantine = pd.DataFrame(payload["quarantine"])

    checks["status"] = checks["status"].map(STATUS_ICON)
    source = checks[checks["layer"] == "source"]
    clean = checks[checks["layer"] == "clean"]
    cols = [
        "dataset",
        "name",
        "category",
        "failing_records",
        "total_records",
        "failure_rate",
        "status",
        "treatment",
    ]
    missing_nonzero = missing[missing["missing"] > 0].sort_values("missing_pct", ascending=False)

    lines = [
        "# Data Quality Report",
        "",
        GENERATED_BANNER,
        "",
        "## Summary",
        "",
        f"- **Overall status:** {summary['overall_status']} "
        f"(clean layer: **{summary['clean_layer_status']}**)",
        f"- **Checks evaluated:** {summary['checks_total']} — "
        + ", ".join(f"{k}: {v}" for k, v in summary["checks_by_status"].items()),
        f"- **Record retention (clean / raw rows):** {summary['record_retention_rate']:.4%}",
        "",
        "Status semantics: `PASS` = no failing records; `WARN (handled)` = defects found in",
        "the source extract within tolerance and quarantined/flagged by the clean layer;",
        "`FAIL` = above tolerance or a clean-layer invariant broken (pipeline stops).",
        "",
        "## Dataset profile (raw extract)",
        "",
        df_to_markdown(
            profile,
            formats={
                "missing_cell_pct": ".3%",
                "duplicate_pct": ".3%",
                "raw_rows": ",d",
                "clean_rows": ",d",
            },
        ),
        "",
        "## Source-extract checks",
        "",
        df_to_markdown(
            source[cols],
            formats={"failure_rate": ".4%", "failing_records": ",d", "total_records": ",d"},
        ),
        "",
        "## Clean-layer invariants (zero tolerance)",
        "",
        df_to_markdown(
            clean[cols[:-1]],
            formats={"failure_rate": ".4%", "failing_records": ",d", "total_records": ",d"},
        ),
        "",
        "## Quarantined records",
        "",
        df_to_markdown(quarantine, formats={"records": ",d"}),
        "",
        "## Missingness (raw columns with any missing values)",
        "",
        df_to_markdown(missing_nonzero, formats={"missing_pct": ".3%", "missing": ",d"})
        if not missing_nonzero.empty
        else "_No missing values._",
        "",
        "## Treatment decisions",
        "",
        "- Outliers are **flagged, never deleted**. Implausible declared incomes (unit errors)",
        "  are set to NULL in `monthly_income_clean` while the raw value is retained.",
        "- Records that make delinquency measurement unreliable (bad dates, non-positive",
        "  amounts, orphans) are quarantined in `clean.rejected_*` with a reason code.",
        "- Repayments on quarantined loans are quarantined as well to keep balances",
        "  consistent.",
        "- See `docs/etl.md` for the full rule set.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
