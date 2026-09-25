"""Build the leakage-audit report (reports/leakage_audit.md)."""

from __future__ import annotations

import duckdb
import pandas as pd

from credit_risk.config import Settings
from credit_risk.features.leakage import (
    SUSPICIOUS_UNIVARIATE_AUC,
    default_cutoffs,
    static_sql_scan,
    truncation_test,
    univariate_auc_screen,
)
from credit_risk.features.registry import FEATURES, REJECTED_CANDIDATES
from credit_risk.logging_utils import get_logger, log_event
from credit_risk.reporting.markdown import GENERATED_BANNER, df_to_markdown

logger = get_logger(__name__)


class LeakageError(RuntimeError):
    """Raised when an automated leakage check fails."""


def run_leakage_audit(con: duckdb.DuckDBPyConnection, settings: Settings) -> dict[str, object]:
    feature_sql = settings.paths.sql / "features" / "01_loan_features.sql"
    forbidden = static_sql_scan(feature_sql)
    truncation = truncation_test(con, settings, default_cutoffs(con))
    train = con.execute("SELECT * FROM features.model_dataset WHERE sample_split = 'train'").df()
    screen = univariate_auc_screen(train)

    registry = pd.DataFrame(
        [
            {
                "feature": f.name,
                "group": f.group.value,
                "available_at_prediction_time": "Yes",
                "source (point-in-time)": f.point_in_time_source,
                "potential_leakage": "No - " + f.leakage_note,
                "decision": "Included",
            }
            for f in FEATURES
        ]
    )
    rejected = pd.DataFrame(
        [
            {
                "candidate": r.name,
                "available_at_prediction_time": "Yes" if r.available_at_decision else "No",
                "leakage_risk": r.leakage_risk,
                "decision": r.decision,
            }
            for r in REJECTED_CANDIDATES
        ]
    )
    trunc_df = pd.DataFrame(
        [
            {
                "cutoff": t.cutoff.isoformat(),
                "loans_compared": t.loans_compared,
                "columns_compared": t.columns_compared,
                "mismatched_cells": t.mismatched_cells,
                "result": "PASS" if t.passed else "FAIL",
            }
            for t in truncation
        ]
    )
    passed = not forbidden and all(t.passed for t in truncation)
    flagged = screen[screen["flag"] == "REVIEW"]

    lines = [
        "# Leakage Audit",
        "",
        GENERATED_BANNER,
        "",
        f"**Automated result: {'PASS' if passed else 'FAIL'}**",
        "",
        "## 1. Static scan of the feature SQL",
        "",
        "`sql/features/01_loan_features.sql` (comments stripped) must not reference outcome",
        "columns or snapshot-dated marts.",
        "",
        f"Forbidden references found: **{', '.join(forbidden) if forbidden else 'none'}**.",
        "",
        "## 2. Truncation test (point-in-time proof)",
        "",
        "For each cut-off date T, all inputs are rebuilt from data that existed at the end of",
        "T - 1 (loans disbursed <= T, repayments and transactions < T), features are",
        "recomputed in an isolated database, and compared cell-by-cell with production",
        "features for loans decided on T. Any difference would prove future information",
        "leaked into a feature.",
        "",
        df_to_markdown(trunc_df),
        "",
        "## 3. Too-good-to-be-true screen (train window)",
        "",
        f"Single features with univariate ROC-AUC > {SUSPICIOUS_UNIVARIATE_AUC} are flagged",
        f"for manual review. Flagged: **{len(flagged)}**.",
        "",
        df_to_markdown(screen.head(15)),
        "",
        "## 4. Feature-by-feature review",
        "",
        df_to_markdown(registry),
        "",
        "## 5. Candidates considered and rejected (or accepted with conditions)",
        "",
        df_to_markdown(rejected),
        "",
        "## 6. Target and sampling safeguards",
        "",
        "- Target `bad_30dpd` is only defined when `due_date + 30 <= snapshot`; immature",
        "  loans are labelled `unobserved` and never used for training or evaluation.",
        "- Splits are out-of-time by decision date; the test window is scored once after",
        "  model selection and threshold choice are frozen on validation.",
        "- A borrower can appear in several windows (behavioural scoring), but each",
        "  application only sees its own past.",
        "",
    ]
    (settings.paths.reports / "leakage_audit.md").write_text("\n".join(lines), encoding="utf-8")
    log_event(
        logger,
        "leakage audit",
        passed=passed,
        forbidden=len(forbidden),
        cutoffs=len(truncation),
        flagged_features=len(flagged),
    )
    if not passed:
        raise LeakageError("Leakage audit failed - see reports/leakage_audit.md")
    return {"passed": passed, "forbidden": forbidden, "truncation": trunc_df, "screen": screen}
