"""Declarative data-quality checks.

Two gates are applied:

* **Source checks** run on the typed staging layer (i.e. the raw extract after casting,
  before anything is removed). They *measure* extract quality. A failure rate within
  tolerance is reported as ``WARN`` because the defect is quarantined or flagged by the
  clean layer; above tolerance the extract is considered unfit and the check ``FAIL``s.
* **Clean checks** run on the ``clean`` schema that feeds every mart and model. They are
  zero-tolerance invariants: any failing row is a ``FAIL`` and stops the pipeline.

Each check is a SQL expression that returns ``(failing_records, total_records)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Layer(StrEnum):
    SOURCE = "source"
    CLEAN = "clean"


class Category(StrEnum):
    UNIQUENESS = "uniqueness"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    VALIDITY = "validity"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    OUTLIER = "outlier"


class Status(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Check:
    name: str
    dataset: str
    layer: Layer
    category: Category
    description: str
    sql: str  # must return one row: (failing_records, total_records)
    tolerance: float = 0.0  # maximum acceptable failure rate
    treatment: str = ""

    def evaluate(self, failing: int, total: int) -> Status:
        if failing == 0:
            return Status.PASS
        rate = failing / total if total else 1.0
        if self.layer is Layer.SOURCE and rate <= self.tolerance:
            return Status.WARN
        return Status.FAIL


def _count(table: str, condition: str) -> str:
    return f"SELECT count(*) FILTER (WHERE {condition}), count(*) FROM {table}"


def _duplicates(raw_table: str, staged_table: str) -> str:
    return (
        f"SELECT (SELECT count(*) FROM {raw_table}) - (SELECT count(*) FROM {staged_table}), "
        f"(SELECT count(*) FROM {raw_table})"
    )


SOURCE_CHECKS: tuple[Check, ...] = (
    # ---------------------------------------------------------------- borrowers
    Check(
        "duplicate_borrower_rows",
        "borrowers",
        Layer.SOURCE,
        Category.UNIQUENESS,
        "Rows repeating an existing borrower_id",
        _duplicates("raw.borrowers", "staging.stg_borrowers"),
        0.01,
        "Collapsed to one row per borrower_id in staging",
    ),
    Check(
        "missing_declared_income",
        "borrowers",
        Layer.SOURCE,
        Category.COMPLETENESS,
        "Declared monthly income missing",
        _count("staging.stg_borrowers", "monthly_income IS NULL"),
        0.05,
        "Kept as NULL; models impute with a missing-indicator",
    ),
    Check(
        "negative_income",
        "borrowers",
        Layer.SOURCE,
        Category.VALIDITY,
        "Declared income < 0",
        _count("staging.stg_borrowers", "monthly_income < 0"),
        0.0,
    ),
    Check(
        "invalid_age",
        "borrowers",
        Layer.SOURCE,
        Category.VALIDITY,
        "Age missing or outside 18-100",
        _count("staging.stg_borrowers", "age IS NULL OR age NOT BETWEEN 18 AND 100"),
        0.0,
    ),
    Check(
        "income_outliers",
        "borrowers",
        Layer.SOURCE,
        Category.OUTLIER,
        "Declared income with robust z-score > 5 within employment group",
        _count("staging.stg_borrowers", "is_income_outlier"),
        0.01,
        "Raw value retained; monthly_income_clean set to NULL",
    ),
    # ---------------------------------------------------------------- loans
    Check(
        "duplicate_loan_rows",
        "loans",
        Layer.SOURCE,
        Category.UNIQUENESS,
        "Rows repeating an existing loan_id",
        _duplicates("raw.loans", "staging.stg_loans"),
        0.01,
        "Collapsed to one row per loan_id",
    ),
    Check(
        "loan_amount_not_positive",
        "loans",
        Layer.SOURCE,
        Category.VALIDITY,
        "loan_amount <= 0 or not numeric",
        _count("staging.stg_loans", "NOT is_amount_valid"),
        0.001,
        "Loan quarantined in clean.rejected_loans",
    ),
    Check(
        "due_before_disbursement",
        "loans",
        Layer.SOURCE,
        Category.VALIDITY,
        "due_date earlier than disbursement_date",
        _count("staging.stg_loans", "NOT is_date_order_valid"),
        0.001,
        "Loan quarantined in clean.rejected_loans",
    ),
    Check(
        "interest_rate_out_of_range",
        "loans",
        Layer.SOURCE,
        Category.VALIDITY,
        "Monthly interest rate outside [0, 0.30]",
        _count("staging.stg_loans", "NOT is_rate_valid"),
        0.001,
        "Loan quarantined in clean.rejected_loans",
    ),
    Check(
        "total_due_not_reconciled",
        "loans",
        Layer.SOURCE,
        Category.CONSISTENCY,
        "total_due != principal + flat interest + fees",
        _count("staging.stg_loans", "NOT is_amount_reconciled"),
        0.001,
        "Loan quarantined in clean.rejected_loans",
    ),
    Check(
        "loan_unknown_borrower",
        "loans",
        Layer.SOURCE,
        Category.REFERENTIAL_INTEGRITY,
        "loan.borrower_id not found in borrowers",
        "SELECT count(*) FILTER (WHERE b.borrower_id IS NULL), count(*) "
        "FROM staging.stg_loans l LEFT JOIN staging.stg_borrowers b USING (borrower_id)",
        0.0,
    ),
    # ---------------------------------------------------------------- repayments
    Check(
        "duplicate_repayments",
        "repayments",
        Layer.SOURCE,
        Category.UNIQUENESS,
        "Exact duplicates and re-posted payments sharing a transaction_reference",
        _duplicates("raw.repayments", "staging.stg_repayments"),
        0.01,
        "Collapsed to one row per transaction_reference",
    ),
    Check(
        "repayment_orphan_loan",
        "repayments",
        Layer.SOURCE,
        Category.REFERENTIAL_INTEGRITY,
        "repayment.loan_id not found in loans",
        _count("staging.stg_repayments", "NOT has_parent_loan"),
        0.005,
        "Quarantined in clean.rejected_repayments",
    ),
    Check(
        "repayment_amount_not_positive",
        "repayments",
        Layer.SOURCE,
        Category.VALIDITY,
        "Repayment amount <= 0",
        _count("staging.stg_repayments", "NOT is_amount_valid"),
        0.001,
        "Quarantined in clean.rejected_repayments",
    ),
    Check(
        "repayment_before_disbursement",
        "repayments",
        Layer.SOURCE,
        Category.VALIDITY,
        "repayment_date earlier than the loan's disbursement_date",
        _count("staging.stg_repayments", "has_parent_loan AND NOT is_date_valid"),
        0.001,
        "Quarantined in clean.rejected_repayments",
    ),
    Check(
        "repayment_borrower_mismatch",
        "repayments",
        Layer.SOURCE,
        Category.CONSISTENCY,
        "repayment.borrower_id differs from the loan's borrower_id",
        _count("staging.stg_repayments", "has_parent_loan AND NOT is_borrower_consistent"),
        0.001,
        "Quarantined in clean.rejected_repayments",
    ),
    # ---------------------------------------------------------------- transactions
    Check(
        "duplicate_transactions",
        "transactions",
        Layer.SOURCE,
        Category.UNIQUENESS,
        "Rows repeating an existing transaction_id",
        _duplicates("raw.transactions", "staging.stg_transactions"),
        0.01,
        "Collapsed to one row per transaction_id",
    ),
    Check(
        "transaction_amount_missing",
        "transactions",
        Layer.SOURCE,
        Category.COMPLETENESS,
        "Transaction amount missing",
        _count("staging.stg_transactions", "NOT is_amount_present"),
        0.001,
        "Quarantined in clean.rejected_transactions",
    ),
    Check(
        "transaction_future_dated",
        "transactions",
        Layer.SOURCE,
        Category.VALIDITY,
        "Transaction dated after the extract snapshot",
        _count("staging.stg_transactions", "NOT is_date_valid"),
        0.001,
        "Quarantined in clean.rejected_transactions",
    ),
    Check(
        "transaction_unknown_borrower",
        "transactions",
        Layer.SOURCE,
        Category.REFERENTIAL_INTEGRITY,
        "transaction.borrower_id not found in borrowers",
        _count("staging.stg_transactions", "NOT has_parent_borrower"),
        0.0,
    ),
    Check(
        "transaction_amount_outliers",
        "transactions",
        Layer.SOURCE,
        Category.OUTLIER,
        "Amount above the borrower's log-scale Tukey fence (Q3 + 2*IQR)",
        _count("staging.stg_transactions", "is_amount_outlier"),
        0.005,
        "Retained and flagged; features use windowed aggregates",
    ),
)


CLEAN_CHECKS: tuple[Check, ...] = (
    Check(
        "pk_borrowers",
        "borrowers",
        Layer.CLEAN,
        Category.UNIQUENESS,
        "borrower_id unique",
        "SELECT count(*) - count(DISTINCT borrower_id), count(*) FROM clean.borrowers",
    ),
    Check(
        "pk_loans",
        "loans",
        Layer.CLEAN,
        Category.UNIQUENESS,
        "loan_id unique",
        "SELECT count(*) - count(DISTINCT loan_id), count(*) FROM clean.loans",
    ),
    Check(
        "pk_repayments",
        "repayments",
        Layer.CLEAN,
        Category.UNIQUENESS,
        "repayment_id and transaction_reference unique",
        "SELECT greatest(count(*) - count(DISTINCT repayment_id), "
        "count(*) - count(DISTINCT transaction_reference)), count(*) FROM clean.repayments",
    ),
    Check(
        "pk_transactions",
        "transactions",
        Layer.CLEAN,
        Category.UNIQUENESS,
        "transaction_id unique",
        "SELECT count(*) - count(DISTINCT transaction_id), count(*) FROM clean.transactions",
    ),
    Check(
        "ri_loans_borrowers",
        "loans",
        Layer.CLEAN,
        Category.REFERENTIAL_INTEGRITY,
        "Every loan.borrower_id exists in borrowers",
        "SELECT count(*) FILTER (WHERE b.borrower_id IS NULL), count(*) "
        "FROM clean.loans l LEFT JOIN clean.borrowers b USING (borrower_id)",
    ),
    Check(
        "ri_repayments_loans",
        "repayments",
        Layer.CLEAN,
        Category.REFERENTIAL_INTEGRITY,
        "Every repayment.loan_id exists in loans",
        "SELECT count(*) FILTER (WHERE l.loan_id IS NULL), count(*) "
        "FROM clean.repayments r LEFT JOIN clean.loans l USING (loan_id)",
    ),
    Check(
        "ri_transactions_borrowers",
        "transactions",
        Layer.CLEAN,
        Category.REFERENTIAL_INTEGRITY,
        "Every transaction.borrower_id exists in borrowers",
        "SELECT count(*) FILTER (WHERE b.borrower_id IS NULL), count(*) "
        "FROM clean.transactions t LEFT JOIN clean.borrowers b USING (borrower_id)",
    ),
    Check(
        "loan_amount_positive",
        "loans",
        Layer.CLEAN,
        Category.VALIDITY,
        "loan_amount > 0",
        _count("clean.loans", "loan_amount <= 0 OR loan_amount IS NULL"),
    ),
    Check(
        "interest_rate_range",
        "loans",
        Layer.CLEAN,
        Category.VALIDITY,
        "interest_rate within [0, 0.30]",
        _count("clean.loans", "NOT coalesce(interest_rate BETWEEN 0 AND 0.30, FALSE)"),
    ),
    Check(
        "due_after_disbursement",
        "loans",
        Layer.CLEAN,
        Category.VALIDITY,
        "due_date >= disbursement_date",
        _count("clean.loans", "NOT coalesce(due_date >= disbursement_date, FALSE)"),
    ),
    Check(
        "repayment_amount_positive",
        "repayments",
        Layer.CLEAN,
        Category.VALIDITY,
        "amount > 0",
        _count("clean.repayments", "NOT coalesce(amount > 0, FALSE)"),
    ),
    Check(
        "repayment_after_disbursement",
        "repayments",
        Layer.CLEAN,
        Category.VALIDITY,
        "repayment_date >= disbursement_date and <= snapshot",
        "SELECT count(*) FILTER (WHERE r.repayment_date < l.disbursement_date "
        "OR r.repayment_date > getvariable('snapshot_date')), count(*) "
        "FROM clean.repayments r JOIN clean.loans l USING (loan_id)",
    ),
    Check(
        "no_overpayment",
        "loans",
        Layer.CLEAN,
        Category.CONSISTENCY,
        "Cumulative repayments do not exceed total_due (+ tolerance)",
        "SELECT count(*) FILTER (WHERE coalesce(p.paid, 0) > l.total_due "
        "+ getvariable('settlement_tolerance')), count(*) FROM clean.loans l "
        "LEFT JOIN (SELECT loan_id, sum(amount) AS paid FROM clean.repayments GROUP BY 1) p "
        "USING (loan_id)",
    ),
    Check(
        "income_non_negative",
        "borrowers",
        Layer.CLEAN,
        Category.VALIDITY,
        "monthly_income_clean >= 0",
        _count("clean.borrowers", "monthly_income_clean < 0"),
    ),
    Check(
        "transaction_amount_present",
        "transactions",
        Layer.CLEAN,
        Category.COMPLETENESS,
        "amount not null",
        _count("clean.transactions", "amount IS NULL"),
    ),
)
