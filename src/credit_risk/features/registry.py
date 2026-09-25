"""Feature registry: the single source of truth for model inputs and the leakage audit.

Every column produced by ``sql/features/01_loan_features.sql`` that may enter a model is
declared here with its business meaning, point-in-time source and leakage review. The
training code selects features *only* through this registry, so an undeclared column
(e.g. something accidentally joined from a snapshot mart) cannot reach a model.

``REJECTED_CANDIDATES`` records features that were considered and deliberately excluded,
with the reason. It is rendered into ``reports/leakage_audit.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Kind(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


class Group(StrEnum):
    APPLICATION = "application"
    BORROWER_PROFILE = "borrower_profile"
    LOAN_HISTORY = "loan_history"
    REPAYMENT = "repayment_behaviour"
    DELINQUENCY = "delinquency"
    COLLECTIONS = "collections"
    TRANSACTIONS = "transaction_behaviour"
    RECENCY = "recency"
    STABILITY = "stability"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: Group
    kind: Kind
    description: str
    point_in_time_source: str
    skewed: bool = False  # log1p-transformed for linear models
    leakage_note: str = "Computed from events dated <= decision_date - 1."


F = FeatureSpec
N, C = Kind.NUMERIC, Kind.CATEGORICAL
PRIOR = "Prior loans (disbursed <= as_of_date) with repayments via ASOF join <= as_of_date"
TXN = "clean.transactions dated in (as_of_date - 90, as_of_date]"

FEATURES: tuple[FeatureSpec, ...] = (
    # application
    F("product_type", Group.APPLICATION, C, "Loan product applied for", "Application"),
    F("loan_amount", Group.APPLICATION, N, "Requested principal", "Application", skewed=True),
    F("term_days", Group.APPLICATION, N, "Contractual term in days", "Application"),
    F(
        "interest_rate",
        Group.APPLICATION,
        N,
        "Flat monthly rate offered at origination",
        "Pricing at origination (uses only prior history)",
    ),
    F(
        "loan_to_declared_income",
        Group.APPLICATION,
        N,
        "Principal / declared monthly income",
        "Application + onboarding form",
        skewed=True,
    ),
    F(
        "loan_to_income_proxy",
        Group.APPLICATION,
        N,
        "Principal / average monthly wallet inflow (90d)",
        TXN,
        skewed=True,
    ),
    F(
        "amount_to_largest_prior",
        Group.APPLICATION,
        N,
        "Principal / largest prior loan (limit step-up)",
        PRIOR,
        skewed=True,
    ),
    # borrower profile
    F(
        "employment_status",
        Group.BORROWER_PROFILE,
        C,
        "Self-reported employment type",
        "Onboarding form",
    ),
    F(
        "declared_monthly_income",
        Group.BORROWER_PROFILE,
        N,
        "Declared income (implausible values nulled)",
        "Onboarding form",
        skewed=True,
    ),
    F(
        "account_age_days",
        Group.BORROWER_PROFILE,
        N,
        "Wallet account age at as_of_date",
        "account_open_date",
        skewed=True,
    ),
    F(
        "bureau_previous_loans",
        Group.BORROWER_PROFILE,
        N,
        "External loans before onboarding",
        "Bureau snapshot at onboarding",
    ),
    F(
        "bureau_previous_defaults",
        Group.BORROWER_PROFILE,
        N,
        "External defaults before onboarding",
        "Bureau snapshot at onboarding",
    ),
    F(
        "pre_onboarding_txn_activity",
        Group.BORROWER_PROFILE,
        N,
        "Avg monthly wallet transactions before onboarding",
        "Onboarding",
    ),
    # loan history
    F("number_of_loans", Group.LOAN_HISTORY, N, "Prior loans with this lender", PRIOR),
    F("total_disbursed", Group.LOAN_HISTORY, N, "Sum of prior principal", PRIOR, skewed=True),
    F("average_loan_amount", Group.LOAN_HISTORY, N, "Mean prior principal", PRIOR, skewed=True),
    F("largest_loan", Group.LOAN_HISTORY, N, "Largest prior principal", PRIOR, skewed=True),
    F("active_loans", Group.LOAN_HISTORY, N, "Prior loans still unsettled at as_of_date", PRIOR),
    F("completed_loans", Group.LOAN_HISTORY, N, "Prior loans settled by as_of_date", PRIOR),
    F(
        "default_count",
        Group.LOAN_HISTORY,
        N,
        "Prior loans that reached 90+ DPD by as_of_date",
        PRIOR,
    ),
    F(
        "outstanding_principal",
        Group.LOAN_HISTORY,
        N,
        "Unpaid prior principal at as_of_date (waterfall)",
        PRIOR,
        skewed=True,
    ),
    # repayment behaviour
    F("repayment_rate", Group.REPAYMENT, N, "Paid / due on matured prior loans", PRIOR),
    F("average_days_to_repay", Group.REPAYMENT, N, "Mean disbursement-to-payoff days", PRIOR),
    F("median_days_to_repay", Group.REPAYMENT, N, "Median disbursement-to-payoff days", PRIOR),
    F(
        "late_payment_ratio",
        Group.REPAYMENT,
        N,
        "Share of settled prior loans paid after due",
        PRIOR,
    ),
    F(
        "repayment_consistency",
        Group.REPAYMENT,
        N,
        "Std. dev. of payoff timing vs due date (lower = more stable)",
        PRIOR,
    ),
    F(
        "partial_payment_ratio",
        Group.REPAYMENT,
        N,
        "Share of prior payments that did not settle the loan",
        PRIOR,
    ),
    # delinquency
    F(
        "max_dpd",
        Group.DELINQUENCY,
        N,
        "Worst DPD reached on prior loans as of as_of_date",
        PRIOR,
        skewed=True,
    ),
    F("avg_dpd", Group.DELINQUENCY, N, "Mean worst DPD on matured prior loans", PRIOR, skewed=True),
    F(
        "current_dpd",
        Group.DELINQUENCY,
        N,
        "DPD on open prior loans at as_of_date",
        PRIOR,
        skewed=True,
    ),
    F("dpd_30_count", Group.DELINQUENCY, N, "Prior loans that reached 30+ DPD", PRIOR),
    F("dpd_60_count", Group.DELINQUENCY, N, "Prior loans that reached 60+ DPD", PRIOR),
    F(
        "days_since_last_delinquency",
        Group.DELINQUENCY,
        N,
        "Days since the borrower was last past due (NULL = never)",
        PRIOR,
    ),
    # collections
    F(
        "collection_rate",
        Group.COLLECTIONS,
        N,
        "Paid by due date / due, matured prior loans",
        PRIOR,
    ),
    F(
        "collection_gap",
        Group.COLLECTIONS,
        N,
        "Arrears on matured prior loans at as_of_date",
        PRIOR,
        skewed=True,
    ),
    F(
        "collections_last_30d",
        Group.COLLECTIONS,
        N,
        "Repayments made in last 30 days",
        "clean.repayments <= as_of_date",
        skewed=True,
    ),
    F(
        "collections_last_90d",
        Group.COLLECTIONS,
        N,
        "Repayments made in last 90 days",
        "clean.repayments <= as_of_date",
        skewed=True,
    ),
    # transactions
    F("transaction_count_90d", Group.TRANSACTIONS, N, "Wallet transactions, last 90 days", TXN),
    F(
        "transaction_volume_90d",
        Group.TRANSACTIONS,
        N,
        "Absolute wallet value, last 90 days",
        TXN,
        skewed=True,
    ),
    F(
        "average_transaction_amount_90d",
        Group.TRANSACTIONS,
        N,
        "Mean absolute amount",
        TXN,
        skewed=True,
    ),
    F(
        "transaction_volatility_90d",
        Group.TRANSACTIONS,
        N,
        "Coefficient of variation of transaction amounts",
        TXN,
    ),
    F(
        "income_proxy_monthly",
        Group.TRANSACTIONS,
        N,
        "Average monthly inflow, last 90 days",
        TXN,
        skewed=True,
    ),
    F("cashflow_ratio_90d", Group.TRANSACTIONS, N, "Inflow / outflow, last 90 days", TXN),
    F(
        "average_balance_90d",
        Group.TRANSACTIONS,
        N,
        "Mean post-transaction balance",
        TXN,
        skewed=True,
    ),
    # recency
    F("days_since_last_loan", Group.RECENCY, N, "Days since previous disbursement", PRIOR),
    F(
        "days_since_last_repayment",
        Group.RECENCY,
        N,
        "Days since last repayment",
        "clean.repayments <= as_of_date",
    ),
    F("days_since_last_transaction", Group.RECENCY, N, "Days since last wallet transaction", TXN),
    # stability
    F(
        "income_volatility_180d",
        Group.STABILITY,
        N,
        "CV of 30-day inflow buckets over 180 days",
        "clean.transactions (180d)",
    ),
    F(
        "income_stability_180d",
        Group.STABILITY,
        N,
        "max(0, 1 - income_volatility_180d)",
        "clean.transactions (180d)",
    ),
    F(
        "inflow_trend_30d",
        Group.STABILITY,
        N,
        "Last-30d inflow / mean of the two preceding 30d buckets (income shock signal)",
        "clean.transactions (90d)",
        skewed=True,
    ),
    F("borrowing_frequency_90d", Group.STABILITY, N, "Loans taken in the last 90 days", PRIOR),
)

IDENTIFIERS = ("loan_id", "borrower_id", "decision_date", "as_of_date")
TARGET = "target"


@dataclass(frozen=True)
class RejectedCandidate:
    name: str
    available_at_decision: bool
    leakage_risk: str
    decision: str


REJECTED_CANDIDATES: tuple[RejectedCandidate, ...] = (
    RejectedCandidate(
        "loan_state / source_loan_status (scored loan)",
        False,
        "High",
        "Excluded - final lifecycle state is the outcome itself.",
    ),
    RejectedCandidate(
        "total_paid / payoff_date / days_to_repay (scored loan)",
        False,
        "High",
        "Excluded - repayment of the scored loan happens after the decision.",
    ),
    RejectedCandidate(
        "max_dpd_to_date / dpd_bucket (scored loan)",
        False,
        "High",
        "Excluded - defines the target.",
    ),
    RejectedCandidate(
        "Post-default recoveries", False, "High", "Excluded - only observable after default."
    ),
    RejectedCandidate(
        "Prior-loan outcomes read from marts at the snapshot date",
        False,
        "High",
        "Replaced by as-of recomputation: prior-loan DPD/paid amounts use only "
        "repayments <= as_of_date (ASOF join).",
    ),
    RejectedCandidate(
        "marts.borrower_mart KPIs / risk_segment",
        False,
        "High",
        "Excluded - measured at the snapshot date, i.e. after most decisions.",
    ),
    RejectedCandidate(
        "Wallet activity after disbursement (e.g. 'transactions next 30d')",
        False,
        "High",
        "Excluded - future information.",
    ),
    RejectedCandidate(
        "Days after maturity computed against the snapshot date",
        False,
        "High",
        "Excluded - DPD features are computed against as_of_date instead.",
    ),
    RejectedCandidate(
        "interest_rate",
        True,
        "Low",
        "Included - set at origination; risk-based uplift uses prior history "
        "only. Monitored because pricing policy changes would shift it.",
    ),
    RejectedCandidate(
        "gender",
        True,
        "n/a (protected)",
        "Excluded - protected characteristic; used only for fairness review.",
    ),
    RejectedCandidate(
        "age",
        True,
        "n/a (protected)",
        "Excluded - protected in many jurisdictions; fairness review only.",
    ),
    RejectedCandidate(
        "region",
        True,
        "n/a (proxy risk)",
        "Excluded - geographic proxy risk (redlining); fairness/monitoring only.",
    ),
)

# Columns / tables the feature SQL must never reference (static leakage scan).
FORBIDDEN_FEATURE_SQL_TOKENS = (
    "loan_state",
    "max_dpd_to_date",
    "bad_30dpd",
    "source_loan_status",
    "total_paid",
    "is_default",
    "risk_segment",
    "borrower_mart",
    "loan_monthly_snapshot",
    "gender",
    "region",
)


def model_features() -> list[FeatureSpec]:
    return list(FEATURES)


def numeric_features() -> list[str]:
    return [f.name for f in FEATURES if f.kind is Kind.NUMERIC]


def categorical_features() -> list[str]:
    return [f.name for f in FEATURES if f.kind is Kind.CATEGORICAL]


def skewed_features() -> list[str]:
    return [f.name for f in FEATURES if f.skewed]
