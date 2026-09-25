"""Synthetic but behaviourally realistic digital-lending portfolio generator.

The generator simulates the *data-generating process* of a short-term digital lender
rather than sampling columns independently:

* every borrower has a hidden creditworthiness trait that drives bureau history,
  income, wallet activity and repayment behaviour (so features carry real signal);
* wallet transactions are simulated month by month with income volatility, temporary
  income shocks and a regional macro-stress episode from ``stress_start``;
* loans are issued sequentially: a borrower can only re-borrow after settling, limits
  grow after good repayments and shrink after late ones, and own repayment history
  feeds back into future risk;
* repayment outcomes follow an ordinal-logit model (early → on-time → late buckets →
  default) and are turned into dated partial/full repayment transactions;
* anything dated after the snapshot date is dropped, so the data is right-censored the
  way a real extract is.

Finally, a small, documented set of data-quality defects is injected into the raw
extract (duplicates, orphans, invalid dates, inconsistent labels, outliers) so the
validation layer has real work to do. The manifest of injected issues is written next
to the data and is used by tests to prove each defect is detected.

All assumptions are documented in ``docs/assumptions.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.config import GeneratorConfig
from credit_risk.logging_utils import get_logger, log_event

logger = get_logger(__name__)

# --------------------------------------------------------------------------------------
# Structural assumptions (documented in docs/assumptions.md)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class EmploymentProfile:
    share: float
    income_median: float  # declared monthly income, local currency units
    income_sigma: float
    risk_effect: float  # additive effect on the repayment-risk logit
    income_volatility: float  # typical coefficient of variation of monthly inflows


EMPLOYMENT: dict[str, EmploymentProfile] = {
    "salaried": EmploymentProfile(0.38, 48_000, 0.45, -0.35, 0.12),
    "self_employed": EmploymentProfile(0.27, 40_000, 0.60, 0.00, 0.35),
    "informal": EmploymentProfile(0.25, 17_000, 0.50, 0.30, 0.40),
    "student": EmploymentProfile(0.05, 7_000, 0.50, 0.40, 0.30),
    "unemployed": EmploymentProfile(0.05, 5_000, 0.60, 0.75, 0.45),
}


@dataclass(frozen=True)
class RegionProfile:
    share: float
    income_multiplier: float
    risk_effect: float
    stressed: bool  # affected by the late-2025 macro stress episode


REGIONS: dict[str, RegionProfile] = {
    "Metro": RegionProfile(0.30, 1.25, -0.10, False),
    "Central": RegionProfile(0.20, 1.00, -0.05, False),
    "Lakeside": RegionProfile(0.14, 0.90, 0.05, False),
    "Highlands": RegionProfile(0.14, 0.95, 0.00, False),
    "Coastal": RegionProfile(0.12, 0.85, 0.10, True),
    "Northern": RegionProfile(0.10, 0.70, 0.15, True),
}


@dataclass(frozen=True)
class ProductProfile:
    terms: tuple[int, ...]
    monthly_rate: float  # flat interest per 30 days
    fee_rate: float  # one-off origination fee, share of principal
    min_amount: float
    max_amount: float
    income_multiple: float  # base limit as a multiple of declared monthly income
    risk_effect: float


PRODUCTS: dict[str, ProductProfile] = {
    "salary_advance": ProductProfile((30,), 0.08, 0.020, 1_000, 100_000, 0.50, -0.20),
    "micro_loan": ProductProfile((14, 30), 0.10, 0.030, 500, 30_000, 0.30, 0.00),
    "emergency_loan": ProductProfile((7,), 0.12, 0.020, 200, 5_000, 0.10, 0.25),
    "sme_working_capital": ProductProfile((60, 90), 0.06, 0.025, 5_000, 250_000, 0.90, 0.10),
}

PRODUCT_MIX: dict[str, dict[str, float]] = {
    "salaried": {
        "salary_advance": 0.55,
        "micro_loan": 0.25,
        "emergency_loan": 0.15,
        "sme_working_capital": 0.05,
    },
    "self_employed": {"micro_loan": 0.35, "emergency_loan": 0.15, "sme_working_capital": 0.50},
    "informal": {"micro_loan": 0.65, "emergency_loan": 0.30, "sme_working_capital": 0.05},
    "student": {"micro_loan": 0.60, "emergency_loan": 0.40},
    "unemployed": {"micro_loan": 0.50, "emergency_loan": 0.50},
}

# Ordinal repayment outcomes, from best to worst, and their cut-points on the latent
# risk scale (P(outcome <= k) = sigmoid(cut_k - risk)).
OUTCOMES = ("early", "on_time", "late_1_6", "late_7_29", "late_30_89", "default")
OUTCOME_CUTPOINTS = np.array([-1.45, 0.20, 1.00, 2.10, 2.70])

# How a loan outcome shifts the borrower's risk on the *next* loan (behavioural memory).
HISTORY_EFFECT = {
    "early": -0.12,
    "on_time": -0.10,
    "late_1_6": 0.15,
    "late_7_29": 0.50,
    "late_30_89": 0.90,
    "default": 1.20,
}
RETURN_PROBABILITY = {
    "early": 0.85,
    "on_time": 0.85,
    "late_1_6": 0.75,
    "late_7_29": 0.55,
    "late_30_89": 0.30,
    "default": 0.10,
}

INFLOW_TYPES = ("deposit", "p2p_received")
OUTFLOW_TYPES = ("withdrawal", "merchant_payment", "bill_payment", "p2p_sent", "airtime")
TXN_CHANNELS = ("app", "ussd", "agent", "card", "bank")
REPAYMENT_CHANNELS = ("mobile_app", "ussd", "agent", "bank_transfer", "auto_debit")

TRANSACTION_HISTORY_MONTHS = 7  # wallet history simulated before the portfolio start
SHOCK_MONTHLY_PROBABILITY = 0.010
STRESS_SHOCK_MULTIPLIER = 3.0  # shock probability multiplier in stressed regions
STRESS_INFLOW_MULTIPLIER = {True: 0.85, False: 0.93}
STRESS_RISK_EFFECT = 0.30


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _month_starts(start: date, end: date) -> list[date]:
    months, current = [], date(start.year, start.month, 1)
    while current <= end:
        months.append(current)
        current = date(current.year + current.month // 12, current.month % 12 + 1, 1)
    return months


# --------------------------------------------------------------------------------------
# Borrowers
# --------------------------------------------------------------------------------------


def _generate_borrowers(cfg: GeneratorConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_borrowers
    trait = rng.normal(size=n)  # hidden creditworthiness (higher = riskier)

    employment = rng.choice(list(EMPLOYMENT), size=n, p=[p.share for p in EMPLOYMENT.values()])
    region = rng.choice(list(REGIONS), size=n, p=[r.share for r in REGIONS.values()])
    emp = [EMPLOYMENT[e] for e in employment]
    reg = [REGIONS[r] for r in region]

    income_median = np.array([e.income_median for e in emp])
    income_sigma = np.array([e.income_sigma for e in emp])
    region_mult = np.array([r.income_multiplier for r in reg])
    declared_income = (
        income_median * region_mult * np.exp(income_sigma * rng.normal(size=n) - 0.10 * trait)
    )
    # Wallet-observed income is typically lower than declared income.
    observed_income = declared_income * rng.uniform(0.55, 1.05, size=n)

    age = np.where(
        employment == "student",
        rng.integers(18, 26, size=n),
        np.clip(20 + rng.gamma(3.0, 5.5, size=n), 18, 70).astype(int),
    )
    gender = rng.choice(["F", "M", "U"], size=n, p=[0.48, 0.50, 0.02])

    activity = -0.3 * trait + np.sqrt(1 - 0.09) * rng.normal(size=n)
    txn_rate = np.clip(
        np.exp(np.log(5) + 0.45 * activity + 0.2 * np.log(declared_income / 20_000)), 1, 40
    )
    base_vol = np.array([e.income_volatility for e in emp])
    income_cv = np.clip(base_vol * np.exp(0.35 * rng.normal(size=n) + 0.15 * trait), 0.05, 1.0)

    # Onboarding: a quarter are pre-existing customers, the rest join with growth.
    start = cfg.portfolio_start
    pre_existing = rng.random(n) < 0.25
    pre_days = rng.integers(0, (start - date(2022, 7, 1)).days, size=n)
    growth_window = (cfg.snapshot_date - start).days - 45
    new_days = (rng.random(n) ** (1 / 1.5) * growth_window).astype(int)
    onboarding = np.where(
        pre_existing,
        np.array([date(2022, 7, 1) + timedelta(days=int(d)) for d in pre_days]),
        np.array([start + timedelta(days=int(d)) for d in new_days]),
    )
    account_age_at_onboarding = rng.exponential(250, size=n).astype(int)
    account_open = np.array(
        [
            o - timedelta(days=int(a))
            for o, a in zip(onboarding, account_age_at_onboarding, strict=True)
        ]
    )

    is_salaried = employment == "salaried"
    is_self = employment == "self_employed"
    previous_loans = rng.poisson(1.2 + 0.8 * is_salaried + 0.5 * is_self)
    previous_defaults = rng.binomial(previous_loans, _sigmoid(-2.2 + 1.0 * trait))
    pre_onboarding_activity = rng.poisson(txn_rate * rng.uniform(0.7, 1.1, size=n))

    risk_logit = (
        np.array([e.risk_effect for e in emp])
        + np.array([r.risk_effect for r in reg])
        + 0.75 * trait
        - 0.35 * np.log(declared_income / 20_000)
        - 0.25 * activity
        + 1.2 * (income_cv - 0.3)
        + 0.35 * np.minimum(previous_defaults, 3)
        - 0.012 * (age - 35)
    )

    return pd.DataFrame(
        {
            "borrower_id": [f"B{i:06d}" for i in range(1, n + 1)],
            "onboarding_date": onboarding,
            "account_open_date": account_open,
            "age": age,
            "gender": gender,
            "employment_status": employment,
            "monthly_income": np.round(declared_income, -1),
            "region": region,
            "previous_loans": previous_loans,
            "previous_defaults": previous_defaults,
            "transaction_activity": pre_onboarding_activity,
            # hidden simulation state (dropped before export)
            "sim_risk_logit": risk_logit,
            "sim_observed_income": observed_income,
            "sim_income_cv": income_cv,
            "sim_txn_rate": txn_rate,
        }
    )


# --------------------------------------------------------------------------------------
# Income shocks and wallet transactions
# --------------------------------------------------------------------------------------


def _simulate_shocks(
    borrowers: pd.DataFrame, months: list[date], cfg: GeneratorConfig, rng: np.random.Generator
) -> np.ndarray:
    """Return an (n_borrowers, n_months) income multiplier; < 1 during a shock."""
    n, m = len(borrowers), len(months)
    stressed = borrowers["region"].map(lambda r: REGIONS[r].stressed).to_numpy()
    multiplier = np.ones((n, m))
    remaining = np.zeros(n, dtype=int)
    depth = np.ones(n)
    for j, month in enumerate(months):
        p = np.full(n, SHOCK_MONTHLY_PROBABILITY)
        if month >= cfg.stress_start:
            p = np.where(stressed, p * STRESS_SHOCK_MULTIPLIER, p * 1.5)
        starts = (remaining == 0) & (rng.random(n) < p)
        remaining[starts] = rng.integers(2, 5, size=starts.sum())
        depth[starts] = rng.uniform(0.35, 0.65, size=starts.sum())
        active = remaining > 0
        multiplier[active, j] = depth[active]
        remaining[active] -= 1
    return multiplier


def _generate_transactions(
    borrowers: pd.DataFrame,
    months: list[date],
    shocks: np.ndarray,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    stressed = borrowers["region"].map(lambda r: REGIONS[r].stressed).to_numpy()
    month_arr = np.array(months)
    open_month = borrowers["account_open_date"].map(lambda d: date(d.year, d.month, 1)).to_numpy()

    # Borrower-month grid restricted to months where the wallet account exists.
    b_idx, m_idx = np.nonzero(month_arr[None, :] >= open_month[:, None])
    macro = np.where(
        month_arr[m_idx] >= cfg.stress_start,
        np.where(stressed[b_idx], STRESS_INFLOW_MULTIPLIER[True], STRESS_INFLOW_MULTIPLIER[False]),
        1.0,
    )
    shock = shocks[b_idx, m_idx]
    cv = borrowers["sim_income_cv"].to_numpy()[b_idx]
    inflow_total = (
        borrowers["sim_observed_income"].to_numpy()[b_idx]
        * np.exp(cv * rng.normal(size=len(b_idx)) - cv**2 / 2)
        * shock
        * macro
    )
    outflow_total = inflow_total * rng.uniform(0.85, 1.03, size=len(b_idx))
    count = rng.poisson(borrowers["sim_txn_rate"].to_numpy()[b_idx] * np.sqrt(shock)) + 1
    n_in = np.clip(rng.binomial(count, 0.35), 1, None)
    n_out = np.clip(count - n_in, 1, None)
    salaried = (borrowers["employment_status"].to_numpy() == "salaried")[b_idx]

    # Last usable day of each borrower-month (the final month is truncated at snapshot).
    days_in_month = np.array(
        [((m.replace(day=28) + timedelta(days=4)).replace(day=1) - m).days for m in months]
    )
    last_day = np.minimum(
        days_in_month[m_idx],
        np.array([(cfg.snapshot_date - m).days + 1 for m in months])[m_idx],
    )

    frames = []
    for direction, n_tx, totals, types in (
        ("in", n_in, inflow_total, INFLOW_TYPES),
        ("out", n_out, outflow_total, OUTFLOW_TYPES),
    ):
        group = np.repeat(np.arange(len(b_idx)), n_tx)
        weights = rng.exponential(size=len(group))
        share = weights / np.bincount(group, weights)[group]
        group_totals = totals.copy()
        if direction == "in":
            # Salaried borrowers receive ~70% of inflows as one salary credit.
            group_totals = np.where(salaried, totals * 0.30, totals)
        amounts = group_totals[group] * share
        day = rng.integers(1, last_day[group] + 1)
        frames.append(
            pd.DataFrame(
                {
                    "_g": group,
                    "day": day,
                    "direction": direction,
                    "transaction_type": rng.choice(types, size=len(group)),
                    "amount": amounts,
                }
            )
        )
    sal_groups = np.nonzero(salaried)[0]
    frames.append(
        pd.DataFrame(
            {
                "_g": sal_groups,
                "day": np.minimum(rng.integers(25, 29, size=len(sal_groups)), last_day[sal_groups]),
                "direction": "in",
                "transaction_type": "salary",
                "amount": inflow_total[sal_groups] * 0.70,
            }
        )
    )
    tx = pd.concat(frames, ignore_index=True)
    tx["borrower_idx"] = b_idx[tx["_g"].to_numpy()]
    month_start = month_arr[m_idx[tx["_g"].to_numpy()]]
    tx["transaction_date"] = pd.to_datetime(month_start) + pd.to_timedelta(tx["day"] - 1, unit="D")
    tx["amount"] = np.maximum(tx["amount"].round(2), 10.0)
    tx["channel"] = rng.choice(TXN_CHANNELS, size=len(tx), p=[0.45, 0.25, 0.15, 0.1, 0.05])

    # Running balance: inflows are processed before outflows on the same day, and an
    # opening balance guarantees the wallet never goes negative.
    tx = tx.sort_values(["borrower_idx", "transaction_date", "direction"]).reset_index(drop=True)
    signed = np.where(tx["direction"] == "in", tx["amount"], -tx["amount"])
    cumulative = pd.Series(signed).groupby(tx["borrower_idx"].to_numpy()).cumsum().to_numpy()
    min_cum = pd.Series(cumulative).groupby(tx["borrower_idx"].to_numpy()).transform("min")
    opening = (
        np.maximum(0.0, -min_cum.to_numpy())
        + rng.uniform(50, 2_000, size=len(tx))[tx.groupby("borrower_idx").ngroup().to_numpy()]
    )
    tx["balance"] = np.round(cumulative + opening, 2)
    tx["amount"] = np.where(tx["direction"] == "in", tx["amount"], -tx["amount"])

    tx["borrower_id"] = borrowers["borrower_id"].to_numpy()[tx["borrower_idx"].to_numpy()]
    tx["transaction_id"] = [f"T{i:09d}" for i in range(1, len(tx) + 1)]
    tx["transaction_date"] = tx["transaction_date"].dt.date
    return tx[
        [
            "transaction_id",
            "borrower_id",
            "transaction_date",
            "transaction_type",
            "amount",
            "balance",
            "channel",
        ]
    ]


# --------------------------------------------------------------------------------------
# Loans and repayments
# --------------------------------------------------------------------------------------


def _split_amount(rng: np.random.Generator, total: float, n: int) -> np.ndarray:
    weights = rng.dirichlet(np.ones(n))
    parts = np.round(weights * total, 2)
    parts[-1] = round(total - parts[:-1].sum(), 2)
    return parts


def _simulate_repayments(
    rng: np.random.Generator, outcome: str, term: int, total_due: float
) -> list[tuple[int, float]]:
    """Return dated repayments as ``(days since disbursement, amount)``."""
    due = term
    if outcome == "default":
        payments: list[tuple[int, float]] = []
        n_pay = int(rng.choice([0, 1, 2, 3], p=[0.35, 0.35, 0.20, 0.10]))
        paid = 0.0
        if n_pay:
            paid_total = round(total_due * rng.uniform(0.05, 0.6), 2)
            days = np.sort(rng.integers(1, due + 76, size=n_pay))
            for d, amt in zip(days, _split_amount(rng, paid_total, n_pay), strict=True):
                payments.append((int(d), float(amt)))
            paid = paid_total
        if rng.random() < 0.30:  # late recovery (collections / restructuring)
            remaining = total_due - paid
            full = rng.random() < 0.5
            amount = round(remaining if full else remaining * rng.uniform(0.2, 0.8), 2)
            payments.append((int(due + rng.integers(95, 421)), float(amount)))
        return payments

    if outcome == "early":
        settle = int(rng.integers(1, max(2, term - 2)))
    elif outcome == "on_time":
        settle = due - int(rng.integers(0, 3))
    else:
        low, high = {"late_1_6": (1, 6), "late_7_29": (7, 29), "late_30_89": (30, 89)}[outcome]
        settle = due + int(rng.integers(low, high + 1))

    partial_probability = {
        "early": 0.15,
        "on_time": 0.25,
        "late_1_6": 0.45,
        "late_7_29": 0.55,
        "late_30_89": 0.75,
    }[outcome]
    n_partials = 0
    if settle > 2 and rng.random() < partial_probability:
        n_partials = int(rng.integers(1, 4 if outcome == "late_30_89" else 3))
    if n_partials == 0:
        return [(settle, round(total_due, 2))]
    partial_total = round(total_due * rng.uniform(0.2, 0.7), 2)
    days = np.sort(rng.integers(1, settle, size=n_partials))
    payments = [
        (int(d), float(a))
        for d, a in zip(days, _split_amount(rng, partial_total, n_partials), strict=True)
    ]
    payments.append((settle, round(total_due - partial_total, 2)))
    return payments


def _repayment_channel(rng: np.random.Generator, dpd_at_payment: int, employment: str) -> str:
    if dpd_at_payment >= 30 and rng.random() < 0.5:
        return "agent"  # field collections
    if employment == "salaried" and dpd_at_payment == 0 and rng.random() < 0.25:
        return "auto_debit"
    return str(rng.choice(REPAYMENT_CHANNELS[:4], p=[0.50, 0.28, 0.12, 0.10]))


@dataclass
class _LoanRecord:
    loan_id: str
    borrower_id: str
    product_type: str
    loan_amount: float
    interest_rate: float
    fees: float
    total_due: float
    term_days: int
    disbursement_date: date
    due_date: date
    loan_status: str


def _source_status(
    payments: list[tuple[date, float]], total_due: float, due_date: date, snapshot: date
) -> str:
    """Loan status as the (simulated) source system reports it at the snapshot."""
    paid = sum(a for d, a in payments if d <= snapshot)
    if total_due - paid <= 1.0:
        return "repaid"
    if snapshot <= due_date:
        return "active"
    return "written_off" if (snapshot - due_date).days >= 180 else "overdue"


def _generate_loans(
    borrowers: pd.DataFrame,
    months: list[date],
    shocks: np.ndarray,
    cfg: GeneratorConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    month_index = {m: j for j, m in enumerate(months)}
    snapshot = cfg.snapshot_date
    loans: list[_LoanRecord] = []
    repayments: list[dict[str, Any]] = []

    def issue_loan(
        b: Any,  # borrower row (itertuples namedtuple)
        b_idx: int,
        start: date,
        history_effect: float,
        limit_mult: float,
        last_outcome: str | None,
    ) -> tuple[str, date | None]:
        mix = PRODUCT_MIX[b.employment_status]
        product = str(rng.choice(list(mix), p=list(mix.values())))
        prof = PRODUCTS[product]
        term = int(rng.choice(prof.terms))
        income = b.monthly_income if b.monthly_income > 0 else prof.min_amount
        amount = float(
            np.clip(
                round(prof.income_multiple * income * limit_mult * rng.uniform(0.5, 1.0), -2),
                prof.min_amount,
                prof.max_amount,
            )
        )
        # Risk-based pricing uses only information known at origination.
        pricing_uplift = 0.02 if last_outcome in {"late_7_29", "late_30_89", "default"} else 0.0
        rate = round(prof.monthly_rate + pricing_uplift, 4)
        interest = round(amount * rate * term / 30, 2)
        fees = round(amount * prof.fee_rate, 2)
        total_due = round(amount + interest + fees, 2)
        due = start + timedelta(days=term)

        j = month_index[date(start.year, start.month, 1)]
        shock_now = shocks[b_idx, j] < 1
        shock_next = j + 1 < shocks.shape[1] and shocks[b_idx, j + 1] < 1
        risk = (
            b.sim_risk_logit
            + prof.risk_effect
            + history_effect
            + 0.6 * max(0.0, amount / b.sim_observed_income - 0.5)
            + 1.0 * shock_now
            + 0.5 * shock_next  # not observable at origination: irreducible risk
            + (STRESS_RISK_EFFECT if start >= cfg.stress_start else 0.0)
        )
        latent = risk + rng.logistic()
        outcome = OUTCOMES[int(np.searchsorted(OUTCOME_CUTPOINTS, latent))]

        schedule = _simulate_repayments(rng, outcome, term, total_due)
        dated = [(start + timedelta(days=d), a) for d, a in schedule]
        loan_id = f"L{len(loans) + 1:07d}"
        observed = [(d, a) for d, a in dated if d <= snapshot]
        for d, a in observed:
            dpd = max(0, (d - due).days)
            repayments.append(
                {
                    "loan_id": loan_id,
                    "borrower_id": b.borrower_id,
                    "repayment_date": d,
                    "amount": a,
                    "payment_channel": _repayment_channel(rng, dpd, b.employment_status),
                }
            )
        loans.append(
            _LoanRecord(
                loan_id,
                b.borrower_id,
                product,
                amount,
                rate,
                fees,
                total_due,
                term,
                start,
                due,
                _source_status(dated, total_due, due, snapshot),
            )
        )
        paid = sum(a for _, a in observed)
        payoff = max(d for d, _ in observed) if observed and total_due - paid <= 1.0 else None
        return outcome, payoff

    b: Any
    for b_idx, b in enumerate(borrowers.itertuples(index=False)):
        if b.onboarding_date < cfg.portfolio_start:
            # Legacy customers: first in-window loan spread over the first ~4 months.
            start = cfg.portfolio_start + timedelta(days=int(rng.integers(0, 120)))
        else:
            start = b.onboarding_date + timedelta(days=int(rng.geometric(1 / 20)))
        history_effect, limit_mult, n_good = 0.0, 0.6, 0
        last_outcome: str | None = None
        while start < snapshot:
            outcome, payoff = issue_loan(
                b,
                b_idx,
                start,
                history_effect,
                limit_mult,
                last_outcome,
            )
            if outcome in {"early", "on_time"}:
                n_good += 1
                history_effect = max(-0.6, history_effect + HISTORY_EFFECT[outcome])
                limit_mult = min(2.0, limit_mult * 1.25)
                # Occasional concurrent top-up loan for established good borrowers.
                if n_good >= 2 and payoff is not None and rng.random() < 0.10:
                    top_up_start = payoff + timedelta(days=int(rng.integers(3, 20)))
                    if top_up_start < snapshot:
                        issue_loan(
                            b, b_idx, top_up_start, history_effect, limit_mult * 0.5, last_outcome
                        )
            else:
                history_effect += HISTORY_EFFECT[outcome]
                if outcome != "late_1_6":
                    limit_mult = max(0.3, limit_mult * 0.7)
            last_outcome = outcome
            if payoff is None or rng.random() > RETURN_PROBABILITY[outcome]:
                break
            start = payoff + timedelta(days=int(rng.geometric(1 / 15)))

    loans_df = pd.DataFrame([vars(r) for r in loans])
    loans_df = loans_df.sort_values(["disbursement_date", "loan_id"]).reset_index(drop=True)
    rep_df = pd.DataFrame(repayments).sort_values(["repayment_date", "loan_id"])
    rep_df = rep_df.reset_index(drop=True)
    rep_df.insert(0, "repayment_id", [f"R{i:08d}" for i in range(1, len(rep_df) + 1)])
    refs = rng.integers(0, 16**6, size=len(rep_df))
    rep_df["transaction_reference"] = [f"MP{i:08d}{r:06X}" for i, r in enumerate(refs, 1)]
    return loans_df, rep_df


# --------------------------------------------------------------------------------------
# Data-quality defect injection
# --------------------------------------------------------------------------------------


def _inject_quality_issues(
    tables: dict[str, pd.DataFrame], rng: np.random.Generator
) -> dict[str, dict[str, int]]:
    """Inject realistic raw-extract defects in place; return a manifest of counts.

    Defect counts are proportional to table size (at least one of each), so small test
    portfolios have the same defect *rates* as the full extract.
    """
    manifest: dict[str, dict[str, int]] = {}

    def pick(df: pd.DataFrame, k: int) -> np.ndarray:
        return rng.choice(df.index.to_numpy(), size=k, replace=False)

    def n_of(df: pd.DataFrame, rate: float) -> int:
        return max(1, round(rate * len(df)))

    # ---- borrowers: missing income, income unit errors, label variants, duplicates
    b = tables["borrowers"]
    b["gender"] = b["gender"].astype(object)
    missing = pick(b, n_of(b, 0.015))
    b.loc[missing, "monthly_income"] = np.nan
    unit_errors = pick(b.drop(index=pd.Index(missing)), n_of(b, 0.0016))
    b.loc[unit_errors, "monthly_income"] = b.loc[unit_errors, "monthly_income"] * 100
    variants = pick(b, n_of(b, 0.03))
    b.loc[variants, "gender"] = b.loc[variants, "gender"].map(
        {"F": "female", "M": " m", "U": "unknown"}
    )
    dups = b.loc[pick(b, n_of(b, 0.0012))]
    tables["borrowers"] = pd.concat([b, dups], ignore_index=True)
    manifest["borrowers"] = {
        "missing_income": len(missing),
        "income_unit_errors": len(unit_errors),
        "label_variants": len(variants),
        "exact_duplicates": len(dups),
    }

    # ---- loans: swapped dates, zero principal, status label variants, duplicates
    ln = tables["loans"]
    ln["loan_status"] = ln["loan_status"].astype(object)
    swapped = pick(ln, n_of(ln, 0.0002))
    ln.loc[swapped, ["disbursement_date", "due_date"]] = ln.loc[
        swapped, ["due_date", "disbursement_date"]
    ].to_numpy()
    zero_amount = pick(ln.drop(index=pd.Index(swapped)), n_of(ln, 0.00013))
    ln.loc[zero_amount, "loan_amount"] = 0.0
    variants = pick(ln, n_of(ln, 0.03))
    ln.loc[variants, "loan_status"] = ln.loc[variants, "loan_status"].map(
        lambda s: rng.choice([s.upper(), f" {s} ", s.title()])
    )
    dups = ln.loc[pick(ln, n_of(ln, 0.00027))]
    tables["loans"] = pd.concat([ln, dups], ignore_index=True)
    rejected_loan_ids = set(ln.loc[np.concatenate([swapped, zero_amount]), "loan_id"])
    manifest["loans"] = {
        "due_before_disbursement": len(swapped),
        "non_positive_amount": len(zero_amount),
        "label_variants": len(variants),
        "exact_duplicates": len(dups),
    }

    # ---- repayments: bad amounts, pre-disbursement dates, orphans, duplicates, re-posts
    r = tables["repayments"]
    loan_disb = tables["loans"].drop_duplicates("loan_id").set_index("loan_id")["disbursement_date"]
    eligible = r[~r["loan_id"].isin(rejected_loan_ids)]
    cascaded = int(r["loan_id"].isin(rejected_loan_ids).sum())
    bad_amount = pick(eligible, n_of(r, 0.0003))
    r.loc[bad_amount, "amount"] = rng.choice([0.0, -50.0, -120.0], size=len(bad_amount))
    eligible = eligible.drop(index=pd.Index(bad_amount))
    early = pick(eligible, n_of(r, 0.00025))
    r.loc[early, "repayment_date"] = [
        loan_disb[lid] - timedelta(days=int(rng.integers(1, 10))) for lid in r.loc[early, "loan_id"]
    ]
    variants = pick(r, n_of(r, 0.05))
    r.loc[variants, "payment_channel"] = r.loc[variants, "payment_channel"].map(
        lambda c: rng.choice([c.upper(), f" {c} ", c.replace("_", " ").title()])
    )
    orphans = r.loc[pick(r, n_of(r, 0.0006))].copy()
    orphans["loan_id"] = [f"L9{i:06d}" for i in range(len(orphans))]
    orphans["repayment_id"] = [f"R9{i:07d}" for i in range(len(orphans))]
    orphans["transaction_reference"] = [f"MPORPHAN{i:06d}" for i in range(len(orphans))]
    exact = r.loc[pick(r, n_of(r, 0.004))]
    reposted = r.loc[pick(r.drop(index=exact.index), n_of(r, 0.0005))].copy()  # retry
    reposted["repayment_id"] = [f"R8{i:07d}" for i in range(len(reposted))]
    tables["repayments"] = pd.concat([r, orphans, exact, reposted], ignore_index=True)
    manifest["repayments"] = {
        "non_positive_amount": len(bad_amount),
        "before_disbursement": len(early),
        "label_variants": len(variants),
        "orphan_loan_id": len(orphans),
        "exact_duplicates": len(exact),
        "reposted_duplicates": len(reposted),
        "linked_to_rejected_loans": cascaded,
    }

    # ---- transactions: null amounts, outliers, future dates, label variants, duplicates
    t = tables["transactions"]
    t["transaction_type"] = t["transaction_type"].astype(object)
    null_amount = pick(t, n_of(t, 0.000025))
    t.loc[null_amount, "amount"] = np.nan
    outl = pick(t.drop(index=pd.Index(null_amount)), n_of(t, 0.00002))
    t.loc[outl, "amount"] = t.loc[outl, "amount"] * 50
    future = pick(t.drop(index=pd.Index(np.concatenate([null_amount, outl]))), n_of(t, 0.0000125))
    t.loc[future, "transaction_date"] = [
        date(2026, 3, 1) + timedelta(days=int(d)) for d in rng.integers(0, 60, size=len(future))
    ]
    variants = pick(t, n_of(t, 0.02))
    t.loc[variants, "transaction_type"] = t.loc[variants, "transaction_type"].str.upper()
    dups = t.loc[pick(t, n_of(t, 0.002))]
    tables["transactions"] = pd.concat([t, dups], ignore_index=True)
    manifest["transactions"] = {
        "null_amount": len(null_amount),
        "amount_outliers": len(outl),
        "future_dated": len(future),
        "label_variants": len(variants),
        "exact_duplicates": len(dups),
    }
    return manifest


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def generate_portfolio(cfg: GeneratorConfig) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Generate raw borrower, loan, repayment and transaction extracts."""
    rng = np.random.default_rng(cfg.seed)
    borrowers = _generate_borrowers(cfg, rng)
    history_start = date(cfg.portfolio_start.year, cfg.portfolio_start.month, 1)
    for _ in range(TRANSACTION_HISTORY_MONTHS):
        history_start = (history_start - timedelta(days=1)).replace(day=1)
    months = _month_starts(history_start, cfg.snapshot_date)
    shocks = _simulate_shocks(borrowers, months, cfg, rng)
    transactions = _generate_transactions(borrowers, months, shocks, cfg, rng)
    loans, repayments = _generate_loans(borrowers, months, shocks, cfg, rng)

    tables = {
        "borrowers": borrowers.drop(
            columns=[c for c in borrowers.columns if str(c).startswith("sim_")]
        ),
        "loans": loans,
        "repayments": repayments,
        "transactions": transactions,
    }
    manifest: dict[str, Any] = {
        "seed": cfg.seed,
        "n_borrowers": cfg.n_borrowers,
        "portfolio_start": cfg.portfolio_start.isoformat(),
        "snapshot_date": cfg.snapshot_date.isoformat(),
        "stress_start": cfg.stress_start.isoformat(),
        "injected_issues": {},
    }
    if cfg.inject_quality_issues:
        manifest["injected_issues"] = _inject_quality_issues(tables, rng)
    manifest["row_counts"] = {name: len(df) for name, df in tables.items()}
    return tables, manifest


def write_raw_extracts(
    tables: dict[str, pd.DataFrame], manifest: dict[str, Any], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log_event(logger, "wrote raw extracts", directory=str(out_dir), **manifest["row_counts"])
