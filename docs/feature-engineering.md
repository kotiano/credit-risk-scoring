# Feature Engineering

Source: `sql/features/01_loan_features.sql`. Registry (single source of truth for model
inputs): `src/credit_risk/features/registry.py`. Audit: `reports/leakage_audit.md`.

## Scoring event and point-in-time contract

One row per **loan application**. For an application decided on date `T`:

```text
as_of_date = T - 1   (end of the previous day)

prior loans   : disbursement_date <= as_of_date          (same-day loans are NOT prior)
repayments    : repayment_date   <= as_of_date           (ASOF join on cumulative paid)
transactions  : transaction_date <= as_of_date, windows of 30/90/180 days
prior-loan DPD: recomputed at as_of_date, never the prior loan's final outcome
```

The hardest part is prior-loan history. A prior loan's *final* state (from
`marts.loan_repayment_summary`) is measured at the snapshot date, i.e. usually after `T`.
Features therefore recompute each prior loan's position **as of `T - 1`**:

```sql
-- cumulative repayment of each prior loan at the cut-off
FROM prior_loans AS pl
ASOF LEFT JOIN marts.loan_payment_events AS e
    ON pl.prior_loan_id = e.loan_id AND pl.as_of_date >= e.repayment_date
```

From `paid_asof` the SQL derives `settled_asof`, `dpd_asof`,
`max_dpd_asof = max(0, min(payoff, as_of) - due)`, principal outstanding, etc. The only
values read from the summary mart are *event dates that are gated*: `payoff_date` is used
only if the loan was settled by the cut-off, and `paid_by_due_date` only for loans whose
due date is before the cut-off.

## Feature catalogue (51 model inputs)

| Group | Features |
|---|---|
| Application | `product_type`, `loan_amount`, `term_days`, `interest_rate`, `loan_to_declared_income`, `loan_to_income_proxy`, `amount_to_largest_prior` |
| Borrower profile | `employment_status`, `declared_monthly_income`, `account_age_days`, `bureau_previous_loans`, `bureau_previous_defaults`, `pre_onboarding_txn_activity` |
| Loan history | `number_of_loans`, `total_disbursed`, `average_loan_amount`, `largest_loan`, `active_loans`, `completed_loans`, `default_count`, `outstanding_principal` |
| Repayment behaviour | `repayment_rate`, `average_days_to_repay`, `median_days_to_repay`, `late_payment_ratio`, `repayment_consistency`, `partial_payment_ratio` |
| Delinquency | `max_dpd`, `avg_dpd`, `current_dpd`, `dpd_30_count`, `dpd_60_count`, `days_since_last_delinquency` |
| Collections | `collection_rate`, `collection_gap`, `collections_last_30d`, `collections_last_90d` |
| Transactions (90d) | `transaction_count_90d`, `transaction_volume_90d`, `average_transaction_amount_90d`, `transaction_volatility_90d`, `income_proxy_monthly`, `cashflow_ratio_90d`, `average_balance_90d` |
| Recency | `days_since_last_loan`, `days_since_last_repayment`, `days_since_last_transaction` |
| Stability | `income_volatility_180d`, `income_stability_180d`, `inflow_trend_30d`, `borrowing_frequency_90d` |

Selected definitions:

| Feature | Definition |
|---|---|
| repayment_rate | Σ paid (to cut-off) / Σ due over prior loans already due |
| collection_rate | Σ paid by due date / Σ due over prior loans already due |
| collection_gap | Arrears on matured prior loans at the cut-off |
| late_payment_ratio | Share of settled prior loans paid after the due date |
| repayment_consistency | Std. dev. of payoff timing vs due date (days); lower = more stable |
| partial_payment_ratio | Share of prior payments that did not settle the loan |
| days_since_last_delinquency | Days since the borrower was last past due; NULL = never (imputed with an indicator) |
| default_count | Prior loans that reached 90 DPD by the cut-off. `dpd_90_count` would be identical, so it is not duplicated |
| income_proxy_monthly | Wallet inflows over 90 days / 3 |
| income_volatility_180d | Coefficient of variation of six 30-day inflow buckets (only buckets after account opening; zero-filled) |
| inflow_trend_30d | Last-30-day inflow / mean of the two previous 30-day buckets - an early income-shock signal |
| loan_to_income_proxy | Principal / observed monthly inflow |

## Missing values

"No history" is information, not noise. Counts default to 0; ratios/averages stay NULL
for first-time borrowers and are imputed with the training median **plus a missing
indicator** in the linear and random-forest pipelines (`SimpleImputer(add_indicator=True)`);
HistGradientBoosting uses native NaN handling.

## Transformations inside the model pipeline (fit on train only)

- Winsorisation at train 1st/99th percentiles (linear model).
- `log1p` on skewed monetary/count features flagged `skewed=True` in the registry.
- Standardisation (so LR coefficients are per standard deviation).
- One-hot encoding for `product_type`, `employment_status` (unknown categories ignored).

## Excluded attributes

`gender`, `age` and `region` are not in the feature table at all. See
`docs/modeling.md#responsible-modeling`.

## Leakage safeguards (automated)

1. Static scan: the feature SQL (comments stripped) may not reference outcome columns
   (`loan_state`, `max_dpd_to_date`, `bad_30dpd`, `source_loan_status`, `total_paid`,
   `is_default`), snapshot marts (`borrower_mart`, `loan_monthly_snapshot`) or
   sensitive attributes.
2. Truncation test: features recomputed from data truncated at `T` must equal production
   features for loans decided on `T` - 0 mismatches required.
3. Negative control: `tests/test_features.py` injects a leaky feature (prior loans'
   final DPD) and asserts the truncation test **fails**, proving the test has power.
4. Univariate AUC screen: any single feature with AUC > 0.85 is flagged for review.
