# ETL Lineage and Transformation Logic

Every transformation is a SQL file executed in lexical order per layer by
`credit_risk.transformations.pipeline`. Parameters (`snapshot_date`, tolerances, DPD
thresholds, split dates) are DuckDB session variables set from `config.py`.

## Lineage

```text
raw.borrowers    -> staging.stg_borrowers    -> clean.borrowers ---------------------------+
raw.loans        -> staging.stg_loans        -> clean.loans / clean.rejected_loans --------+
raw.repayments   -> staging.stg_repayments   -> clean.repayments / clean.rejected_repayments
raw.transactions -> staging.stg_transactions -> clean.transactions / clean.rejected_transactions
                                                   |
clean.repayments -> marts.loan_payment_events (daily cumulative paid per loan)
clean.loans + events -> marts.loan_repayment_summary (loan grain, as of snapshot)
                     -> marts.loan_default_recovery (defaulted loans)
                     -> marts.loan_monthly_snapshot (loan x month-end)
                         -> marts.portfolio_monthly (PAR, collections, KPIs by month)
                         -> marts.roll_rates
summary + clean.borrowers -> marts.borrower_mart (borrower KPIs + risk_segment)
summary -> marts.cohort_performance, marts.vintage_curves, marts.collections_due_cohort
clean.repayments + summary + borrower_mart -> marts.repayment_facts
clean.* + events + summary (event dates only) -> features.loan_features (point-in-time)
features.loan_features + summary (target) -> features.model_dataset
```

## Layer contracts

| Layer | Contract |
|---|---|
| `raw` | Byte-for-byte copy of the extract; **every column VARCHAR**; `_source_file`, `_loaded_at` added. Never modified. |
| `staging` | Typed with `TRY_CAST` (failures become NULL and are caught by checks), labels standardised, duplicates collapsed, **defects flagged, nothing removed** except duplicate copies. |
| `clean` | Only rows passing validity rules; failures moved to `clean.rejected_*` with a `rejection_reason`. Zero-tolerance invariants are enforced by `CLEAN_CHECKS`. |
| `marts` | Business logic (DPD, PAR, waterfall, lifecycle, segments). Measured as of `snapshot_date`. |
| `features` | Point-in-time, one row per application; must not read snapshot-dated outcomes (enforced by the leakage audit). |

## Staging and clean rules

### Borrowers (`01_stg_borrowers.sql`, `01_clean_borrowers.sql`)

| Aspect | Rule |
|---|---|
| Grain | one row per `borrower_id` |
| De-duplication | `row_number() OVER (PARTITION BY borrower_id)`; exact duplicate rows collapse |
| Labels | gender from mixed codes/words (`F`, `female`, ` m`, `unknown`) -> `female` / `male` / `undisclosed` via first letter |
| Outliers | declared income robust z-score (median/MAD of log income **within employment group**) > 5 -> `is_income_outlier`. Treatment: raw value kept in `monthly_income`; `monthly_income_clean` set to NULL (likely unit error - we do not guess the intended value) |
| Missing | income missing stays NULL; models impute median + missing indicator |
| Region | NULL -> `unknown` |
| Rejection | none (no borrower-level defect is disqualifying) |

### Loans (`02_stg_loans.sql`, `02_clean_loans.sql`)

| Aspect | Rule |
|---|---|
| Grain | one row per `loan_id` |
| Derived | `interest_amount = total_due - principal - fees` |
| Status | `lower(trim())`; kept as `source_loan_status` for **reconciliation only**; lifecycle state is re-derived from repayments in marts (`sql/analytics/19_source_status_reconciliation.sql`) |
| Validity flags | amount > 0; rate in [0, 0.30]; `due_date >= disbursement_date`; `total_due` reconciles to principal + flat interest + fees within 0.02 |
| Rejection | any failed validity flag or unknown borrower -> `clean.rejected_loans` |
| Derived in clean | `borrower_loan_sequence` (1 = first loan) |

### Repayments (`03_stg_repayments.sql`, `03_clean_repayments.sql`)

| Aspect | Rule |
|---|---|
| Grain | one row per money movement (`transaction_reference`) |
| De-duplication | two patterns: exact duplicate rows, and integration re-posts (same reference, new `repayment_id`). Both collapse on `transaction_reference`, keeping the earliest id |
| Labels | channel `'Mobile App'`, `' mobile_app '`, `'MOBILE_APP'` -> `mobile_app` |
| Joins | LEFT JOIN `stg_loans` to test parent existence, dates, borrower consistency |
| Rejection | orphan loan id, parent loan quarantined (cascade), amount <= 0, dated before disbursement, borrower mismatch -> `clean.rejected_repayments`; payments after snapshot are excluded |

### Transactions (`04_stg_transactions.sql`, `04_clean_transactions.sql`)

| Aspect | Rule |
|---|---|
| Grain | one row per `transaction_id` (wallet activity; **lending flows are not in this feed**) |
| Derived | `direction` from sign; `abs_amount` |
| Outliers | per-borrower upper Tukey fence on log amount (Q3 + 2·IQR) -> `is_amount_outlier`; retained |
| Rejection | missing amount, future-dated (after snapshot), unknown borrower |

## Mart logic

### `marts.loan_payment_events`
Daily grain per loan: `paid_on_date`, running `cumulative_paid` / `cumulative_payments`.
Enables **as-of balances** via `ASOF JOIN` (latest event on or before a date).

### `marts.loan_repayment_summary` (loan grain)
Totals paid, paid by due date, paid within 30 days of due, payoff date (first date
cumulative paid >= total due - tolerance), waterfall principal, DPD and worst DPD,
lifecycle state (`repaid` / `active` / `overdue` / `written_off` at DPD >= 180),
repayment timing (early / on_time / late), default flag and date, target `bad_30dpd`
and its observability flags.

### `marts.loan_monthly_snapshot` (loan x month-end)
Grid of loans x month-ends where the loan was on book during the month; `ASOF JOIN`
gives cumulative paid at each month-end -> balances, DPD, bucket, state, collections in
month. Current month-end is capped at the snapshot date.

### `marts.portfolio_monthly` (month)
Book size, active loans/borrowers, PAR1-90, delinquency rate, collections vs amounts
falling due, disbursements, new defaults. See `docs/metrics.md` for formulas.

### `marts.borrower_mart` (borrower, as of snapshot)
Total loans, disbursed, repaid, outstanding principal, repayments, repayment rate,
average repayment amount, repayment frequency (per 30 days), late repayments, defaults,
current and worst DPD, recency, and the rule-based `risk_segment`.

### Other marts
`roll_rates` (bucket migration), `cohort_performance` (disbursement month x product x
new/repeat), `vintage_curves` (cumulative 30+/90+ by months on book), `repayment_facts`
(repayment grain with DPD at payment, channel, region, segment),
`collections_due_cohort` (expected vs actual by due month), `loan_default_recovery`
(exposure at default and recoveries).

## Reproducibility

- Deterministic generator (seeded NumPy `Generator`); identical inputs produce identical
  warehouses (`tests/test_generator.py::test_generation_is_deterministic`).
- `CREATE OR REPLACE` everywhere: re-running a stage is idempotent.
- The injected-defect manifest (`data/raw/_manifest.json`) lets tests assert that each
  defect class is detected exactly (`tests/test_data_quality.py`).
