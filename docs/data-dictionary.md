# Data Dictionary

All data is **synthetic** (see `docs/assumptions.md`). Amounts are in a notional local
currency unit. Dates are calendar dates (no time component). The snapshot (extract) date
is **2025-12-31**; the lending window starts **2024-01-01**; wallet history starts in
mid-2023.

## Raw extracts (`data/raw/*.csv` -> `raw.*`)

### borrowers - one row per borrower (plus injected duplicates)

| Column | Type | Description |
|---|---|---|
| borrower_id | string | `B000001`... Anonymous surrogate key; no PII |
| onboarding_date | date | Date the borrower registered for lending |
| account_open_date | date | Date the wallet account was opened (precedes onboarding) |
| age | int | Age in years **at onboarding** (sensitive - never a model input) |
| gender | string | Raw codes/words: `F`, `M`, `U`, `female`, ` m`, `unknown` (sensitive - never a model input) |
| employment_status | string | `salaried`, `self_employed`, `informal`, `student`, `unemployed` (self-declared) |
| monthly_income | float | Declared monthly income; may be missing or mis-keyed |
| region | string | `Metro`, `Central`, `Lakeside`, `Highlands`, `Coastal`, `Northern` (proxy risk - never a model input) |
| previous_loans | int | External (bureau) loans before onboarding |
| previous_defaults | int | External (bureau) defaults before onboarding |
| transaction_activity | int | Average monthly wallet transactions before onboarding |

### loans - one row per loan

| Column | Type | Description |
|---|---|---|
| loan_id | string | `L0000001`... |
| borrower_id | string | FK -> borrowers |
| product_type | string | `salary_advance` (30d), `micro_loan` (14/30d), `emergency_loan` (7d), `sme_working_capital` (60/90d) |
| loan_amount | float | Principal disbursed |
| interest_rate | float | Flat interest per 30 days (e.g. 0.10 = 10%/30d), includes risk-based uplift |
| fees | float | One-off origination fee |
| total_due | float | principal + principal × rate × term/30 + fees; single bullet instalment |
| term_days | int | Contractual term |
| disbursement_date | date | Decision / disbursement date |
| due_date | date | disbursement_date + term_days |
| loan_status | string | Source-system status at extract: `active`, `overdue`, `repaid`, `written_off` (with casing noise). **Reconciliation only.** |

### repayments - one row per payment event

| Column | Type | Description |
|---|---|---|
| repayment_id | string | `R00000001`... |
| loan_id | string | FK -> loans |
| borrower_id | string | Must equal the loan's borrower |
| repayment_date | date | Value date of the payment |
| amount | float | Amount paid (partial or full) |
| payment_channel | string | `mobile_app`, `ussd`, `agent`, `bank_transfer`, `auto_debit` (with label noise) |
| transaction_reference | string | Payment-rail reference; business key for de-duplication |

### transactions - one row per wallet transaction (lending flows excluded)

| Column | Type | Description |
|---|---|---|
| transaction_id | string | `T000000001`... |
| borrower_id | string | FK -> borrowers |
| transaction_date | date | |
| transaction_type | string | Inflows: `salary`, `deposit`, `p2p_received`; outflows: `withdrawal`, `merchant_payment`, `bill_payment`, `p2p_sent`, `airtime` |
| amount | float | Signed: positive inflow, negative outflow |
| balance | float | Wallet balance after the transaction |
| channel | string | `app`, `ussd`, `agent`, `card`, `bank` |

## Key marts

### marts.loan_repayment_summary (loan grain, as of snapshot)

| Column | Description |
|---|---|
| total_paid, n_payments, n_late_payments | Repayments to date |
| paid_by_due_date, paid_within_30d_of_due | Repayments by due date / due + 30 |
| payoff_date, is_settled | First date balance <= tolerance |
| outstanding_total, outstanding_principal | Balance and principal (waterfall) outstanding |
| dpd, dpd_bucket | Contractual DPD at snapshot |
| max_dpd_to_date, worst_dpd_bucket | Worst DPD ever reached |
| loan_state | `repaid`, `active`, `overdue`, `written_off` (derived) |
| repayment_timing, repayment_timing_days, days_to_repay | Settled loans only |
| has_partial_payment | Received a payment that did not settle it |
| is_default, default_date | Reached 90 DPD; date = due + 90 |
| is_target_observable, bad_30dpd | Model target and its observability |
| is_default_observable | due + 90 <= snapshot |
| borrower_loan_sequence, is_first_loan | Loan order per borrower |

### marts.loan_monthly_snapshot (loan x month-end)
`month_end, paid_to_date, collected_in_month, outstanding_total, outstanding_principal,
dpd, dpd_bucket, loan_state`.

### marts.portfolio_monthly (month-end)
`loans_disbursed, amount_disbursed, collections, amount_falling_due, collection_rate,
active_loans, active_borrowers, delinquent_loans, delinquency_rate, outstanding_principal,
par1, par7, par30, par60, par90, new_defaults, new_default_exposure`.

### marts.borrower_mart (borrower, as of snapshot)
`total_loans, total_disbursed, total_due, total_repaid, outstanding_principal,
active_loans, completed_loans, n_repayments, n_late_repayments, late_settled_loans,
loans_reached_30dpd, n_defaults, repayment_rate, avg_repayment_amount,
repayment_frequency_30d, late_payment_ratio, current_dpd, max_dpd_ever,
avg_days_to_repay, days_since_last_loan, days_since_last_repayment, risk_segment`, plus
region / employment / gender / age band for reporting and fairness only.

### features.loan_features / features.model_dataset
One row per application; feature definitions in `docs/feature-engineering.md` and
`src/credit_risk/features/registry.py`. `model_dataset` adds `target`,
`is_target_observable` and `sample_split` (`train`, `validation`, `test`, `recent`,
`unobserved`).

### model.loan_scores
`loan_id, sample_split, pd, score, risk_band, model_name` - champion output for every
application.

### quality.* and monitoring.*
`quality.check_results`, `quality.dataset_profile`, `quality.missingness`;
`monitoring.feature_drift`, `monitoring.prediction_drift`, `monitoring.monthly`,
`monitoring.performance`.
