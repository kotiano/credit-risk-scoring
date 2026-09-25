# Metric Dictionary

All thresholds come from `RiskDefinitions` in `src/credit_risk/config.py` and reach SQL as
DuckDB variables/macros (`getvariable('snapshot_date')`, `dpd_bucket(dpd)`), so SQL and
Python share one definition. `tests/test_sql_toy.py` and `tests/test_marts.py` assert
SQL/Python agreement.

**Conventions**

- *As-of date*: balances are measured at the **end of day** (payments dated D count on D).
- *Settled*: outstanding total balance <= settlement tolerance (1.00). Residues below
  this are treated as paid (rounding / small-change residue).
- *Waterfall*: repayments settle **fees, then interest, then principal**.
- *On book*: loan state `active` (not yet due) or `overdue`; excludes `repaid` and
  `written_off`.
- *Refresh*: every metric is recomputed on each pipeline run for the configured snapshot
  date (monthly metrics are recomputed for every month-end).

---

## Delinquency

### DPD - days past due

| | |
|---|---|
| **Definition** | Contractual, loan-level days past due: how many days the loan's single scheduled instalment has been unpaid. |
| **Formula** | `DPD(d) = d - due_date` if the loan is unsettled at end of day `d` and `d > due_date`, else `0`. |
| **Type** | *Contractual* delinquency at *account (loan)* level. Borrower-level DPD, where shown, is the **max** over the borrower's on-book loans (`borrower_mart.current_dpd`). *Collection* delinquency (days since last payment) is **not** used and never mixed in. |
| **Data source** | `clean.loans`, `clean.repayments` |
| **SQL** | `sql/marts/02_loan_repayment_summary.sql` (`dpd`, `max_dpd_to_date`), `04_loan_monthly_snapshot.sql` (per month-end), features re-compute it as of each decision date |
| **Python** | `credit_risk.analytics.metrics.days_past_due` |
| **Grain** | loan x as-of date |
| **Limitations** | Loans are single-instalment (bullet); an instalment schedule would need per-instalment DPD (oldest unpaid instalment). Partial payments do not reset DPD. |

`max_dpd_to_date = max(0, min(payoff_date, as_of) - due_date)` is the worst DPD a loan
ever reached - used for the target and for default.

### DPD buckets

`current (0)`, `1-6`, `7-29`, `30-59`, `60-89`, `90+`. Buckets are **lower-inclusive and
aligned to the PAR thresholds**, so PAR_n is exactly the principal share of the buckets
at or above n. (The common "1-30 / 31-60" labelling puts a loan at exactly 30 DPD in a
bucket below PAR30's threshold; aligning avoids that inconsistency.) SQL macro
`dpd_bucket()`; Python `metrics.dpd_bucket`.

### PAR1 / PAR7 / PAR30 / PAR60 / PAR90 - Portfolio at Risk

| | |
|---|---|
| **Definition** | Share of outstanding principal on loans that are at least *n* days past due. |
| **Formula** | `PAR_n = Σ outstanding_principal (on-book loans with DPD >= n) / Σ outstanding_principal (all on-book loans)` |
| **Denominator** | **All on-book loans** - current, not-yet-due and delinquent - excluding settled and **written-off** loans (DPD >= 180). Written-off loans leave both numerator and denominator, so charge-offs cannot mechanically inflate PAR. |
| **Numerator** | Outstanding **principal** (after the waterfall), not total balance: interest and fees are excluded. |
| **Worked example** | Loan A: 1,000 at 35 DPD; loan B: 2,000 current -> PAR30 = 1,000 / 3,000 = **33.33%** (`tests/test_metrics_toy.py`). |
| **Data source** | `marts.loan_monthly_snapshot` |
| **SQL** | `sql/marts/05_portfolio_monthly.sql` (`par1` ... `par90`), `sql/analytics/03_par_by_month.sql` |
| **Python** | `metrics.portfolio_at_risk(loans, n)` |
| **Grain** | portfolio (or any filtered segment) x month-end |
| **Limitations** | Short-tenor books turn over fast, so the on-book stock is skewed toward delinquent loans; read PAR together with vintage bad rates. |

### Delinquency rate / average, median, max DPD

- **Delinquency rate** = on-book loans with DPD > 0 / on-book loans (count basis).
- **Average / median / max DPD** over delinquent on-book loans.
- SQL `sql/analytics/05_delinquency_summary.sql`; Python `metrics.delinquency_stats`.

### Roll rate

| | |
|---|---|
| **Definition** | Share of loans in DPD bucket A at month-end M that are in state B at month-end M+1. |
| **Formula** | `roll(A -> B) = #loans(A at M and B at M+1) / #loans(A at M)`; B includes `repaid` and `written_off`. Balance-based variant uses outstanding principal at M (`marts.roll_rates.principal`). |
| **SQL** | `sql/marts/06_roll_rates.sql`, matrix in `sql/analytics/11_roll_rate_matrix.sql` (`PIVOT`) |
| **Python** | `metrics.roll_rate_matrix` |
| **Grain** | from-month x from-bucket x to-state |
| **Limitations** | Monthly granularity lets short loans skip buckets (e.g. current -> 7-29). The current (incomplete) month has no M+1 and is excluded. |

---

## Repayment and collections

| Metric | Formula | Population | SQL | Python |
|---|---|---|---|---|
| **Repayment rate** | Σ repaid to date / Σ total due | loans with `due_date <= as_of` (matured) | `07_borrower_mart.sql`, `16_repayment_behaviour_by_product.sql` | `metrics.repayment_rate` |
| **On-time collection rate** | Σ repaid on/before due date / Σ total due | matured loans | `02_loan_repayment_summary.sql` (`paid_by_due_date`) | `metrics.on_time_collection_rate` |
| **Collection rate (period)** | cash collected in month / amount falling due in month | month | `05_portfolio_monthly.sql` | dashboard |
| **Collection efficiency** (by due cohort) | collected by due date / within 30d / to date ÷ expected | loans due in month | `11_collections_due_cohort.sql`, `09_collection_efficiency_by_product.sql` | dashboard |
| **Collection gap** | expected collections - collected to date | due cohort | `11_collections_due_cohort.sql` | dashboard |
| **Repayment completion rate** | settled loans / matured loans | matured loans | `16_repayment_behaviour_by_product.sql` | `metrics.completion_rate` |
| **Early repayment rate** | payoff <= due - 3 days, among settled loans | settled loans | `02_loan_repayment_summary.sql` (`repayment_timing`) | `metrics.timing_rates` |
| **Late repayment rate** | payoff after due date, among settled loans | settled loans | same | same |

Notes: the period collection rate can exceed 100% when arrears from earlier months are
recovered; due-cohort efficiency cannot. Collections by DPD bucket at payment, channel
and segment come from `marts.repayment_facts`.

---

## Credit quality

### Default rate

| | |
|---|---|
| **Definition** | Share of loans that reached **90 DPD** (Basel-style default). |
| **Formula** | `#loans with max_dpd_to_date >= 90 / #loans with due_date + 90 <= as_of` |
| **Why the maturity filter** | A loan disbursed last month cannot yet be 90 DPD; counting it as "not defaulted" biases the rate down. Only loans whose 90-day window has elapsed are in the denominator. |
| **SQL** | `is_default`, `is_default_observable` in `02_loan_repayment_summary.sql`; `08_cohort_performance.sql` |
| **Python** | `metrics.default_rate` |
| **Grain** | loan (aggregated to any cohort) |

### Bad rate (30+ DPD) - the model target

`bad_30dpd = 1` if the loan reached 30 DPD (unsettled on `due_date + 30`); observable only
when `due_date + 30 <= as_of`. Python `metrics.bad_rate`. See `docs/modeling.md`.

### First-payment default (FPD30)

Borrower's **first** loan reaches 30 DPD. Loans are single-instalment, so the first
payment is the only scheduled one. SQL `sql/analytics/15_first_payment_default.sql`.

### Recovery rate

| | |
|---|---|
| **Definition** | Share of the exposure at default recovered afterwards. |
| **Formula** | `recovered_amount / exposure_at_default`, where `exposure_at_default = total_due - paid by default_date` and `recovered_amount` = repayments after `default_date` (= due_date + 90). |
| **SQL** | `sql/marts/03_loan_default_recovery.sql`, `sql/analytics/14_recovery_rate.sql` |
| **Grain** | defaulted loan, aggregated by default quarter / product |
| **Limitations** | Recoveries are still accruing for recent defaults (right-censored); no collection costs are netted. |

### Repeat borrowing

Loan sequence per borrower (`borrower_loan_sequence`); outcomes by sequence in
`sql/analytics/07_repeat_borrower_performance.sql`.

---

## Borrower segmentation rules

Evaluated top-down on `marts.borrower_mart` as of the snapshot (first match wins):

| Order | Segment | Rule |
|---|---|---|
| 1 | Default / Recovery | any loan reached 90+ DPD |
| 2 | High Risk / Persistent Delinquency | current DPD >= 30, or >= 2 loans reached 30+ DPD |
| 3 | Emerging Delinquency | current DPD 1-29, or latest settled loan reached >= 7 DPD |
| 4 | Moderate Risk | ever >= 7 DPD, or > 1/3 of settled loans repaid late |
| 5 | Low Risk / New Borrower | <= 1 settled loan |
| 6 | Low Risk / Strong Repayer | everyone else |

Labels are analytical conventions, not established truths; outcome rates by segment are
descriptive (segments are assigned with hindsight). K-Means comparison:
`reports/segmentation_report.md`.

## Model metrics

Defined in `src/credit_risk/modeling/metrics.py` (module docstring explains why each
matters) and `docs/validation.md`: ROC-AUC, Gini, KS, PR-AUC, Brier, log loss,
calibration ratio, ECE, precision, recall/sensitivity, specificity, F1, lift, gains, PSI.
