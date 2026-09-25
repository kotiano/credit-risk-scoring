# Final Audit

Audit of the repository against the CV claims it is meant to substantiate, performed
after a clean end-to-end rebuild. All data is synthetic; nothing here claims use by a
real lender.

## 1. CV claim → evidence

### Claim 1 - "Analyzed lending metrics - repayment rates, PAR, DPD, disbursements, collections - across loan repayment and transaction data by writing SQL queries to extract, clean, and transform borrower-level data."

| Evidence | Where |
|---|---|
| Raw extracts landed untyped; SQL typing, standardisation, two de-duplication patterns, robust outlier flags | `sql/staging/01-04_*.sql` |
| SQL cleaning with quarantine tables and reason codes | `sql/clean/01-04_*.sql` |
| DPD (contractual, loan-level, end-of-day), worst DPD, buckets via a config-generated macro | `sql/marts/02_loan_repayment_summary.sql`, `src/credit_risk/db.py` |
| Month-end balances with `ASOF JOIN`, PAR1/7/30/60/90 by month | `sql/marts/04_loan_monthly_snapshot.sql`, `05_portfolio_monthly.sql` |
| Repayment, on-time collection, completion, early/late rates; collection efficiency and gap by due cohort; collections by DPD bucket and channel | `sql/marts/07_borrower_mart.sql`, `11_collections_due_cohort.sql`, `10_repayment_facts.sql` |
| Disbursement and collection trends, roll-rate `PIVOT`, vintages, recovery, FPD | `sql/analytics/01-19_*.sql` → `reports/sql_analytics.md` |
| Borrower-level extraction (loans, disbursed, repaid, outstanding, repayments, frequency, late repayments, defaults) | `sql/marts/07_borrower_mart.sql` |
| Transaction data used for cash-flow features | `sql/features/01_loan_features.sql` (`txn_window`, `monthly_inflows`) |
| Toy-case SQL tests (PAR = 1,000/3,000, waterfall, timing, write-off) and SQL = Python parity for every month | `tests/test_sql_toy.py`, `tests/test_marts.py` |

### Claim 2 - "Segmented borrowers by risk profile and repayment behavior to support credit risk indicator monitoring and model performance tracking against actual outcomes."

| Evidence | Where |
|---|---|
| Six ordered, documented behavioural segment rules in SQL | `sql/marts/07_borrower_mart.sql`, `docs/metrics.md` |
| K-Means on scaled behavioural features, silhouette analysis, ARI/NMI and cross-tab vs rules | `src/credit_risk/segmentation/clustering.py`, `reports/segmentation_report.md` |
| Segment-level outcomes, collections and PAR | `sql/analytics/08_default_rate_by_segment.sql`, dashboard "Borrower risk" / "Collections" |
| Analytical risk bands with predicted PD vs actual bad rate per window | `sql/analytics/18_risk_band_performance.sql`, `reports/model_validation_report.md` §4 |
| Predicted vs actual by decision month; deciles; calibration; lift/gains | `sql/analytics/17_actual_vs_predicted_by_month.sql`, `modeling/evaluate.py`, dashboard "Model performance" |
| Monitoring: PSI/KS data and prediction drift, matured-outcome AUC/PR-AUC/calibration over time | `src/credit_risk/monitoring/`, `reports/monitoring_report.md` |

### Claim 3 - "Partnered with senior data scientists on feature engineering and validation testing for classification and credit-scoring model prototypes, documenting datasets, metrics definitions, and ETL logic for reproducibility."

| Evidence | Where |
|---|---|
| 51 point-in-time features in 9 groups, registry-driven | `sql/features/01_loan_features.sql`, `src/credit_risk/features/registry.py`, `docs/feature-engineering.md` |
| Leakage audit: static scan, truncation test (0 mismatches), negative control, univariate screen | `src/credit_risk/features/leakage.py`, `reports/leakage_audit.md`, `tests/test_features.py` |
| Classification prototypes (LR, RF, HGB), out-of-time split, parsimony-based selection | `src/credit_risk/modeling/train.py`, `docs/modeling.md` |
| Credit-scoring prototype: PDO score scaling and analytical bands | `src/credit_risk/modeling/scorecard.py` |
| Validation testing: ROC/PR-AUC, Gini, KS, Brier, ECE, calibration, confusion matrix, thresholds, over-fitting and temporal diagnostics, explainability, fairness | `modeling/metrics.py`, `evaluate.py`, `explain.py`, `fairness.py`, `docs/validation.md` |
| Dataset documentation | `docs/data-dictionary.md`, `docs/assumptions.md` |
| Metric definitions | `docs/metrics.md` |
| ETL logic | `docs/etl.md` |
| Reproducibility | `Makefile`, `README.md` §12, deterministic generator and single-threaded DuckDB |

## 2. Status

| Area | Status | Evidence |
|---|---|---|
| Tests | **77 passed** (~55 s) | `make test` |
| Lint | **Pass** - `ruff check` and `ruff format --check` on src, tests, dashboard, scripts, notebooks | `make lint` |
| Type checking | **Pass** - `mypy` (`disallow_untyped_defs`) on `src/credit_risk` (36 files) | `make typecheck` |
| Data validation | **Clean layer PASS** (15/15 invariants); source extract WARN - 15 defect classes found, all quarantined or flagged; 99.79% record retention | `reports/data_quality_report.md` |
| Leakage audit | **PASS** - 0 forbidden references; 0 mismatched cells over 4 cut-offs × 53 columns; negative control detected | `reports/leakage_audit.md` |
| Model validation | Champion LR: test ROC-AUC 0.784 / Gini 0.568 / KS 0.430 / PR-AUC 0.392; generalises (train-validation gap 0.008); trees over-fit (gap ~0.09); calibration ratio 0.92 test, 0.88 recent (under-prediction under stress) | `reports/model_validation_report.md` |
| Monitoring | Score PSI 0.007 (stable); 1 feature moderate drift (`account_age_days`, portfolio growth); performance drift detected via calibration | `reports/monitoring_report.md` |
| Dashboard | 8 tabs render without exceptions (Playwright check and Streamlit `AppTest`); filters recompute metrics | `tests/test_dashboard.py` |
| Reproducibility | **Verified from a fresh clone**: `make setup` → `make all` → `make test` → `make lint typecheck` all succeed; two consecutive runs produce **identical** reports | this audit |
| Secrets | None required or committed; `.env` git-ignored; `.env.example` holds non-secret settings | `.gitignore` |
| PII | Synthetic surrogate IDs only; no names, contacts or national IDs; explanations show model features only | `docs/data-dictionary.md` |
| Production claims | None. README, reports and dashboard state "synthetic" and "not a lending decision system" | README |

## 3. Skeptical-reviewer checklist

| Reviewer | Question | Answer |
|---|---|---|
| SQL reviewer | Can this person write SQL? | 49 SQL files: window functions, `FILTER`, `ASOF JOIN`, `PIVOT`, CTE pipelines, macros; point-in-time feature SQL; tested against hand-computed cases. |
| Data scientist | Feature engineering and validation? | Point-in-time contract proven by a truncation test with a negative control; out-of-time windows; selection on validation only; calibration and over-fitting diagnostics. |
| Credit-risk analyst | DPD, PAR, repayment, collections? | Contractual vs collection delinquency distinguished; PAR denominator and write-off treatment explicit; aligned buckets; maturity-adjusted default rates; roll rates, vintages, recoveries, FPD. |
| ML engineer | Reproducible pipeline? | One command per stage; deterministic outputs; artifacts separated from code; fresh-clone verified. |
| Senior engineer | Maintainable? | Typed package, central config, declarative checks, registry-driven features, structured logging, no logic in notebooks, 77 tests. |
| Hiring manager | Genuine analysis, not a tutorial? | Findings are non-trivial and honest: PSI stable while calibration degrades; trees over-fit; on-book delinquency skew explained; silhouette shows no natural cluster structure. |
| Interviewer | Can every decision be explained? | Decisions and alternatives are tabled in `docs/architecture.md`, `docs/modeling.md`, `docs/metrics.md`. |

## 4. Known limitations

See `docs/limitations.md`. Most important: synthetic data (performance is not
real-world evidence); single-instalment loans; monthly roll rates; hindsight segments;
point-estimate fairness metrics; no automated recalibration; DuckDB-specific SQL.

## 5. Remaining technical debt

- The dashboard (`dashboard/app.py`) is a single ~1,100-line script; splitting tabs into
  modules would help as it grows. It is linted but not type-checked.
- `evaluate.py` recomputes permutation importance for all three models (~40 s, mostly the
  400-tree random forest); caching or fewer repeats would speed iteration.
- Notebook outputs carry execution metadata, so re-running them produces diffs even when
  results are unchanged.
- No CI workflow is configured; `make check` is the local equivalent.
