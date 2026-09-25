# Model Monitoring

Code: `src/credit_risk/monitoring/`. Output: `reports/monitoring_report.md`,
`monitoring.*` tables, dashboard "Monitoring" tab.

## Populations

| Window | Definition | Outcomes known? |
|---|---|---|
| Reference | Training window (decisions <= 2025-03-31) | Yes |
| Validation / test | Out-of-time windows | Yes |
| Current | Decisions after 2025-09-30 (matured `recent` + `unobserved`) | Partly |

## Three kinds of drift

| Kind | Measure | Available | Thresholds |
|---|---|---|---|
| **Data drift** | PSI per feature (reference-decile bins; categories for categoricals), two-sample KS statistic and p-value, reference vs current means | Immediately | PSI < 0.10 stable, 0.10-0.25 moderate, > 0.25 significant |
| **Prediction drift** | PSI and KS of the champion PD; mean PD vs reference; score PSI by decision month | Immediately | same |
| **Performance drift** | ROC-AUC, PR-AUC, Brier and calibration ratio by window and by decision month | Only for **matured** loans | Investigate a sustained fall in AUC; recalibrate when the calibration ratio leaves ~0.9-1.1 |

**Maturity matters.** `bad_30dpd` is only known at `due_date + 30`. For a 90-day SME loan
that is four months after disbursement, so the latest months always have partial
outcomes (`maturity_share` in the monthly table). Performance for months with fewer than
100 matured loans is not reported rather than reported noisily.

PSI formula: `PSI = Σ (a_i - e_i) · ln(a_i / e_i)` with empty bins floored at 1e-4.

## What the current run shows (synthetic)

- Score PSI current vs train ~0.01 (stable); one feature (`account_age_days`) drifts
  moderately because the book grows with newer customers - expected and benign.
- Calibration ratio falls to ~0.88 on the most recent matured loans while ROC-AUC
  stays within ~0.02 of training. The late-2025 stress in the generator raised the
  **level** of risk (income shocks, macro effect) without shifting the inputs much.
- This is the key monitoring lesson: **stable PSI does not mean a healthy model**.
  Input-drift metrics would not have caught this; matured-outcome calibration did.
- Proportionate response: recalibrate on recent matured loans; keep monitoring AUC.
- The early months show high score PSI because almost every borrower is new at portfolio
  launch (no history features) - a composition effect, not model decay.
