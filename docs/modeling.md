# Modeling Methodology

Code: `src/credit_risk/modeling/`. Results: `reports/model_validation_report.md`.

## Target definition

| Item | Definition |
|---|---|
| Name | `bad_30dpd` ("reaches 30 days past due") |
| Positive (bad) | The loan is still unsettled at the end of `due_date + 30` days |
| Negative (good) | Settled before `due_date + 30` (early, on time or up to 29 days late) |
| Unit | Loan application (one scoring event per loan) |
| Observation window | All borrower history up to `decision_date - 1` |
| Performance window | Disbursement -> `due_date + 30` days (7-90 day terms => 37-120 days) |
| Why 30 DPD, not 90 | 30 DPD is observable sooner (more matured, more recent training data), matches PAR30 and is the standard early-delinquency target for short-tenor lending; 90+ default is reported as a portfolio metric |

**Why not `default_next_loan`?** Only repeat borrowers would be scoreable and first loans
(33% of volume and the riskiest) would be excluded.

## Inclusion, exclusion and censoring

- **Included:** every clean loan with a complete performance window.
- **Censored:** loans with `due_date + 30 > snapshot` are labelled `unobserved`. They are
  scored (for monitoring) but never used for training or evaluation - treating them as
  "good" would bias the bad rate down.
- **Censoring side effect:** near the end of each window, long-term products mature
  later, so the latest windows slightly over-represent short products.
- **Excluded upstream:** quarantined loans (invalid terms) - see `docs/etl.md`.
- A borrower can appear in several windows (behavioural scoring); every application only
  sees its own past.

## Out-of-time split

| Split | Decision dates | Loans | Bad rate | Use |
|---|---|---|---|---|
| train | <= 2025-03-31 | 8,813 | 12.1% | fit |
| validation | 2025-04-01 - 2025-06-30 | 2,058 | 12.9% | hyper-parameters, champion, threshold |
| test | 2025-07-01 - 2025-09-30 | 2,198 | 14.4% | scored once, after all choices were frozen |
| recent | >= 2025-10-01, matured | 863 | 16.0% | monitoring of the newest matured loans |
| unobserved | performance window open | 1,188 | - | prediction/feature drift only |

A random split would put future loans in training and past loans in test, leaking
calendar effects (e.g. the late-2025 stress) and overstating performance.

## Candidate models

| Family | Preprocessing | Grid (selected on validation ROC-AUC) |
|---|---|---|
| Logistic Regression (baseline) | winsorise -> log1p (skewed) -> median impute + indicator -> standardise; one-hot | `C` in {0.01, 0.1, 1.0} |
| Random Forest (400 trees) | median impute + indicator; one-hot | `min_samples_leaf` {20, 50} x `max_features` {sqrt, 0.3} |
| HistGradientBoosting | native NaN; one-hot | 5 configs of learning rate / iterations / leaf nodes |

No class re-weighting: it would distort PDs. Imbalance (~12% bads) is handled in
evaluation (PR-AUC, threshold analysis). XGBoost/LightGBM were not added: scikit-learn's
HistGradientBoosting is the same algorithm class without an extra binary dependency, and
tree models did not beat the linear baseline out of time.

## Champion selection

1. Best configuration per family by **validation ROC-AUC**.
2. **Parsimony rule:** keep Logistic Regression unless a challenger improves validation
   ROC-AUC by more than 0.01. Rationale: interpretability (coefficients, reason codes),
   stability over time, and PDs that are calibrated by construction.
3. Reference threshold = max F1 on **validation** (never test). This is an analytical
   operating point, not a credit policy. A pricing view would use a break-even PD
   (`margin / (margin + LGD)`); see the threshold table in the validation report.

Current outcome: the best challenger (Random Forest, 0.766) is 0.011 **below** the
baseline (0.777) on validation, and both tree models over-fit (train-validation AUC gap
0.087-0.089 vs 0.008 for LR), so Logistic Regression is the champion.

## Score and risk bands

`score = offset + factor * ln((1 - PD) / PD)`, `factor = 20 / ln 2`, 600 points at 50:1
odds (points-to-double-odds = 20). Analytical bands on PD: A < 4%, B 4-8%, C 8-15%,
D 15-30%, E >= 30%. Bands are **monitoring groupings of model output**, not underwriting
policy.

## Explainability

- **Global:** standardised LR coefficients; permutation importance on validation for all
  models (primary measure); RF impurity importance (reported with its bias caveat).
  Correlated features (e.g. `income_proxy_monthly` and `loan_to_income_proxy`) share
  signal, and permutation importance can overstate each individually - read importances
  by feature group.
- **Local:** baseline substitution - each feature of one application is replaced by the
  training median/mode and the change in PD is reported (positive = raises risk). It is
  model-agnostic and auditable. SHAP was considered and not added (dependency weight,
  and for the linear champion the exact log-odds decomposition is already available from
  coefficients).

## Responsible modeling

- **Sensitive attributes:** `gender` and `age` (protected characteristics in many
  jurisdictions) and `region` (geographic proxy / redlining risk) are **excluded from the
  feature table**, not merely from the model. `employment_status` is retained as a
  legitimate, self-declared affordability attribute; it would need legal review in a real
  deployment.
- **Group review** (`modeling/fairness.py`, test window): bad rate, mean PD, calibration
  ratio, ROC-AUC, share flagged at the reference threshold, TPR and FPR per group.
- **Findings on this synthetic data:** young borrowers (18-24) are flagged about twice as
  often as average, but the model is calibrated for them (ratio ~0.97): the higher flag
  rate tracks a genuinely higher bad rate built into the generator, via proxies such as
  income and history rather than age itself. Gender differences in calibration are within
  sampling noise at these group sizes (~1,000 loans; no gender effect exists in the
  generator). Group-level confidence intervals would be required before drawing any
  conclusion.
- **This prototype is not an automated lending decision system.** It makes no approve /
  decline decisions, has no adverse-action process and has not been reviewed against any
  lending regulation.
