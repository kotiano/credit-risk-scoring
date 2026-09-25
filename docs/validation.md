# Validation Approach

Code: `src/credit_risk/modeling/evaluate.py`, `metrics.py`. Output:
`reports/model_validation_report.md`, `artifacts/evaluation/evaluation.json`, dashboard
"Model performance" tab.

## What is validated, and why each metric matters

| Question | Metric | Why |
|---|---|---|
| Does the model rank risk? | ROC-AUC, Gini = 2·AUC - 1, KS | Threshold-free discrimination; Gini and KS are the credit-industry conventions |
| Does it find the minority class? | PR-AUC (average precision) | With ~12-16% bads, ROC-AUC can look good while precision is poor; PR-AUC's baseline equals the bad rate |
| Are the probabilities right? | Brier score, log loss, calibration ratio (mean PD / bad rate), ECE, calibration table | PDs feed pricing, expected loss and risk bands; a model can rank well and still be mis-calibrated |
| What happens at a cut-off? | Precision, recall (sensitivity), specificity, F1, confusion matrix, approval rate, bad rate among approved | Operating trade-offs once a threshold is chosen |
| Is value concentrated? | Decile table, lift, cumulative gains | How much risk the riskiest x% of applications contain |
| Does it generalise? | Train vs validation vs test vs recent | Over-fitting (train >> validation) and temporal degradation (recent < validation) |
| Is it fair? | Per-group calibration, AUC, flag rate, TPR/FPR | Proxy effects from excluded sensitive attributes |

Accuracy is not used: a model predicting "good" for everyone would score ~87%.

## Protocol

1. Out-of-time windows (see `docs/modeling.md`); all tuning on validation.
2. Test window scored once; no decision uses it.
3. Every metric is reported for every model and every window, so over-fitting and
   drift are visible side by side.
4. Conclusions in the report are **generated from the numbers** (`_conclusions` in
   `evaluate.py`), so narrative and metrics cannot contradict each other.

## Current results (seed 42, synthetic)

| Model | Train AUC | Validation AUC | Test AUC | Recent AUC | Diagnosis |
|---|---|---|---|---|---|
| Logistic Regression (champion) | 0.785 | 0.777 | 0.784 | 0.764 | generalises (gap 0.008) |
| Random Forest | 0.853 | 0.766 | 0.779 | 0.768 | over-fits (gap 0.087) |
| Hist Gradient Boosting | 0.853 | 0.764 | 0.782 | 0.770 | over-fits (gap 0.089) |

Champion on the test window: Gini 0.568, KS 0.430, PR-AUC 0.392 (bad rate 14.4%), Brier
0.105. The riskiest decile's actual bad rate is 46.8% vs 1.4% in the safest; the top two
deciles capture 52.1% of bads. Calibration ratio falls from 1.00
(train) to 0.92 (test) and 0.88 (recent): the model under-predicts during the synthetic
late-2025 stress while rank-ordering holds.

Exact, current figures are always in `reports/model_validation_report.md`.

## Interpretation for a model-risk reviewer

- Rank-ordering is stable across windows -> the model is fit for relative risk ranking.
- The level shift (calibration < 1) is the actionable finding: recalibrate the intercept
  (or Platt-scale) on recent matured outcomes and re-validate; do not redevelop unless
  AUC also degrades over several matured months.
- These are synthetic results that demonstrate the process, not evidence of real-world
  performance. A high AUC on synthetic data only means the generator put signal where
  the features can see it.
