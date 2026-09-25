# Assumptions

## Synthetic data - why and how

No real lending data is used or implied. A synthetic portfolio was generated because
real borrower data is confidential, and public credit datasets (e.g. single-table
application snapshots) lack the loan -> repayment -> transaction structure needed to
demonstrate DPD, PAR, collections and point-in-time feature engineering.

The generator (`src/credit_risk/ingestion/synthetic.py`) simulates a **data-generating
process**, not independent columns:

| Mechanism | Assumption |
|---|---|
| Hidden creditworthiness | Each borrower has a latent trait `~ N(0,1)` that drives bureau defaults, income, wallet activity and repayment risk. It is never exported - the model must infer it. |
| Employment | Salaried / self-employed / informal / student / unemployed with different income levels, income volatility and baseline risk. |
| Income | Declared income is log-normal by employment and region; wallet-observed income is 55-105% of declared (declared incomes are typically overstated). |
| Wallet activity | Monthly transaction counts ~ Poisson with a borrower-specific rate correlated with lower risk; salaried borrowers receive a monthly salary credit; balances never go negative. |
| Income shocks | Each month a borrower has a ~1% chance of a 2-4 month income shock (inflows cut to 35-65%). A shock during the loan raises risk (+1.0 logit); a shock starting the month after disbursement also raises risk (+0.5) but is **not observable** at decision time - irreducible risk. |
| Macro stress | From 2025-08-01 inflows fall 7-15%, shocks become 1.5-3x more frequent (3x in Coastal and Northern) and all loans carry +0.3 logit risk. Creates realistic temporal drift. |
| Products | Salary advance (30d, 8%/30d), micro loan (14/30d, 10%), emergency loan (7d, 12%), SME working capital (60/90d, 6%); origination fees 2-3%. Bullet repayment. |
| Pricing | +2 points per 30 days for borrowers whose previous loan went 7+ DPD (uses prior history only). |
| Outcome model | Ordinal logit over {early, on time, 1-6, 7-29, 30-89 days late, default} on a latent risk that combines the borrower trait, product, loan-to-income stress, shocks, macro stress and own repayment history. |
| Behavioural memory | Good repayments lower future risk (up to -0.6 logit) and raise limits (x1.25, cap 2x); late repayments raise risk (+0.15 to +1.2) and cut limits (x0.7). |
| Re-borrowing | A new loan only after settling the previous one (plus ~10% concurrent top-ups for good borrowers); return probability falls with worse outcomes; defaulters rarely return. |
| Repayment patterns | Partial payments before settlement (15-75% of loans depending on outcome); defaults pay 0-60% before stopping; 30% of defaults get a late recovery 95-420 days after due. |
| Censoring | Nothing dated after the snapshot (2025-12-31) exists in the extract. |
| Sensitive attributes | Gender has **no** effect on risk. Age has a small effect (younger = riskier). Region has small risk/income effects and determines stress exposure. |

### Injected data-quality defects

Proportional to table size (manifest in `data/raw/_manifest.json`): duplicate borrowers,
missing and mis-keyed (x100) incomes, gender label variants; duplicate loans, swapped
disbursement/due dates, zero principal, status casing noise; exact-duplicate and re-posted
repayments, orphan repayments, non-positive amounts, pre-disbursement dates, channel label
noise; duplicate / null / future-dated / outlier transactions. Tests assert each defect
class is detected.

## Business-rule assumptions

| Area | Assumption | Where |
|---|---|---|
| DPD | Contractual, loan level, end-of-day | `docs/metrics.md` |
| Settlement tolerance | balance <= 1.00 = settled | `config.RiskDefinitions` |
| Waterfall | fees -> interest -> principal | `02_loan_repayment_summary.sql` |
| Default | 90 DPD | config |
| Write-off | 180 DPD; leaves PAR numerator and denominator | config |
| Early repayment | settled >= 3 days before due | config |
| Model target | reached 30 DPD | config |
| Snapshot | 2025-12-31 | config |
| Split dates | 2025-03-31 / 2025-06-30 / 2025-09-30 | config |
| Risk bands | PD cut-offs 4 / 8 / 15 / 30% | config |
| Currency | notional local units; no FX | - |
