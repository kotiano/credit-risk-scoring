# Limitations and Future Work

## Data

- **Synthetic data.** Relationships exist because the generator put them there. Model
  performance (AUC ~0.78) demonstrates the *process*, not real-world predictive power,
  and must not be quoted as such.
- **Single-instalment loans.** Instalment schedules would need per-instalment DPD
  (oldest unpaid instalment), and PAR would use the loan's oldest arrears.
- **No restructuring, rescheduling, fees on arrears or penalty interest.** Balances do not
  accrue after the due date.
- **No collection costs** in recovery rates (gross recoveries).
- **Wallet feed excludes lending flows.** A real mobile-money feed would include loan
  disbursements and repayments, which would have to be removed or handled carefully to
  avoid leakage.

## Methodology

- **Monthly roll rates** on short-tenor loans skip buckets; weekly snapshots would be more
  granular for 7-30 day products.
- **Segments are assigned with hindsight** (as of the snapshot); outcome rates by segment
  are descriptive. A forward test would assign segments at month M and observe M+1..M+3.
- **Fairness review** reports point estimates. Group-level bootstrap confidence intervals
  and a formal disparity policy would be needed before any conclusion; small groups
  (e.g. `undisclosed` gender) are noisy.
- **Permutation importance** is inflated/deflated for correlated features; importances
  should be read at feature-group level.
- **No recalibration step** is automated; the monitoring report recommends it when the
  calibration ratio drifts (currently 0.88 on recent loans).
- **Reject inference** is not addressed: the synthetic lender approves everyone who
  applies, so there is no selection bias to correct - a real portfolio would have it.
- **Hyper-parameter search** is a small explicit grid (intentional, for transparency).

## Engineering

- **DuckDB-specific SQL:** `ASOF JOIN`, `GROUP BY ALL`, `PIVOT`, `FILTER`, macros and
  `getvariable`. Porting to PostgreSQL: `ASOF` -> `LEFT JOIN LATERAL (... ORDER BY date
  DESC LIMIT 1)`, variables -> a parameters table, macros -> SQL functions.
- **Single-node, batch.** No scheduler, incremental loads or late-arriving-data handling;
  every run rebuilds all tables (fast at this scale: ~80 s end to end).
- **mypy covers `src/`**; the Streamlit app and tests are linted but not type-checked.
- **Dashboard** is light-theme only and not load-tested; it reads a local DuckDB file.

## Future improvements

1. Instalment loans with per-instalment DPD and weekly snapshots.
2. Automated recalibration (Platt / intercept adjustment) triggered by the monitoring
   calibration ratio, with a champion-challenger comparison.
3. Bootstrap confidence intervals for all validation and fairness metrics.
4. Forward-looking segment validation (segment at M, outcome at M+3).
5. dbt project and PostgreSQL target for multi-user deployment.
6. Expected-loss view (PD x LGD x EAD) using the recovery mart for LGD.
