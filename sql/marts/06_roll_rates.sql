-- =====================================================================================
-- marts.roll_rates
-- Source : marts.loan_monthly_snapshot
-- Grain  : one row per (from_month, from_state, to_state)
-- Purpose: month-over-month migration of on-book loans between DPD buckets.
--   from_state : DPD bucket at month-end M (on-book loans only)
--   to_state   : DPD bucket at month-end M+1, or 'repaid' / 'written_off'
--   Roll rate (count basis) = loans moving from A to B / loans in A at M.
--   Balance basis uses outstanding principal at M.
-- The current (incomplete) month has no M+1 and is therefore excluded as a "from" month.
-- =====================================================================================
CREATE OR REPLACE TABLE marts.roll_rates AS
WITH sequenced AS (
    SELECT
        loan_id,
        month_end,
        dpd_bucket,
        loan_state,
        outstanding_principal,
        lead(month_end)  OVER w AS next_month_end,
        lead(dpd_bucket) OVER w AS next_bucket,
        lead(loan_state) OVER w AS next_state
    FROM marts.loan_monthly_snapshot
    WINDOW w AS (PARTITION BY loan_id ORDER BY month_end)
)
SELECT
    month_end                                       AS from_month,
    dpd_bucket                                      AS from_state,
    CASE
        WHEN next_state IN ('repaid', 'written_off') THEN next_state
        ELSE next_bucket
    END                                             AS to_state,
    count(*)                                        AS n_loans,
    sum(outstanding_principal)                      AS principal
FROM sequenced
WHERE loan_state IN ('active', 'overdue')
  AND next_month_end IS NOT NULL
GROUP BY ALL;
