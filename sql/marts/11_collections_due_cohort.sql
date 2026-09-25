-- =====================================================================================
-- marts.collections_due_cohort
-- Source : marts.loan_repayment_summary
-- Grain  : one row per (due_month, product_type) for months already fallen due
-- Purpose: expected vs actual collections for the amounts falling due in each month.
--   expected_collections      = total_due of loans due in the month
--   collected_by_due_date     = repayments on/before the due date
--   collected_within_30d      = repayments on/before due_date + 30
--   collected_to_date         = all repayments up to the snapshot
--   collection_efficiency_*   = collected_* / expected_collections
--   collection_gap            = expected_collections - collected_to_date
-- =====================================================================================
CREATE OR REPLACE TABLE marts.collections_due_cohort AS
SELECT
    due_month,
    product_type,
    count(*)                                                 AS loans_due,
    sum(total_due)                                           AS expected_collections,
    sum(paid_by_due_date)                                    AS collected_by_due_date,
    sum(paid_within_30d_of_due)                              AS collected_within_30d,
    sum(total_paid)                                          AS collected_to_date,
    sum(paid_by_due_date) / nullif(sum(total_due), 0)        AS collection_efficiency_on_time,
    sum(paid_within_30d_of_due) / nullif(sum(total_due), 0)  AS collection_efficiency_30d,
    sum(total_paid) / nullif(sum(total_due), 0)              AS collection_efficiency_to_date,
    sum(total_due) - sum(total_paid)                         AS collection_gap,
    -- 30-day efficiency is only final once due_date + 30 has passed for every loan
    max(due_date) + getvariable('target_dpd')::INTEGER <= max(as_of_date)
                                                             AS is_30d_window_complete
FROM marts.loan_repayment_summary
WHERE due_date <= as_of_date
GROUP BY ALL;
