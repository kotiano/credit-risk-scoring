-- =====================================================================================
-- marts.cohort_performance
-- Source : marts.loan_repayment_summary
-- Grain  : one row per (disbursement_month, product_type, borrower_type)
-- Purpose: compare loan cohorts on repayment, delinquency, default and current PAR.
--   Outcome rates are computed ONLY over loans whose observation window has matured
--   (e.g. bad_30dpd_rate uses loans with due_date + 30 <= snapshot) to avoid the
--   downward bias of counting immature loans as "good".
-- =====================================================================================
CREATE OR REPLACE TABLE marts.cohort_performance AS
SELECT
    disbursement_month,
    product_type,
    CASE WHEN is_first_loan THEN 'new' ELSE 'repeat' END                    AS borrower_type,
    count(*)                                                                AS loans,
    sum(loan_amount)                                                        AS amount_disbursed,
    sum(total_due)                                                          AS total_due,
    sum(total_paid)                                                         AS total_repaid,
    sum(total_paid) / nullif(sum(total_due), 0)                             AS repayment_rate_to_date,
    avg(is_settled::INTEGER)                                                AS completion_rate_to_date,
    count(*) FILTER (WHERE is_target_observable)                            AS loans_30dpd_observable,
    avg(bad_30dpd) FILTER (WHERE is_target_observable)                      AS bad_30dpd_rate,
    count(*) FILTER (WHERE is_default_observable)                           AS loans_default_observable,
    avg(is_default::INTEGER) FILTER (WHERE is_default_observable)           AS default_rate,
    coalesce(sum(outstanding_principal)
             FILTER (WHERE loan_state IN ('active', 'overdue')), 0)         AS outstanding_principal,
    coalesce(sum(outstanding_principal)
             FILTER (WHERE loan_state IN ('active', 'overdue') AND dpd >= 30), 0)
        / nullif(sum(outstanding_principal)
                 FILTER (WHERE loan_state IN ('active', 'overdue')), 0)     AS par30_current,
    sum(total_paid) FILTER (WHERE due_date <= as_of_date)
        / nullif(sum(total_due) FILTER (WHERE due_date <= as_of_date), 0)   AS collection_rate_matured
FROM marts.loan_repayment_summary
GROUP BY ALL;
