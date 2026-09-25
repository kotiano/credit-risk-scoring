-- =====================================================================================
-- marts.portfolio_monthly
-- Source : marts.loan_monthly_snapshot, marts.loan_repayment_summary, clean.repayments
-- Grain  : one row per month-end
--
-- PAR_n = outstanding principal of loans with DPD >= n
--         / outstanding principal of all loans on book (DPD any, incl. not yet due)
--   * "On book" excludes settled and written-off (DPD >= write_off_dpd) loans in BOTH
--     numerator and denominator, so write-offs cannot mechanically inflate PAR.
--
-- Collection rate (period) = cash collected in month / amount falling due in month.
--   It can exceed 100% when arrears from earlier months are recovered.
-- =====================================================================================
CREATE OR REPLACE TABLE marts.portfolio_monthly AS
WITH book AS (
    SELECT
        month_end,
        count(*) FILTER (WHERE loan_state IN ('active', 'overdue'))                 AS active_loans,
        count(DISTINCT borrower_id) FILTER (WHERE loan_state IN ('active', 'overdue'))
                                                                                    AS active_borrowers,
        sum(outstanding_principal) FILTER (WHERE loan_state IN ('active', 'overdue'))
                                                                                    AS outstanding_principal,
        sum(outstanding_principal) FILTER (WHERE loan_state IN ('active', 'overdue') AND dpd >= 1)  AS par1_principal,
        sum(outstanding_principal) FILTER (WHERE loan_state IN ('active', 'overdue') AND dpd >= 7)  AS par7_principal,
        sum(outstanding_principal) FILTER (WHERE loan_state IN ('active', 'overdue') AND dpd >= 30) AS par30_principal,
        sum(outstanding_principal) FILTER (WHERE loan_state IN ('active', 'overdue') AND dpd >= 60) AS par60_principal,
        sum(outstanding_principal) FILTER (WHERE loan_state IN ('active', 'overdue') AND dpd >= 90) AS par90_principal,
        count(*) FILTER (WHERE loan_state = 'overdue')                              AS delinquent_loans,
        avg(dpd) FILTER (WHERE loan_state = 'overdue')                              AS avg_dpd_delinquent,
        sum(collected_in_month)                                                     AS collections
    FROM marts.loan_monthly_snapshot
    GROUP BY month_end
),
flows AS (
    SELECT
        last_day(disbursement_month)                                        AS month_key,
        count(*)                                                            AS loans_disbursed,
        sum(loan_amount)                                                    AS amount_disbursed,
        count(DISTINCT borrower_id)                                         AS borrowers_disbursed,
        count(*) FILTER (WHERE NOT is_first_loan)                           AS repeat_loans_disbursed
    FROM marts.loan_repayment_summary
    GROUP BY ALL
),
dues AS (
    SELECT last_day(due_month) AS month_key, sum(total_due) AS amount_falling_due,
           count(*) AS loans_falling_due
    FROM marts.loan_repayment_summary
    GROUP BY ALL
),
defaults AS (
    SELECT last_day(default_month) AS month_key, count(*) AS new_defaults,
           sum(exposure_at_default) AS new_default_exposure
    FROM marts.loan_default_recovery
    GROUP BY ALL
)
SELECT
    b.month_end,
    coalesce(f.loans_disbursed, 0)                          AS loans_disbursed,
    coalesce(f.amount_disbursed, 0)                         AS amount_disbursed,
    coalesce(f.borrowers_disbursed, 0)                      AS borrowers_disbursed,
    coalesce(f.repeat_loans_disbursed, 0)                   AS repeat_loans_disbursed,
    b.collections,
    coalesce(d.amount_falling_due, 0)                       AS amount_falling_due,
    b.collections / nullif(d.amount_falling_due, 0)         AS collection_rate,
    b.active_loans,
    b.active_borrowers,
    b.delinquent_loans,
    b.delinquent_loans / nullif(b.active_loans, 0)          AS delinquency_rate,
    b.avg_dpd_delinquent,
    coalesce(b.outstanding_principal, 0)                    AS outstanding_principal,
    coalesce(b.par1_principal, 0) / nullif(b.outstanding_principal, 0)  AS par1,
    coalesce(b.par7_principal, 0) / nullif(b.outstanding_principal, 0)  AS par7,
    coalesce(b.par30_principal, 0) / nullif(b.outstanding_principal, 0) AS par30,
    coalesce(b.par60_principal, 0) / nullif(b.outstanding_principal, 0) AS par60,
    coalesce(b.par90_principal, 0) / nullif(b.outstanding_principal, 0) AS par90,
    coalesce(df.new_defaults, 0)                            AS new_defaults,
    coalesce(df.new_default_exposure, 0)                    AS new_default_exposure
FROM book AS b
LEFT JOIN flows AS f ON f.month_key = last_day(b.month_end)
LEFT JOIN dues AS d ON d.month_key = last_day(b.month_end)
LEFT JOIN defaults AS df ON df.month_key = last_day(b.month_end)
ORDER BY b.month_end;
