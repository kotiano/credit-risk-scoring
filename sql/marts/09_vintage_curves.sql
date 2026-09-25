-- =====================================================================================
-- marts.vintage_curves
-- Source : marts.loan_repayment_summary
-- Grain  : one row per (disbursement_month, months_on_book)
-- Purpose: cumulative share of each monthly vintage that has reached 30+ DPD and
--          90+ DPD (default) by each month-on-book (MOB). A loan "reaches 30 DPD" on
--          due_date + 30 if it is still unsettled then. Only fully observed MOB points
--          (MOB month-end <= snapshot) are emitted, so late vintages have shorter curves.
-- =====================================================================================
CREATE OR REPLACE TABLE marts.vintage_curves AS
WITH loans AS (
    SELECT
        loan_id,
        disbursement_month,
        loan_amount,
        CASE WHEN max_dpd_to_date >= getvariable('target_dpd')::INTEGER
             THEN due_date + getvariable('target_dpd')::INTEGER END AS reached_30dpd_date,
        default_date,
        as_of_date
    FROM marts.loan_repayment_summary
),
grid AS (
    SELECT
        l.*,
        mob.k                                                        AS months_on_book,
        last_day(l.disbursement_month + to_months(mob.k::INTEGER))    AS mob_end
    FROM loans AS l
    CROSS JOIN range(0, 25) AS mob(k)
)
SELECT
    disbursement_month,
    months_on_book,
    mob_end,
    count(*)                                                         AS loans,
    sum(loan_amount)                                                 AS amount_disbursed,
    avg((reached_30dpd_date <= mob_end)::INTEGER)                    AS cum_bad_30dpd_rate,
    sum(loan_amount) FILTER (WHERE reached_30dpd_date <= mob_end)
        / sum(loan_amount)                                           AS cum_bad_30dpd_rate_amount,
    avg((default_date <= mob_end)::INTEGER)                          AS cum_default_rate
FROM grid
WHERE mob_end <= as_of_date
GROUP BY ALL;
